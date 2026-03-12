# main.py
import os
import shutil
import subprocess
import threading
import traceback
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Sequence
from uuid import uuid4
import secrets
import re
import sys
import logging

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

# Whisper (https://github.com/openai/whisper#python-usage)
import whisper
from whisper_progress import transcribe as transcribe_with_progress  # your edited file

import pycountry
from openai import OpenAI

# Load .env if present (does nothing if missing)
load_dotenv()


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


APP_NAME = "video-to-srt-api"
DEFAULT_BATCH_SIZE = 20
JOBS_DIR = Path(os.getenv("JOBS_DIR", "./jobs")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "large")
BATCH_SIZE = os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)
API_URL = os.getenv("API_URL")
WEBUI_API_KEY = os.environ.get("WEBUI_API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME")
ENABLE_AUDIO = _env_flag("ENABLE_AUDIO", default=True)
ENABLE_CONVERSION = _env_flag("ENABLE_CONVERSION", default=True)

_TS_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$"
)
_TIMING_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s+(?P<settings>.*))?$"
)

# --- Bearer auth ---
bearer_scheme = HTTPBearer(auto_error=False)

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
if not API_BEARER_TOKEN:
    # Random token at startup if not set (print it so you can copy it)
    API_BEARER_TOKEN = secrets.token_urlsafe(32)
    logger.info(f"[{APP_NAME}] Generated API_BEARER_TOKEN:\n{API_BEARER_TOKEN}\n")

# def require_bearer_token(
#     creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
# ) -> None:
#     if creds is None or creds.scheme.lower() != "bearer":
#         raise HTTPException(status_code=401, detail="Missing Bearer token")
#     if creds.credentials != API_BEARER_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid token")

def require_bearer_token(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> None:
    if request.method == "OPTIONS":
        return  # allow CORS preflight through

    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if creds.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


@dataclass
class JobInfo:
    status: str
    video_path: Path
    audio_path: Path
    out_path: Path
    out_translated_path: Path
    language: Optional[str] = None
    translate_to: Optional[str] = None
    error: Optional[str] = None
    stage: str = "queued"          # "extracting" | "transcribing" | "writing" | ...
    progress: float = 0.0          # 0..100


@dataclass
class AudioExtractionJobInfo:
    status: str
    video_path: Path
    audio_path: Path
    error: Optional[str] = None
    stage: str = "queued"          # "extracting" | "completed" | ...
    progress: float = 0.0          # 0..100

OPENAPI_TAGS = [{"name": "general", "description": "Service metadata endpoints."}]
if ENABLE_AUDIO:
    OPENAPI_TAGS.append(
        {"name": "audio-extraction", "description": "Extract WAV audio tracks from uploaded media."}
    )
if ENABLE_CONVERSION:
    OPENAPI_TAGS.append(
        {"name": "conversion", "description": "Transcription and subtitle conversion endpoints."}
    )

app = FastAPI(title=APP_NAME, openapi_tags=OPENAPI_TAGS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # keep False if you use Authorization header (typical)
    allow_methods=["*"],
    allow_headers=["*"],      # must allow "Authorization"
    expose_headers=["Content-Disposition"],  # useful for FileResponse downloads
)

client = OpenAI(
    api_key=WEBUI_API_KEY,
    base_url=API_URL,
)

logger = logging.getLogger("uvicorn.error")

_jobs: Dict[str, JobInfo] = {}
_jobs_lock = threading.Lock()
_audio_jobs: Dict[str, AudioExtractionJobInfo] = {}
_audio_jobs_lock = threading.Lock()

# Cache the model in memory (loaded on first job). Loading can be slow.
_model_lock = threading.Lock()
_model = None


def _set_job_progress(job_id: str, pct: float, lang: str = None, stage: str = "transcribing") -> None:
    """
    Safely update job progress (0..100) and optionally the stage.

    Call it from the progress_callback, e.g.:
        progress_callback=lambda pct: _set_job_progress(job_id, pct)
    """
    # clamp + sanitize
    try:
        pct_f = float(pct)
    except (TypeError, ValueError):
        return

    if pct_f != pct_f:  # NaN check
        return

    pct_f = max(0.0, min(100.0, pct_f))

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.progress = pct_f
        if lang:
            job.language = lang
        if stage:
            job.stage = stage


def _ensure_ffmpeg_available() -> None:
    """Raise HTTPException if ffmpeg is not available on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    except Exception:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH "
            "(e.g., apt install ffmpeg / brew install ffmpeg)."
        )


def _extract_audio_to_wav(video_path: Path, audio_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDERR:\n{proc.stderr.strip()}")


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        # Don't fail the job just because cleanup failed
        pass


def lang_code_to_name(code: str) -> str:
    """
    Convert a 2-letter ISO-639-1 language code (e.g. 'en')
    into the full language name (e.g. 'English').
    """
    language = pycountry.languages.get(alpha_2=code.lower())

    if language is None:
        raise ValueError(f"Unknown language code: {code}")

    return language.name


def _ts_to_ms(ts: str) -> int:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts!r}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms = int(m.group("ms"))
    return (((h * 60 + mi) * 60) + s) * 1000 + ms


def _ms_to_ts(total_ms: int) -> str:
    if total_ms < 0:
        raise ValueError("Timestamp cannot be negative")
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h > 99:
        raise ValueError("SRT hours exceed 99 (unsupported by this formatter)")
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_to_json(srt_text: str, *, as_string: bool = False) -> List[Dict[str, Any]] | str:
    """
    Convert SRT text into a JSON-friendly list of dicts.

    Output schema (per cue):
      {
        "id": int,                       # subtitle index in file
        "start": "HH:MM:SS,mmm",
        "end": "HH:MM:SS,mmm",
        "start_ms": int,
        "end_ms": int,
        "settings": str | null,          # any trailing SRT timing settings
        "text": str                      # cue text (may contain newlines)
      }

    If as_string=True, returns a pretty-printed JSON string; otherwise returns Python objects.
    """
    # Normalize newlines and strip BOM if present
    text = srt_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    # Split on blank lines (one or more)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.split("\n") if ln.strip() != "" or True]
        if len(lines) < 2:
            continue

        # Parse id
        idx_line = lines[0].strip()
        if not idx_line.isdigit():
            raise ValueError(f"Invalid SRT index line: {idx_line!r}")
        cue_id = int(idx_line)

        # Parse timing line
        timing_line = lines[1].strip()
        tm = _TIMING_RE.match(timing_line)
        if not tm:
            raise ValueError(f"Invalid SRT timing line: {timing_line!r}")

        start = tm.group("start")
        end = tm.group("end")
        settings = tm.group("settings")

        # Remaining lines are text (can be multiple lines)
        cue_text = "\n".join(lines[2:]).rstrip()

        cue = {
            "id": cue_id,
            "start": start,
            "end": end,
            "start_ms": _ts_to_ms(start),
            "end_ms": _ts_to_ms(end),
            "settings": settings if settings else None,
            "text": cue_text,
        }
        out.append(cue)

    # Preserve original ordering; also useful to sort if needed:
    out.sort(key=lambda d: d["id"])

    if as_string:
        return json.dumps(out, ensure_ascii=False, indent=2)
    return out


def json_to_srt(items: Sequence[Dict[str, Any]] | str) -> str:
    """
    Convert JSON (list of dicts or a JSON string) back into SRT text.

    Accepts items with either:
      - "start"/"end" as SRT timestamps ("HH:MM:SS,mmm"), or
      - "start_ms"/"end_ms" as integers (ms)
    Optional: "settings" appended after timing arrow.
    Required: "id" (int), "text" (str).
    """
    if isinstance(items, str):
        items = json.loads(items)

    # Basic validation + normalization
    cues: List[Dict[str, Any]] = []
    for it in items:  # type: ignore[assignment]
        if not isinstance(it, dict):
            raise ValueError("Each JSON item must be an object/dict")

        if "id" not in it or "text" not in it:
            raise ValueError("Each item must contain 'id' and 'text'")

        cue_id = int(it["id"])
        cue_text = "" if it["text"] is None else str(it["text"])

        settings = it.get("settings")
        settings_str = f" {settings}".rstrip() if settings else ""

        if "start" in it and "end" in it:
            start = str(it["start"]).strip()
            end = str(it["end"]).strip()
            # validate format
            _ = _ts_to_ms(start)
            _ = _ts_to_ms(end)
        elif "start_ms" in it and "end_ms" in it:
            start = _ms_to_ts(int(it["start_ms"]))
            end = _ms_to_ts(int(it["end_ms"]))
        else:
            raise ValueError("Each item must contain either ('start','end') or ('start_ms','end_ms')")

        cues.append(
            {
                "id": cue_id,
                "start": start,
                "end": end,
                "settings_str": settings_str,
                "text": cue_text,
            }
        )

    cues.sort(key=lambda d: d["id"])

    parts: List[str] = []
    for c in cues:
        parts.append(str(c["id"]))
        parts.append(f"{c['start']} --> {c['end']}{c['settings_str']}".rstrip())
        # Keep text lines as-is; SRT expects newline separation
        parts.append(c["text"].rstrip("\n"))
        parts.append("")  # blank line between cues

    return "\n".join(parts).rstrip() + "\n"


def _format_srt_timestamp(seconds: float) -> str:
    # SRT uses comma for milliseconds: HH:MM:SS,mmm
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hh = millis // 3_600_000
    mm = (millis % 3_600_000) // 60_000
    ss = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_timestamp(float(seg["start"]))
        end = _format_srt_timestamp(float(seg["end"]))
        text = (seg.get("text") or "").strip()
        # SRT blocks:
        # index
        # start --> end
        # text
        # blank line
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text if text else "[inaudible]")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = whisper.load_model(DEFAULT_MODEL_NAME, device="mps")
            dev = next(_model.parameters()).device
            logger.info(f"Whisper model param device: {str(dev)}")
    return _model


def _run_job(job_id: str) -> None:
    """
    Background worker:
    - ffmpeg: video -> wav
    - whisper: wav -> segments
    - write .srt
    """
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return  # job deleted / not found

        with _jobs_lock:
            job.stage = "extracting audio"
            job.progress = 0.0

        _extract_audio_to_wav(job.video_path, job.audio_path)

        with _jobs_lock:
            job.stage = "loading model"
            job.progress = 0.0
        model = _load_model()

        with _jobs_lock:
            job.stage = "transcribing"
            job.progress = 0.0

        # fp16 is only meaningful on GPUs; on CPU it can be False safely.
        # result = model.transcribe(
        #     str(job.audio_path),
        #     language=job.language,
        #     fp16=False,
        #     verbose=False,
        # )
        result = transcribe_with_progress(
            model,
            str(job.audio_path),
            language=job.language,
            fp16=False,
            verbose=False,
            progress_callback=lambda pct, lang: _set_job_progress(job_id, pct, lang=lang),
        )

        segments = result.get("segments") or []
        srt_text = _segments_to_srt(segments)

        job.out_path.write_text(srt_text, encoding="utf-8")
        job.language = result.get("language")

        if job.translate_to:
            if job.translate_to != job.language:
                with _jobs_lock:
                    job.stage = "translating"
                    job.progress = 0.0

                o = srt_to_json(job.out_path.read_text(encoding="utf-8"))
                input_lang_str = lang_code_to_name(job.language)
                output_lang_str = lang_code_to_name(job.translate_to)

                sentences = []

                total = len(o)
                for start in range(0, total, BATCH_SIZE):
                    batch = o[start : start + BATCH_SIZE]

                    list_of_strings = "\n".join(
                        f"[{el['id']}] {el['text']}" for el in batch
                    )
                    
                    plural = "s"
                    if len(batch) == 1:
                        plural = ""

                    prompt = f"""You are a professional {input_lang_str} ({job.language}) to {output_lang_str} ({job.translate_to}) translator.
Your goal is to accurately convey the meaning and nuances of the original {input_lang_str} text while adhering to {output_lang_str} grammar, vocabulary, and cultural sensitivities.
Produce only the {output_lang_str} translation, without any additional explanations or commentary. Please translate the follwing {input_lang_str} texts (representing subtitles) into {output_lang_str}.
Return the same exact {len(batch)} ID{plural} with translated text; do not merge or split entries.
I expect to receive a list of {len(batch)} {output_lang_str} sentence{plural}.

{list_of_strings}
"""

                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )

                    out_raw = resp.choices[0].message.content
                    out = out_raw.rsplit("<|message|>", 1)[-1]

                    out_sentences = re.findall(r"\[\d+\]\s*(.+)", out)
                    sentences.extend(out_sentences)

                    progress = (min(start + BATCH_SIZE, total) / total) * 100
                    job.progress = progress

                # if len(sentences) != len(o):
                #     raise Exception(f"The resulting list has a wrong size: {len(sentences)} instead of {len(o)}")
                if len(sentences) > len(o):
                    sentences = sentences[0:len(o)]
                if len(sentences) < len(o):
                    sentences.extend([""] * (len(o) - len(sentences)))

                i = 0
                for element in o:
                    element['text'] = sentences[i]
                    i += 1

                out_string = json_to_srt(o)

                job.out_translated_path.write_text(out_string, encoding="utf-8")

        with _jobs_lock:
            job.stage = "completed"
            job.progress = 100.0
            job.status = "completed"
            job.error = None
            logger.info(f"Process {job_id} finished")

    except Exception as e:
        err = f"{e}\n\n{traceback.format_exc()}"
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.error = err

    finally:
        # Always remove intermediate files; keep the .srt
        if job is not None:
            _safe_unlink(job.audio_path)
            _safe_unlink(job.video_path)
            # Optional: remove the job dir if it only contains the srt (or if srt missing on error)
            try:
                if job.out_path.exists():
                    # keep directory for srt
                    pass
                else:
                    shutil.rmtree(job.out_path.parent, ignore_errors=True)
            except Exception:
                pass


def _run_audio_extraction_job(job_id: str) -> None:
    job = None
    try:
        with _audio_jobs_lock:
            job = _audio_jobs.get(job_id)
        if job is None:
            return

        with _audio_jobs_lock:
            job.stage = "extracting audio"
            job.progress = 0.0

        _extract_audio_to_wav(job.video_path, job.audio_path)

        with _audio_jobs_lock:
            job.stage = "completed"
            job.progress = 100.0
            job.status = "completed"
            job.error = None
            logger.info(f"Audio extraction {job_id} finished")

    except Exception as e:
        err = f"{e}\n\n{traceback.format_exc()}"
        with _audio_jobs_lock:
            job = _audio_jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.error = err

    finally:
        if job is not None:
            _safe_unlink(job.video_path)
            try:
                if not job.audio_path.exists():
                    shutil.rmtree(job.audio_path.parent, ignore_errors=True)
            except Exception:
                pass


def _get_audio_job_or_404(job_id: str) -> AudioExtractionJobInfo:
    with _audio_jobs_lock:
        job = _audio_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown id")
    return job


def _get_conversion_job_or_404(job_id: str) -> JobInfo:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown id")
    return job


@app.get("/", tags=["general"])
def root():
    return {
        "service": APP_NAME,
        "whisper_model": DEFAULT_MODEL_NAME,
        "enable_audio": ENABLE_AUDIO,
        "enable_conversion": ENABLE_CONVERSION,
    }


#
# Audio extraction endpoints
#
if ENABLE_AUDIO:
    @app.post("/audio-extraction-start", tags=["audio-extraction"])
    async def audio_extraction_start(
        file: UploadFile = File(...),
        _: None = Depends(require_bearer_token),
    ):
        """
        Upload a video file, extract audio to wav, return job id.
        """
        try:
            _ensure_ffmpeg_available()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        job_id = str(uuid4())
        job_dir = JOBS_DIR / f"audio_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        orig_name = file.filename or "input"
        ext = Path(orig_name).suffix or ".mp4"
        video_path = job_dir / f"input{ext}"
        audio_path = job_dir / "output.wav"

        try:
            with video_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            await file.close()

        info = AudioExtractionJobInfo(
            status="running",
            video_path=video_path,
            audio_path=audio_path,
            error=None,
        )

        with _audio_jobs_lock:
            _audio_jobs[job_id] = info

        t = threading.Thread(target=_run_audio_extraction_job, args=(job_id,), daemon=True)
        t.start()

        return {"id": job_id}


    @app.get("/audio-extraction-status", tags=["audio-extraction"])
    def audio_extraction_status(id: str, _: None = Depends(require_bearer_token)):
        job = _get_audio_job_or_404(id)
        payload = {"id": id, "status": job.status, "stage": job.stage, "progress": job.progress}
        if job.status == "error":
            payload["error"] = job.error or "Unknown error"
        return JSONResponse(payload)


    @app.get("/audio-extraction-out", tags=["audio-extraction"])
    def audio_extraction_out(id: str, _: None = Depends(require_bearer_token)):
        job = _get_audio_job_or_404(id)
        if job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (current status: {job.status})",
            )

        if not job.audio_path.exists():
            raise HTTPException(status_code=500, detail="Output file missing on server")

        return FileResponse(
            path=str(job.audio_path),
            media_type="audio/wav",
            filename=f"{id}.wav",
        )


#
# Conversion endpoints
#
if ENABLE_CONVERSION:
    @app.post("/conversion-start", tags=["conversion"])
    async def conversion_start(
        file: UploadFile = File(...),
        target: str = "",
        source: str = "",
        _: None = Depends(require_bearer_token),
    ):
        """
        Upload a video file, start conversion+transcription, return job id.
        """
        try:
            _ensure_ffmpeg_available()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        translate_to = target
        language = source

        if language == "auto":
            language = ""

        job_id = str(uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        orig_name = file.filename or "input"
        ext = Path(orig_name).suffix or ".mp4"
        video_path = job_dir / f"input{ext}"
        audio_path = job_dir / "audio.wav"
        out_path = job_dir / "output.srt"
        out_translated_path = job_dir / "output_translated.srt"

        try:
            with video_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            await file.close()

        translate_to = translate_to.strip() or None
        if language:
            language = language.strip() or ""

        if not language:
            language = None

        info = JobInfo(
            status="running",
            video_path=video_path,
            audio_path=audio_path,
            out_path=out_path,
            out_translated_path=out_translated_path,
            language=language,
            error=None,
            translate_to=translate_to,
        )

        with _jobs_lock:
            _jobs[job_id] = info

        t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        t.start()

        return {"id": job_id}


    @app.get("/conversion-status", tags=["conversion"])
    def conversion_status(id: str, _: None = Depends(require_bearer_token)):
        job = _get_conversion_job_or_404(id)
        payload = {"id": id, "status": job.status, "stage": job.stage, "progress": job.progress}
        if job.status == "error":
            payload["error"] = job.error or "Unknown error"
        return JSONResponse(payload)


    @app.get("/conversion-lang", tags=["conversion"])
    def conversion_lang(id: str, _: None = Depends(require_bearer_token)):
        job = _get_conversion_job_or_404(id)
        if not job.language:
            raise HTTPException(
                status_code=409,
                detail="Language not available yet",
            )

        return job.language


    @app.get("/conversion-out", tags=["conversion"])
    def conversion_out(id: str, _: None = Depends(require_bearer_token)):
        job = _get_conversion_job_or_404(id)
        if job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (current status: {job.status})",
            )

        if not job.out_path.exists():
            raise HTTPException(status_code=500, detail="Output file missing on server")

        return FileResponse(
            path=str(job.out_path),
            media_type="application/x-subrip",
            filename=f"{id}.srt",
        )


    @app.get("/conversion-translated", tags=["conversion"])
    def conversion_translated(id: str, _: None = Depends(require_bearer_token)):
        job = _get_conversion_job_or_404(id)
        if job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (current status: {job.status})",
            )

        if not job.translate_to:
            raise HTTPException(status_code=409, detail="No translation was requested for this job")
        if job.translate_to == job.language:
            raise HTTPException(status_code=409, detail="Translation not produced because source language equals translate_to")

        if not job.out_translated_path.exists():
            raise HTTPException(status_code=500, detail="Output file missing on server")

        return FileResponse(
            path=str(job.out_translated_path),
            media_type="application/x-subrip",
            filename=f"{id}_translated.srt",
        )
