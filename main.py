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
from typing import Dict, Optional, List
from uuid import uuid4
import secrets
import re
import sys
import logging
import time
import urllib
import yaml

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

from http_utils import build_url, encode_multipart_formdata, request_json, download_file
from srt_functions import json_to_srt, srt_to_json
from extract_portions import extract_wave_portions

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

SEGMENT_URL = os.getenv("SEGMENT_URL", "")
SEGMENT_TOKEN = os.getenv("SEGMENT_TOKEN", "")
SEGMENT_POLL_INTERVAL = float(os.getenv("SEGMENT_POLL_INTERVAL", "3"))
SEGMENT_TIMEOUT = float(os.getenv("SEGMENT_TIMEOUT", "300"))
SEGMENT_DELTA = float(os.getenv("SEGMENT_DELTA", "0.1"))

SEGMENT_LANGS = os.getenv("SEGMENT_LANGS", "")  # comma-separated list of language codes, e.g. "en,es,fr"
if SEGMENT_LANGS:
    SEGMENT_LANGS = [lang.strip() for lang in SEGMENT_LANGS.split(",") if lang.strip()]
else:
    SEGMENT_LANGS = None

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


def _set_job_progress(job_id: str, pct: float, lang: str = None, stage: str = "transcribing", start: float = 0.0, end: float = 1.0) -> None:
    """
    Safely update job progress (0..100) and optionally the stage.

    Call it from the progress_callback, e.g.:
        progress_callback=lambda pct: _set_job_progress(job_id, pct)
    """
    # clamp + sanitize
    try:
        pct_f = float(start * 100 + pct * 100 * (end - start))
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

        # Check the SEGMENT_URL variable
        perform_transcription = True

        if job.language is not None:
            this_segment_url = SEGMENT_URL
            if SEGMENT_LANGS is not None and job.language in SEGMENT_LANGS:
                this_segment_url = os.environ.get(f"SEGMENT_URL_{job.language.lower()}", SEGMENT_URL)
                this_segment_token = os.environ.get(f"SEGMENT_TOKEN_{job.language.lower()}", SEGMENT_TOKEN)
            
            if this_segment_url:
                perform_transcription = False

                with _jobs_lock:
                    job.stage = "segmenting audio"
                    job.progress = 0.0
                
                segment_output_path = job.out_path.with_suffix(".yaml")

                body, content_type = encode_multipart_formdata({}, "wav_file", job.audio_path)
                start_url = build_url(this_segment_url, "/segment-start")
                status_url = build_url(this_segment_url, "/segment-status")
                output_url = build_url(this_segment_url, "/segment-out")
                logger.info(f"Submitting {job.audio_path} to {start_url}")
                start_payload = request_json(
                    start_url,
                    method="POST",
                    token=this_segment_token,
                    data=body,
                    content_type=content_type,
                )
                segment_job_id = start_payload["job_id"]
                logger.info(f"Job started: {segment_job_id}")

                started_at = time.monotonic()
                last_progress = None

                while True:
                    if SEGMENT_TIMEOUT and (time.monotonic() - started_at) > SEGMENT_TIMEOUT:
                        raise TimeoutError(
                            f"Timed out after {SEGMENT_TIMEOUT} seconds while waiting for job {segment_job_id} to complete"
                        )

                    query = urllib.parse.urlencode({"job_id": segment_job_id})
                    status_payload = request_json(
                        f"{status_url}?{query}",
                        method="GET",
                        token=this_segment_token,
                    )
                    status = status_payload["status"]
                    progress = status_payload.get("progress")

                    if progress != last_progress:
                        if progress is None:
                            logger.info(f"Status: {status}")
                        else:
                            logger.info(f"Status: {status} ({progress:.1f}%)")
                        last_progress = progress

                    if status == "completed":
                        break
                    if status == "error":
                        raise RuntimeError(
                            f"Segmentation job {segment_job_id} failed: {status_payload.get('error', 'unknown error')}"
                        )

                    time.sleep(max(SEGMENT_POLL_INTERVAL, 0.1))

                query = urllib.parse.urlencode({"job_id": segment_job_id})
                download_file(f"{output_url}?{query}", this_segment_token, segment_output_path)
                logger.info(f"Saved YAML output to {segment_output_path}")

                with segment_output_path.open("r", encoding="utf-8") as f:
                    segments = yaml.safe_load(f)

                with _jobs_lock:
                    job.stage = "saving segments"
                    job.progress = 0.0

                total_duration = extract_wave_portions(
                    wav_file=str(job.audio_path),
                    segments=segments,
                    output_folder=str(job.out_path.parent / f"segments"),
                    delta=SEGMENT_DELTA,
                )
                logger.info(f"Extracted audio segments to {job.out_path.parent / f'segments'}")

                with _jobs_lock:
                    job.stage = "transcribing"
                    job.progress = 0.0

                # Run transcribe_with_progress on each segment and combine results into a single .srt
                segment_results = []
                duration_until_now = 0.0
                for i, segment in enumerate(segments):
                    offset = float(segment["offset"] - SEGMENT_DELTA)
                    segment_output = job.out_path.parent / f"segments" / f"portion_{i}.wav"
                    if segment_output.is_file():
                        result = transcribe_with_progress(
                            model,
                            str(segment_output),
                            language=job.language,
                            fp16=False,
                            verbose=False,
                            progress_callback=lambda pct, lang: _set_job_progress(job_id, pct, lang=lang,
                                start=duration_until_now / total_duration,
                                end=(duration_until_now + segment["duration"]) / total_duration,
                                stage=f"transcribing segment {i+1}/{len(segments)}"),
                                # stage=f"transcribing"),
                        )
                        for seg in result.get("segments") or []:
                            seg["start"] += offset
                            seg["end"] += offset
                        segment_results.extend(result.get("segments") or [])
                        duration_until_now += segment["duration"]
                    else:
                        raise FileNotFoundError(f"Segment output not found: {segment_output}")

        if perform_transcription:
            logger.info("Starting complete transcription with Whisper model")
            result = transcribe_with_progress(
                model,
                str(job.audio_path),
                language=job.language,
                fp16=False,
                verbose=False,
                progress_callback=lambda pct, lang: _set_job_progress(job_id, pct, lang=lang),
            )

            segment_results = result.get("segments") or []
        
        srt_text = _segments_to_srt(segment_results)

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
        logger.warning("Conversion job %s failed:\n%s", job_id, err)
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
        logger.warning("Audio extraction job %s failed:\n%s", job_id, err)
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


def _get_persisted_audio_job(job_id: str) -> Optional[AudioExtractionJobInfo]:
    job_dir = JOBS_DIR / f"audio_{job_id}"
    audio_path = job_dir / "output.wav"
    if not audio_path.exists():
        return None

    return AudioExtractionJobInfo(
        status="completed",
        video_path=job_dir / "input",
        audio_path=audio_path,
        error=None,
        stage="completed",
        progress=100.0,
    )


def _get_persisted_conversion_job(job_id: str) -> Optional[JobInfo]:
    job_dir = JOBS_DIR / job_id
    out_path = job_dir / "output.srt"
    out_translated_path = job_dir / "output_translated.srt"
    if not out_path.exists() and not out_translated_path.exists():
        return None

    return JobInfo(
        status="completed",
        video_path=job_dir / "input",
        audio_path=job_dir / "audio.wav",
        out_path=out_path,
        out_translated_path=out_translated_path,
        language=None,
        translate_to="persisted" if out_translated_path.exists() else None,
        error=None,
        stage="completed",
        progress=100.0,
    )


def _get_audio_job_or_404(job_id: str) -> AudioExtractionJobInfo:
    with _audio_jobs_lock:
        job = _audio_jobs.get(job_id)
    if job is not None:
        return job

    job = _get_persisted_audio_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown id")
    return job


def _get_conversion_job_or_404(job_id: str) -> JobInfo:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job

    job = _get_persisted_conversion_job(job_id)
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
