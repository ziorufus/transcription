# Whisper Video-to-SRT API

Small FastAPI service for:
- extracting WAV audio from uploaded media files
- transcribing media to `.srt` with OpenAI Whisper
- optionally translating subtitles through an OpenAI-compatible chat/completions API

## Requirements

- Python 3.10+
- `ffmpeg` available on `PATH`
- a machine that can run Whisper locally
- optional: an OpenAI-compatible API for subtitle translation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the example file and fill in the values you need:

```bash
cp .env.example .env
```

Main variables:
- `JOBS_DIR`: directory used to store uploaded files and outputs
- `WHISPER_MODEL`: Whisper model name, default `large`
- `BATCH_SIZE`: subtitle batch size for translation requests
- `API_URL`: base URL for the translation backend
- `WEBUI_API_KEY`: API key for the translation backend
- `MODEL_NAME`: model used for subtitle translation
- `ENABLE_AUDIO`: enables audio extraction endpoints
- `ENABLE_CONVERSION`: enables transcription/translation endpoints
- `API_BEARER_TOKEN`: bearer token required by the API

If `API_BEARER_TOKEN` is not set, the app generates one at startup.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open the docs at `http://localhost:8000/docs`.

## Authentication

All processing endpoints require a bearer token:

```bash
-H "Authorization: Bearer YOUR_TOKEN"
```

The root endpoint `/` does not require authentication.

## API

### Health / metadata

- `GET /`
  - returns service name, Whisper model, and which endpoint groups are enabled

### Audio extraction

Available only when `ENABLE_AUDIO=true`.

- `POST /audio-extraction-start`
  - multipart field: `file`
  - returns `{ "id": "..." }`
- `GET /audio-extraction-status?id=...`
  - returns job status, stage, and progress
- `GET /audio-extraction-out?id=...`
  - downloads `{id}.wav` when the job is complete

### Transcription / conversion

Available only when `ENABLE_CONVERSION=true`.

- `POST /conversion-start`
  - multipart field: `file`
  - query params:
    - `source`: source language code, or `auto`
    - `target`: target language code for translated subtitles
  - returns `{ "id": "..." }`
- `GET /conversion-status?id=...`
  - returns job status, stage, and progress
- `GET /conversion-lang?id=...`
  - returns the detected or selected source language code
- `GET /conversion-out?id=...`
  - downloads `{id}.srt`
- `GET /conversion-translated?id=...`
  - downloads `{id}_translated.srt` if translation was requested and produced

## Example

Start a transcription job:

```bash
curl -X POST "http://localhost:8000/conversion-start?source=auto&target=it" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/path/to/video.mp4"
```

Check progress:

```bash
curl "http://localhost:8000/conversion-status?id=JOB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Download the SRT:

```bash
curl -L "http://localhost:8000/conversion-out?id=JOB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o output.srt
```

Download the translated SRT:

```bash
curl -L "http://localhost:8000/conversion-translated?id=JOB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o output_translated.srt
```

## Notes

- `ffmpeg` is required for both audio extraction and transcription jobs.
- The service keeps jobs in memory, so active job state is lost on restart.
- Whisper is loaded lazily on the first transcription job.
- `main.py` currently loads Whisper on `device="mps"`, so this setup is primarily aimed at Apple Silicon.
- Translation is skipped when `target` matches the detected source language.
