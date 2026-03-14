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

Segmentation variables:
- `SEGMENT_URL`: base URL of the segmentation service used before Whisper transcription
- `SEGMENT_TOKEN`: bearer token for the segmentation service
- `SEGMENT_POLL_INTERVAL`: polling interval in seconds while waiting for segmentation jobs
- `SEGMENT_TIMEOUT`: maximum wait time in seconds for a segmentation job, `0` disables the timeout
- `SEGMENT_DELTA`: seconds added around segmentation boundaries when extracting audio portions
- `SEGMENT_LANGS`: comma-separated list of language codes that should use language-specific segmentation endpoints
- `SEGMENT_URL_<lang>`: per-language segmentation URL override, for example `SEGMENT_URL_EN`
- `SEGMENT_TOKEN_<lang>`: per-language segmentation token override, for example `SEGMENT_TOKEN_EN`

If `API_BEARER_TOKEN` is not set, the app generates one at startup.

Segmentation behavior:
- if `SEGMENT_URL` is empty, the app transcribes the full extracted WAV directly with Whisper
- if `source` is provided and `SEGMENT_URL` is set, the app sends the WAV to the segmentation service, downloads the YAML result, extracts audio portions, and transcribes each portion
- if `SEGMENT_LANGS` includes the selected source language, the app first checks `SEGMENT_URL_<lang>` and `SEGMENT_TOKEN_<lang>` before falling back to the default `SEGMENT_URL` and `SEGMENT_TOKEN`
- segmentation is only attempted when the source language is known up front; `source=auto` continues with direct Whisper transcription

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open the docs at `http://localhost:8000/docs`.

Example `.env` snippet with segmentation enabled:

```dotenv
API_BEARER_TOKEN=replace-with-a-secure-token

SEGMENT_URL=http://127.0.0.1:8001
SEGMENT_TOKEN=replace-with-segmentation-token
SEGMENT_POLL_INTERVAL=3
SEGMENT_TIMEOUT=300
SEGMENT_DELTA=0.1

# Use a dedicated segmentation backend only for English and Italian jobs
SEGMENT_LANGS=en,it
SEGMENT_URL_EN=http://127.0.0.1:8001
SEGMENT_TOKEN_EN=replace-with-en-token
SEGMENT_URL_IT=http://127.0.0.1:8002
SEGMENT_TOKEN_IT=replace-with-it-token
```

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
  - when `source` is an explicit language and segmentation is configured through `SEGMENT_*`, the server may segment first and then transcribe segment-by-segment
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

Start a job that uses the segmentation service for a known source language:

```bash
curl -X POST "http://localhost:8000/conversion-start?source=en&target=it" \
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
- Running jobs are still kept in memory, so in-flight progress is lost on restart.
- Completed `audio-extraction-out`, `conversion-out`, and `conversion-translated` downloads can be recovered after restart if their output files are still present in `JOBS_DIR`.
- Whisper is loaded lazily on the first transcription job.
- `main.py` currently loads Whisper on `device="mps"`, so this setup is primarily aimed at Apple Silicon.
- Translation is skipped when `target` matches the detected source language.
