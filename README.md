# Translatorul — YouTube Transcriber & Translator

Downloads YouTube videos, extracts or transcribes subtitles, translates them to Romanian
(or another language) using a local Qwen2.5-7B model, and produces a hardsubbed video.

Designed to run fully offline on CPU-only hardware (e.g. an i5-12500 / 32GB server).

## Run with Docker (recommended)

```bash
docker compose pull
docker compose up -d
```

Open `http://<server-ip>:8000`, paste a YouTube URL, pick a target language, and start.

On first run the container downloads the translation model (~4.4 GB) into the `models`
volume, and faster-whisper downloads its model into the same volume on first use.

To build the image locally instead of pulling it, uncomment `build: .` in
`docker-compose.yml` and run `docker compose up -d --build`.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `TRANSLATE_MODEL_FILE` | `Qwen2.5-14B-Instruct-Q8_0.gguf` | Translation GGUF model file (in `models/`); use a Q4_K_M file or the 7B for less RAM |
| `TRANSLATE_MODEL_REPO` | `bartowski/Qwen2.5-14B-Instruct-GGUF` | Hugging Face repo the model is downloaded from on first run |
| `TRANSLATE_N_CTX` | `8192` | Context window for the translation model |
| `TRANSLATE_BATCH_SIZE` | `12` | Subtitle cues translated per LLM call |
| `TRANSLATE_CONTEXT_CUES` | `3` | Preceding cues shown as context for each batch |
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model size (`small`/`medium` are faster, lower quality) |
| `WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper compute type (CPU) |
| `WHISPER_VAD` | `true` | Trim non-speech (VAD) to keep subtitle timing in sync |
| `MAX_CONCURRENT_JOBS` | `1` | Jobs run in parallel (each loads Whisper + the LLM) |
| `JOB_TIMEOUT` | `7200` | Max seconds before a job is killed |
| `RESULT_TTL` | `21600` | Seconds a finished video is kept if not downloaded |

## Run without Docker

Requirements:

- Python 3.12+
- ~16 GB RAM free for Qwen2.5-14B Q8_0 on CPU (less for a Q4_K_M / 7B model)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- [Node.js](https://nodejs.org/) (for yt-dlp JS challenge bypass)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the translation model (~15.7 GB):

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Qwen2.5-14B-Instruct-GGUF',
    filename='Qwen2.5-14B-Instruct-Q8_0.gguf',
    local_dir='models'
)
"
```

### Web UI

```bash
.venv/bin/python app.py
```

Open http://localhost:8000, paste a YouTube URL, and click **Pornește**.

The UI shows live progress for each pipeline step, a "Videoclipuri recente" section
listing finished videos with a countdown until they expire (`RESULT_TTL`) and a delete
button, and a Sistem/Luminos/Întunecat (system/light/dark) theme toggle that defaults to
the OS preference and is remembered via `localStorage`.

### CLI

```bash
.venv/bin/python main.py "https://youtube.com/watch?v=..." --target-lang ro
```

## Pipeline Steps

1. **Extract** — fetch video info (title, available subtitles)
2. **Download subtitles** or **transcribe** with faster-whisper (if no subs)
3. **Translate** cues to the target language in batches with surrounding context
   (via Qwen2.5-7B), so fragmented subtitle lines stay coherent
4. **Download video** + **hardsub** subtitles with ffmpeg

## Architecture notes

- Each web job runs `worker.py` as its own subprocess (its own process group),
  reporting progress as JSON lines on stdout consumed by `app.py` over SSE.
- Cancelling a job (`POST /cancel/{task_id}`) sends `SIGTERM`/`SIGKILL` to the
  whole process group, so ffmpeg/yt-dlp/Whisper/llama are actually killed —
  not left running in the background.
- Job files live under `jobs/<task_id>/`. Intermediate files are deleted once a
  job finishes; the final video is removed after download or after `RESULT_TTL`.

## Files

- `app.py` — FastAPI server: job orchestration, SSE progress, cancel, download
- `worker.py` — per-job subprocess: download/transcribe/translate/hardsub
- `transcribe.py` — yt-dlp wrapper, faster-whisper, SRT/VTT parsing
- `translate.py` — batched, context-aware translation via Qwen2.5-7B (llama-cpp-python)
- `main.py` — CLI pipeline
- `static/index.html` — Romanian web interface
- `jobs/` — per-job working directories (created at runtime)
- `models/` — GGUF model file + faster-whisper cache
