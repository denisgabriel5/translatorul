# Translatorul — YouTube Transcriber & Translator

Downloads YouTube videos, extracts or transcribes subtitles, translates them to Romanian using Qwen2.5-7B, and produces a hardsubbed video.

## Requirements

- Python 3.12+
- 24 GB RAM (for Qwen2.5-7B Q4_K_M on CPU)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- [Node.js](https://nodejs.org/) (for yt-dlp JS challenge bypass)

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Download the translation model (~4.4 GB):

```bash
.venv\Scripts\python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='lmstudio-community/Qwen2.5-7B-Instruct-GGUF',
    filename='Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    local_dir='models'
)
"
```

## Usage

### Web UI

```bash
.venv\Scripts\python app.py
```

Open http://localhost:8000, paste a YouTube URL, and click **Trimite**.

### CLI

```bash
.venv\Scripts\python main.py "https://youtube.com/watch?v=..." --target-lang ro
```

## Pipeline Steps

1. **Extract** — fetch video info (title, available subtitles)
2. **Download subtitles** or **Transcribe** with Whisper (if no subs)
3. **Translate** each cue to Romanian via Qwen2.5-7B
4. **Download video** + **hardsub** subtitles with ffmpeg

## Files

- `app.py` — FastAPI server with SSE progress, cancel, download
- `transcribe.py` — yt-dlp wrapper, Whisper, SRT/VTT parsing
- `translate.py` — Qwen2.5-7B via llama-cpp-python GGUF
- `main.py` — CLI pipeline
- `static/index.html` — Romanian web interface
- `input/` — audio cache
- `output/` — SRT transcripts
- `videos/` — final hardsubbed videos
- `models/` — GGUF model file
