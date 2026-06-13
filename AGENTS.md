# AGENTS.md — Session Summary

## Goal

Build a local pipeline that downloads YouTube videos, extracts subtitles or transcribes with Whisper, translates to Romanian with Qwen2.5-7B, and serves it through a modern web interface with hardsubbed output.

## Project Layout

```
yt_transcribe/
├── app.py              # FastAPI server (SSE, cancel, hardsub)
├── transcribe.py       # YouTube download, subtitle extraction, Whisper, SRT I/O
├── translate.py        # Qwen2.5-7B translation via llama-cpp-python GGUF
├── main.py             # CLI orchestrator (download → transcribe → translate)
├── static/index.html   # Romanian web UI
├── models/             # GGUF model directory
└── input/ output/ videos/  # Runtime directories
```

## Architecture

- **FastAPI** + **SSE** for real-time progress streaming (4 steps: extract → content → translate → download)
- Pipeline runs in a **thread pool executor** to keep the event loop responsive
- **Cancel**: `asyncio.Task.cancel()` → propagates to thread-pool future; SSE sends `cancelled` event
- **Auto-cancel**: browser `beforeunload` sends `POST /cancel/{task_id}` via `navigator.sendBeacon`

## Translation Model

- **Qwen2.5-7B-Instruct** (Q4_K_M GGUF, 4.36 GB)
- Source: `lmstudio-community/Qwen2.5-7B-Instruct-GGUF`
- Loaded via `llama-cpp-python 0.3.28` (pre-built wheel)
- Prompt-based: system prompt + `"English: {text}\nRomanian:"`
- Runs on CPU with 4 threads; ~2-4s per cue

## Fallback Transcription

- Uses `openai-whisper` (model `base`) if no manual subtitles exist
- Audio downloaded with yt-dlp, cached in `input/`

## Server

- Start with `python app.py` (uses uvicorn with reload)
- Requires `watchfiles` for reload to work (missing package caused hangs)
- yt-dlp configured with `socket_timeout: 30` and Node.js JS runtime for challenge bypass

## Key Config

| Setting | Value |
|---------|-------|
| Default target language | `ro` (Romanian) |
| FFmpeg hardsub filter | `subtitles=` |
| n_ctx | 4096 |
| ThreadPool workers | 2 |
| yt-dlp socket timeout | 30s |
| Pipeline step timeout | 300s |

## Known Issues

- No GPU support; 7B model runs on CPU (slower but functional)
- Romanian character display in PowerShell console may need `$env:PYTHONIOENCODING='utf-8'`
