# AGENTS.md — Session Summary

## Goal

Build a local pipeline that downloads YouTube videos, extracts subtitles or transcribes with
faster-whisper, translates to Romanian (or another language) with Qwen2.5-7B, and serves it
through a modern web interface with hardsubbed output. Runs on CPU-only hardware via Docker.

## Project Layout

```
translatorul/
├── app.py                  # FastAPI server (SSE, cancel, download, job orchestration)
├── worker.py               # per-job subprocess: download/transcribe/translate/hardsub
├── transcribe.py           # YouTube download, subtitle extraction, faster-whisper, SRT I/O
├── translate.py            # Qwen2.5-7B translation via llama-cpp-python GGUF (batched)
├── main.py                 # CLI orchestrator (download → transcribe → translate)
├── static/index.html       # Romanian web UI
├── Dockerfile, docker-compose.yml, docker-entrypoint.sh
├── models/                  # GGUF model + faster-whisper cache (volume)
└── jobs/                     # per-job working directories (volume, runtime)
```

## Architecture

- **FastAPI** + **SSE** for real-time progress streaming (5 statuses: queued → extract →
  content → translate → download)
- Each job runs **`worker.py` as its own subprocess**, started with `start_new_session=True`
  (its own process group), emitting one JSON progress object per stdout line.
- **Cancel**: `POST /cancel/{task_id}` sends `SIGTERM` then (after a grace period) `SIGKILL`
  to the worker's whole process group via `os.killpg`, killing ffmpeg/yt-dlp/Whisper/llama
  together. This replaced the old `ThreadPoolExecutor` + `task.cancel()` approach, which left
  threads (and ffmpeg) running after "cancel" and could deadlock the 2-worker pool.
- **Concurrency**: `asyncio.Semaphore(MAX_CONCURRENT_JOBS)` (default 1) — extra jobs emit a
  `queued` status while waiting.
- **Cleanup**: on completion, intermediate files in `jobs/<task_id>/` are deleted, keeping
  only `final.mp4`; on cancel/error the whole job dir is removed. A periodic sweep removes
  finished-but-undownloaded results after `RESULT_TTL`.
- **Auto-cancel**: SSE disconnect (browser closed/navigated away) cancels the job; the
  `beforeunload` `navigator.sendBeacon` call remains as a backup.
- **UI** (`static/index.html`): Apple-style design (SF Pro fonts, glassmorphism, blue
  accent). A "Videoclipuri recente" card lists finished jobs from `GET /jobs` with a
  live countdown to `expires_at` and a delete button (`DELETE /jobs/{task_id}`). A
  Sistem/Luminos/Întunecat segmented control toggles `data-theme` on `<html>`, persisted
  in `localStorage`; with no choice saved it follows `prefers-color-scheme`.

## Translation: batched, context-aware (`translate.py`)

The original approach translated each subtitle cue independently, which produced broken
Romanian because cues are sentence fragments. Now:

- Cues are translated in batches (`TRANSLATE_BATCH_SIZE`, default 12), with the preceding
  `TRANSLATE_CONTEXT_CUES` (default 3) source cues included as read-only context.
- The model is asked to return numbered lines (`"1. ..."`) matching the input count.
- If the returned line count doesn't match the batch size (model merged/split lines), that
  batch falls back to per-cue translation — guaranteeing cue/timestamp alignment is never
  broken.
- **Qwen2.5-7B-Instruct** (Q4_K_M GGUF, 4.36 GB), `lmstudio-community/Qwen2.5-7B-Instruct-GGUF`,
  loaded via `llama-cpp-python`. Model path/`n_ctx`/threads are env-configurable
  (`TRANSLATE_MODEL_DIR`, `TRANSLATE_MODEL_FILE`, `TRANSLATE_N_CTX`, `TRANSLATE_N_THREADS`).
- A module-level lock guards `_llm` calls (only matters if `MAX_CONCURRENT_JOBS > 1`).

## Transcription

- **faster-whisper** (CTranslate2) instead of `openai-whisper` — ~4x faster on CPU, lower
  memory, streams segments. Model size/compute type via `WHISPER_MODEL` (default `small`) and
  `WHISPER_COMPUTE_TYPE` (default `int8`).

## Docker

- `Dockerfile`: CPU-only `python:3.12-slim`, builds `llama-cpp-python` from source, installs
  ffmpeg + Node.js (yt-dlp JS challenge bypass) + `tini` (PID 1 / zombie reaping for killed
  process groups).
- `docker-entrypoint.sh`: downloads the Qwen GGUF into `/app/models` on first run if missing.
- `docker-compose.yml`: pulls `ghcr.io/denisgabriel5/translatorul:latest`; `models` and `jobs`
  named volumes persist weights and (briefly) job output.
- `.github/workflows/docker-publish.yml`: builds and pushes the image to GHCR on push to
  `main` / tags.

## Known Issues / Notes

- No GPU support; everything runs on CPU.
- A fresh worker process per job means the Qwen model reloads each run (a few seconds) —
  the cost of guaranteed killability via process isolation.
- Romanian character display in PowerShell console may need `$env:PYTHONIOENCODING='utf-8'`.
