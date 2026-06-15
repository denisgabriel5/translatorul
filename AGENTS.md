# AGENTS.md — Session Summary

## Goal

Build a local pipeline that downloads YouTube videos, extracts subtitles or transcribes with
faster-whisper, translates to Romanian (or another language) with Madlad-400, and serves it
through a modern web interface with hardsubbed output. Runs on CPU-only hardware via Docker.

## Project Layout

```
translatorul/
├── app.py                  # FastAPI server (SSE, cancel, download, job orchestration)
├── worker.py               # per-job subprocess: download/transcribe/translate/hardsub
├── transcribe.py           # YouTube download, subtitle extraction, faster-whisper, SRT I/O
├── translate.py            # Madlad-400 translation via CTranslate2 (NMT)
├── main.py                 # CLI orchestrator (download → transcribe → translate)
├── static/index.html       # Romanian web UI
├── Dockerfile, docker-compose.yml, docker-entrypoint.sh
├── models/                  # Madlad CT2 model (madlad-ct2/) + faster-whisper cache (volume)
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
- **Progress persists across refresh**: each job keeps a full in-memory **event log**
  (`job["events"]` + an `asyncio.Condition`), not a single-consumer queue. `GET /progress/{id}`
  replays the whole log before streaming live, so a reload or reopened tab reattaches and
  rebuilds the bars. The browser stores the in-flight `task_id` in `localStorage` and
  reconnects on load. Closing the tab no longer cancels the job — it runs to completion;
  cancellation happens only via the explicit **Cancel** button (`POST /cancel/{task_id}`).
  A job does **not** survive a server restart, though: the worker is killed and its
  unfinished `jobs/<id>/` dir is swept on startup (`load_completed_jobs`), so the
  reconnecting client gets a 404 and shows the job as cancelled.
- **UI** (`static/index.html`): Apple-style design (SF Pro fonts, glassmorphism, blue
  accent). A "Videoclipuri recente" card lists finished jobs from `GET /jobs` with a
  live countdown to `expires_at` and a delete button (`DELETE /jobs/{task_id}`). A
  Sistem/Luminos/Întunecat segmented control toggles `data-theme` on `<html>`, persisted
  in `localStorage`; with no choice saved it follows `prefers-color-scheme`.

## Translation: Madlad-400 NMT (`translate.py`)

Earlier iterations used a general LLM (Qwen2.5), first per-cue then batched-with-context.
That fixed alignment but was slow on CPU (a 14B model ran 30+ min for a 20-min video) and
still occasionally fabricated non-words. Now translation uses a dedicated NMT:

- **Madlad-400 3B** via **CTranslate2** (`santhosh/madlad400-3b-ct2` — a CT2 build with
  `model.bin` + `sentencepiece.model`; the plain HF/transformers repo is NOT CT2-loadable),
  `int8` on CPU. ~10-30× faster than the LLM, won't invent words, no source-lang detection.
- Each cue is prefixed with the Madlad target token `<2xx>` (e.g. `<2ro>`), tokenized with
  SentencePiece, and translated; one output per input cue keeps timestamps aligned.
- Env-configurable: `TRANSLATE_MODEL_REPO`, `TRANSLATE_MODEL_DIR`/`TRANSLATE_MODEL_SUBDIR`,
  `TRANSLATE_COMPUTE_TYPE` (int8), `TRANSLATE_MAX_BATCH_SIZE` (1024 tokens), `TRANSLATE_BEAM_SIZE`
  (4), `TRANSLATE_REPETITION_PENALTY` (1.1), `TRANSLATE_NO_REPEAT_NGRAM` (0), `TRANSLATE_THREADS`.
  The SentencePiece file is auto-detected. A module-level lock guards the translator.

## Transcription

- **faster-whisper** (CTranslate2) instead of `openai-whisper` — ~4x faster on CPU, lower
  memory, streams segments. Model size/compute type via `WHISPER_MODEL` (default
  `large-v3-turbo`) and `WHISPER_COMPUTE_TYPE` (default `int8`); `WHISPER_VAD` (default on)
  trims non-speech. `word_timestamps=True` + tightening each cue to its first/last spoken
  word sharpens sync; `SUBTITLE_OFFSET_MS` nudges all cues if they read early/late.

## Docker

- `Dockerfile`: CPU-only `python:3.12-slim`, installs ffmpeg + Node.js (yt-dlp JS challenge
  bypass) + `tini` (PID 1 / zombie reaping for killed process groups). No build toolchain —
  ctranslate2/faster-whisper/sentencepiece ship prebuilt wheels, so the image builds fast.
- `docker-entrypoint.sh`: downloads the Madlad CT2 model into `/app/models/madlad-ct2`
  whenever `model.bin` is missing (self-heals a partial/incompatible model).
- `docker-compose.yml`: pulls `ghcr.io/denisgabriel5/translatorul:latest`; `models` and `jobs`
  named volumes persist weights and (briefly) job output.
- `.github/workflows/docker-publish.yml`: builds and pushes the image to GHCR on push to
  `main` / tags.

## Known Issues / Notes

- No GPU support; everything runs on CPU.
- A fresh worker process per job means the translation/Whisper models reload each run (a few
  seconds) — the cost of guaranteed killability via process isolation.
- Romanian character display in PowerShell console may need `$env:PYTHONIOENCODING='utf-8'`.
