import asyncio
import json
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from starlette.responses import RedirectResponse

from transcribe import (
    BASE_YDL_OPTS,
    download_audio,
    download_subs,
    extract_info,
    has_manual_subtitles,
    predict_audio_path,
    read_vtt_cues,
    transcribe,
    transcribe_to_cues,
    write_srt,
)
from translate import translate_text

import yt_dlp

app = FastAPI(title="YT Transcribe")

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_DIR = BASE_DIR / "videos"
STATIC_DIR = BASE_DIR / "static"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

tasks: dict[str, asyncio.Queue] = {}
task_futures: dict[str, asyncio.Task] = {}
pool = ThreadPoolExecutor(max_workers=2)

SOCKET_TIMEOUT = 30


def download_video(info: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")
    ydl_opts = {
        **BASE_YDL_OPTS,
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(info["webpage_url"], download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


def embed_subtitles(video_path: Path, srt_path: Path) -> Path:
    out_path = video_path.with_stem(video_path.stem + "_subtitled")
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\\\''")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles={srt_escaped}",
            "-c:a", "copy",
            str(out_path),
        ],
        capture_output=True,
    )
    return out_path


async def run_pipeline(url: str, target_lang: str, task_id: str):
    q = tasks[task_id]

    async def emit(step: str, message: str, progress: float, status: str = "active"):
        await q.put({"step": step, "message": message, "progress": progress, "status": status})

    async def done(step: str, message: str):
        await emit(step, message, 1.0, "done")

    loop = asyncio.get_event_loop()

    async def run_blocking(fn, *args):
        fut = loop.run_in_executor(pool, fn, *args)
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"Step timed out: {fn.__name__}")

    try:
        await emit("extract", "Se extrag informațiile...", 0.1)
        info = await run_blocking(extract_info, url)
        stem = predict_audio_path(info, INPUT_DIR).stem
        await done("extract", "Informații extrase")

        if has_manual_subtitles(info):
            await emit("content", "Subtitrări găsite", 0.2)
            await emit("content", "Se descarcă subtitrările...", 0.25)
            sub_file = await run_blocking(download_subs, info, OUTPUT_DIR)
            cues = read_vtt_cues(sub_file)
            sub_file.unlink()
            await done("content", "Subtitrări descărcate")
        else:
            await emit("content", "Fără subtitrări, se transcrie...", 0.2)
            audio_path = predict_audio_path(info, INPUT_DIR)
            if audio_path.exists():
                await emit("content", "Audio în cache, se încarcă modelul...", 0.25)
            else:
                await emit("content", "Se descarcă audio...", 0.2)
                audio_path = await run_blocking(download_audio, info, INPUT_DIR)
                await emit("content", "Audio descărcat, se încarcă modelul...", 0.25)
            model_name = "base"
            cues = transcribe_to_cues(await run_blocking(transcribe, audio_path, model_name))
            await done("content", "Transcriere completă")

        await emit("translate", "Se traduce...", 0.35)
        translated = await run_blocking(translate_text, [t for _, t in cues], target_lang)
        translated_cues = [(timing, t) for (timing, _), t in zip(cues, translated)]
        await done("translate", "Traducere completă")

        srt_path = OUTPUT_DIR / f"{stem}.{target_lang}.srt"
        write_srt(translated_cues, srt_path)

        await emit("download", "Se descarcă videoclipul...", 0.7)
        video_path = await run_blocking(download_video, info, VIDEO_DIR)
        await emit("download", "Se încorporează subtitrările...", 0.85)
        final_path = await run_blocking(embed_subtitles, video_path, srt_path)
        await done("download", "Videoclip gata")

        await emit("done", json.dumps({"file": final_path.name, "stem": final_path.stem}), 1.0)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        await emit("error", str(e), 0)
    finally:
        await q.put(None)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return html


@app.post("/start")
async def start_task(url: str, target_lang: str = "ro"):
    task_id = str(uuid.uuid4())
    tasks[task_id] = asyncio.Queue()
    task = asyncio.create_task(run_pipeline(url, target_lang, task_id))
    task_futures[task_id] = task
    return {"task_id": task_id}


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    if task_id in task_futures:
        fut = task_futures.pop(task_id, None)
        if fut:
            fut.cancel()
        q = tasks.get(task_id)
        if q:
            await q.put({"step": "cancelled", "status": "cancelled"})
        return {"status": "cancelled"}
    raise HTTPException(404, "Task not found")


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    q = tasks.get(task_id)
    if q is None:
        raise HTTPException(404, "Task not found")

    async def generator():
        while True:
            msg = await q.get()
            if msg is None:
                break
            yield {"event": "progress", "data": json.dumps(msg)}
        del tasks[task_id]
        task_futures.pop(task_id, None)

    return EventSourceResponse(generator())


@app.get("/download/{stem:path}")
async def download(stem: str):
    for f in VIDEO_DIR.iterdir():
        if f.stem == stem and f.suffix == ".mp4":
            return FileResponse(f, filename=f.name)
    raise HTTPException(404, "File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=".")
