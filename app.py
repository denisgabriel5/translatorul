import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

import job_manager

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.load_completed_jobs()
    sweep_task = asyncio.create_task(job_manager.sweep_loop())
    try:
        yield
    finally:
        sweep_task.cancel()


app = FastAPI(title="Translatorul", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # Revalidate on every load so a redeploy never serves a stale UI.
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.post("/start")
async def start_task(url: str, target_lang: str = "ro"):
    task_id = job_manager.start_job(url, target_lang)
    return {"task_id": task_id}


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    if not await job_manager.cancel_job(task_id):
        raise HTTPException(404, "Task not found")
    return {"status": "cancelled"}


@app.get("/progress/{task_id}")
async def progress(request: Request, task_id: str):
    job = job_manager.jobs.get(task_id)
    # Jobs reloaded from disk after a restart are finished and have no live
    # stream; treat them as gone so the client falls back to the history list.
    if job is None or "event_cond" not in job:
        raise HTTPException(404, "Task not found")
    cond: asyncio.Condition = job["event_cond"]

    async def generator():
        # Replay the whole event log first so a reconnecting/refreshed client
        # rebuilds the current progress, then keep streaming new events.
        # Closing the tab no longer cancels the job -- it runs to completion.
        idx = 0
        while True:
            if await request.is_disconnected():
                break
            async with cond:
                while idx >= len(job["events"]) and not job["closed"]:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        break
                pending = job["events"][idx:]
                idx += len(pending)
                closed = job["closed"]
            for msg in pending:
                yield {"event": "progress", "data": json.dumps(msg)}
            if closed and idx >= len(job["events"]):
                break

    return EventSourceResponse(generator())


@app.get("/download/{task_id}")
async def download(task_id: str):
    job = job_manager.jobs.get(task_id)
    if job is None:
        raise HTTPException(404, "Task not found")

    final_path = job["job_dir"] / "final.mp4"
    if not job.get("succeeded") or not final_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(final_path, filename=job.get("display_name", "video.mp4"))


@app.get("/jobs")
async def list_jobs():
    return job_manager.list_completed_jobs()


@app.delete("/jobs/{task_id}")
async def delete_job(task_id: str):
    if not job_manager.delete_job(task_id):
        raise HTTPException(404, "Task not found")
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
