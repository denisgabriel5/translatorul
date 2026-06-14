"""Job lifecycle management.

Each translation job runs as its own `worker.py` subprocess (in its own
process group), so a cancelled or timed-out job can be killed outright --
including any ffmpeg/yt-dlp/Whisper/llama children. This module owns the
in-memory job registry, spawning/cancelling/streaming worker output, and
disk cleanup (intermediate files, TTL sweep, persisted result metadata so
finished jobs survive a server restart).
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import time
import uuid
from collections import deque
from pathlib import Path

logger = logging.getLogger("translatorul")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
WORKER_SCRIPT = BASE_DIR / "worker.py"
RESULT_META_FILE = "result.json"
FINAL_VIDEO_FILE = "final.mp4"

JOBS_DIR.mkdir(exist_ok=True)

# How many pipeline jobs may run at once. Each one loads Whisper + the
# translation LLM, so keep this low on CPU-only hosts.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

# Overall wall-clock budget for a single job before it's killed.
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", str(2 * 60 * 60)))

# How long a finished job's output is kept on disk and listed for download.
RESULT_TTL = int(os.environ.get("RESULT_TTL", str(24 * 60 * 60)))
SWEEP_INTERVAL = int(os.environ.get("SWEEP_INTERVAL", str(5 * 60)))

PROCESS_TERM_GRACE = 5  # seconds to wait after SIGTERM before SIGKILL

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# task_id -> job state dict. Entries persist after completion so the result
# can be listed/downloaded until the TTL sweep or a manual delete.
jobs: dict[str, dict] = {}


async def _terminate_process_group(proc: asyncio.subprocess.Process):
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=PROCESS_TERM_GRACE)
    except asyncio.TimeoutError:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def _stream_output(proc: asyncio.subprocess.Process, q: asyncio.Queue, job: dict, short: str):
    # Progress arrives in ~1% ticks; only log on step change / each ~10% so the
    # container log stays readable while the UI still gets every update.
    last_step = None
    last_decile = -1
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = msg.get("step")
        if step == "error":
            job["errored"] = True
            logger.error("job %s | eroare | %s", short, msg.get("message", ""))
        elif step == "done":
            job["succeeded"] = True
            try:
                job["display_name"] = json.loads(msg["message"])["file"]
            except (KeyError, json.JSONDecodeError):
                job["display_name"] = "video.mp4"
            logger.info("job %s | gata | %s", short, job["display_name"])
        else:
            progress = msg.get("progress") or 0
            decile = int(progress * 10)
            if step != last_step or decile != last_decile:
                last_step, last_decile = step, decile
                logger.info("job %s | %-9s %3d%% | %s", short, step, int(progress * 100), msg.get("message", ""))
        await q.put(msg)


async def _stream_stderr(proc: asyncio.subprocess.Process, tail: deque, short: str):
    """Forward the worker's stderr (ffmpeg/yt-dlp/Whisper/model download) to the
    container log, keeping the last lines so a failure can report a useful message."""
    async for raw_line in proc.stderr:
        line = raw_line.decode("utf-8", errors="ignore").rstrip()
        if not line:
            continue
        tail.append(line)
        logger.info("job %s |   %s", short, line)


def _rm(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_intermediate_files(job_dir: Path):
    """Remove everything except the final hardsubbed video."""
    if not job_dir.exists():
        return
    for f in job_dir.iterdir():
        if f.name != FINAL_VIDEO_FILE:
            _rm(f)


def _write_result_meta(job_dir: Path, display_name: str, completed_at: float):
    meta = {"display_name": display_name, "completed_at": completed_at}
    try:
        (job_dir / RESULT_META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    except OSError:
        pass


async def run_job(task_id: str, url: str, target_lang: str):
    job = jobs[task_id]
    q: asyncio.Queue = job["queue"]
    job_dir: Path = job["job_dir"]
    short = task_id[:8]

    async def emit(step: str, message: str, progress: float, status: str = "active"):
        await q.put({"step": step, "message": message, "progress": progress, "status": status})

    try:
        await emit("queued", "În coadă...", 0.0, "queued")

        async with _semaphore:
            if job.get("cancelled"):
                return

            logger.info("job %s | pornire | %s -> %s", short, url, target_lang)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(WORKER_SCRIPT), url, target_lang, str(job_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            job["process"] = proc
            stderr_tail: deque = deque(maxlen=25)

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _stream_output(proc, q, job, short),
                        _stream_stderr(proc, stderr_tail, short),
                    ),
                    timeout=JOB_TIMEOUT,
                )
                await proc.wait()
            except asyncio.TimeoutError:
                await _terminate_process_group(proc)
                job["errored"] = True
                logger.error("job %s | timeout după %ss", short, JOB_TIMEOUT)
                await emit("error", "Procesul a expirat (timeout)", 0)

            if proc.returncode != 0 and not job.get("succeeded") and not job.get("cancelled"):
                job["errored"] = True
                message = stderr_tail[-1] if stderr_tail else "Procesul a eșuat"
                logger.error("job %s | eșuat (cod %s)", short, proc.returncode)
                await emit("error", message, 0)

    except asyncio.CancelledError:
        proc = job.get("process")
        if proc is not None:
            await _terminate_process_group(proc)
        job["cancelled"] = True
        logger.info("job %s | anulat", short)

    finally:
        if job.get("succeeded") and not job.get("errored"):
            _cleanup_intermediate_files(job_dir)
            completed_at = time.time()
            job["completed_at"] = completed_at
            _write_result_meta(job_dir, job.get("display_name", "video.mp4"), completed_at)
        else:
            _rm(job_dir)
            jobs.pop(task_id, None)
        await q.put(None)


def start_job(url: str, target_lang: str) -> str:
    task_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / task_id
    jobs[task_id] = {
        "queue": asyncio.Queue(),
        "job_dir": job_dir,
        "process": None,
        "cancelled": False,
        "errored": False,
        "succeeded": False,
    }
    jobs[task_id]["task"] = asyncio.create_task(run_job(task_id, url, target_lang))
    return task_id


async def cancel_job(task_id: str) -> bool:
    job = jobs.get(task_id)
    if job is None:
        return False

    job["cancelled"] = True
    proc = job.get("process")
    if proc is not None:
        await _terminate_process_group(proc)
    job["task"].cancel()
    await job["queue"].put({"step": "cancelled", "message": "Anulat", "progress": 0, "status": "cancelled"})
    return True


def delete_job(task_id: str) -> bool:
    job = jobs.get(task_id)
    if job is None:
        return False
    _rm(job["job_dir"])
    jobs.pop(task_id, None)
    return True


def list_completed_jobs() -> list[dict]:
    result = []
    for task_id, job in jobs.items():
        if not job.get("succeeded") or job.get("completed_at") is None:
            continue
        if not (job["job_dir"] / FINAL_VIDEO_FILE).exists():
            continue
        result.append({
            "task_id": task_id,
            "display_name": job.get("display_name", "video.mp4"),
            "completed_at": job["completed_at"],
            "expires_at": job["completed_at"] + RESULT_TTL,
        })
    result.sort(key=lambda j: j["completed_at"], reverse=True)
    return result


def load_completed_jobs():
    """Rebuild job entries for finished results that survived a restart."""
    if not JOBS_DIR.exists():
        return
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        task_id = job_dir.name
        if task_id in jobs:
            continue
        meta_file = job_dir / RESULT_META_FILE
        final_path = job_dir / FINAL_VIDEO_FILE
        if not meta_file.exists() or not final_path.exists():
            _rm(job_dir)
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _rm(job_dir)
            continue
        jobs[task_id] = {
            "job_dir": job_dir,
            "cancelled": False,
            "errored": False,
            "succeeded": True,
            "display_name": meta.get("display_name", "video.mp4"),
            "completed_at": meta.get("completed_at", time.time()),
        }


async def sweep_loop():
    while True:
        now = time.time()
        for task_id, job in list(jobs.items()):
            completed_at = job.get("completed_at")
            if completed_at is not None and now - completed_at > RESULT_TTL:
                _rm(job["job_dir"])
                jobs.pop(task_id, None)
        await asyncio.sleep(SWEEP_INTERVAL)
