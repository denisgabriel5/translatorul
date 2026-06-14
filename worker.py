"""Runs a single transcription/translation/hardsub job in its own process.

Invoked as: python worker.py <url> <target_lang> <job_dir>

Emits one JSON object per line on stdout describing progress, matching the
shape expected by app.py's SSE stream:
    {"step": ..., "message": ..., "progress": ..., "status": ...}

Running this as a separate process (in its own process group, started by
app.py with start_new_session=True) means a cancelled job can be killed
outright -- including any ffmpeg/yt-dlp children -- which is not possible
for work running inside a thread pool.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yt_dlp

from transcribe import (
    BASE_YDL_OPTS,
    download_subs,
    extract_info,
    has_manual_subtitles,
    read_vtt_cues,
    transcribe,
    transcribe_to_cues,
    write_srt,
)
from translate import translate_text

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "1800"))


def emit(step: str, message: str, progress: float, status: str = "active"):
    print(
        json.dumps({"step": step, "message": message, "progress": progress, "status": status}),
        flush=True,
    )


def done(step: str, message: str):
    emit(step, message, 1.0, "done")


def progress_emitter(step: str, message: str):
    """Callback that emits monotonically increasing per-step progress, throttled
    to ~1% steps so the bar fills smoothly without flooding the output stream."""
    state = {"last": -1.0}

    def cb(fraction: float):
        fraction = max(0.0, min(1.0, fraction))
        if fraction >= state["last"] + 0.01:
            state["last"] = fraction
            emit(step, message, fraction)

    return cb


def download_video(info: dict, job_dir: Path, on_progress=None) -> Path:
    def hook(d):
        if on_progress is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(d.get("downloaded_bytes", 0) / total)
        elif d.get("status") == "finished":
            on_progress(1.0)

    output_template = str(job_dir / "video.%(ext)s")
    ydl_opts = {
        **BASE_YDL_OPTS,
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(info["webpage_url"], download=True)
    return job_dir / "video.mp4"


def _parse_ffmpeg_time(value: str):
    try:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return None


def embed_subtitles(video_path: Path, srt_path: Path, job_dir: Path,
                    duration: float = 0, on_progress=None) -> Path:
    out_path = job_dir / "final.mp4"
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\\\''")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles={srt_escaped}",
            "-c:a", "copy",
            "-progress", "pipe:1", "-nostats", "-loglevel", "error",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = []
    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("out_time=") and on_progress and duration:
                secs = _parse_ffmpeg_time(line.split("=", 1)[1])
                if secs is not None:
                    on_progress(min(0.99, secs / duration))
            elif line.strip() and "=" not in line:
                tail.append(line.strip())
                del tail[:-10]
    try:
        proc.wait(timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if proc.returncode not in (0, None) and tail:
        sys.stderr.write("\n".join(tail) + "\n")
        sys.stderr.flush()
    if on_progress:
        on_progress(1.0)
    return out_path


def run(url: str, target_lang: str, job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)

    emit("extract", "Se extrag informațiile...", 0.0)
    info = extract_info(url)
    title = info.get("title") or "video"
    duration = info.get("duration") or 0

    emit("extract", "Se descarcă videoclipul...", 0.0)
    video_path = download_video(
        info, job_dir, on_progress=progress_emitter("extract", "Se descarcă videoclipul...")
    )
    done("extract", "Videoclip descărcat")

    if has_manual_subtitles(info):
        emit("content", "Subtitrări găsite, se descarcă...", 0.0)
        sub_file = download_subs(info, job_dir)
        cues = read_vtt_cues(sub_file)
        sub_file.unlink()
        done("content", "Subtitrări descărcate")
    else:
        emit("content", "Se transcrie audio...", 0.0)
        result = transcribe(
            video_path, WHISPER_MODEL,
            on_progress=progress_emitter("content", "Se transcrie audio..."),
        )
        cues = transcribe_to_cues(result)
        done("content", "Transcriere completă")

    emit("translate", "Se traduce...", 0.0)
    translated = translate_text(
        [t for _, t in cues], target_lang,
        on_progress=progress_emitter("translate", "Se traduce..."),
    )
    translated_cues = [(timing, t) for (timing, _), t in zip(cues, translated)]
    done("translate", "Traducere completă")

    srt_path = job_dir / "translated.srt"
    write_srt(translated_cues, srt_path)

    emit("embed", "Se încorporează subtitrările...", 0.0)
    final_path = embed_subtitles(
        video_path, srt_path, job_dir, duration=duration,
        on_progress=progress_emitter("embed", "Se încorporează subtitrările..."),
    )
    done("embed", "Videoclip gata")

    if not final_path.exists():
        raise RuntimeError("ffmpeg failed to produce the subtitled video")

    display_name = f"{title}.{target_lang}.mp4"
    emit("done", json.dumps({"file": display_name}), 1.0)


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"step": "error", "message": "usage: worker.py <url> <target_lang> <job_dir>", "progress": 0}), flush=True)
        sys.exit(2)

    url, target_lang, job_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    try:
        run(url, target_lang, job_dir)
    except Exception as e:
        emit("error", str(e), 0)
        sys.exit(1)


if __name__ == "__main__":
    main()
