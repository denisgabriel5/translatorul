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
    download_audio,
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


def download_video(info: dict, job_dir: Path) -> Path:
    output_template = str(job_dir / "video.%(ext)s")
    ydl_opts = {
        **BASE_YDL_OPTS,
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(info["webpage_url"], download=True)
    return job_dir / "video.mp4"


def embed_subtitles(video_path: Path, srt_path: Path, job_dir: Path) -> Path:
    out_path = job_dir / "final.mp4"
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
        timeout=FFMPEG_TIMEOUT,
    )
    return out_path


def run(url: str, target_lang: str, job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)

    emit("extract", "Se extrag informațiile...", 0.1)
    info = extract_info(url)
    title = info.get("title") or "video"
    done("extract", "Informații extrase")

    if has_manual_subtitles(info):
        emit("content", "Subtitrări găsite", 0.2)
        emit("content", "Se descarcă subtitrările...", 0.25)
        sub_file = download_subs(info, job_dir)
        cues = read_vtt_cues(sub_file)
        sub_file.unlink()
        done("content", "Subtitrări descărcate")
    else:
        emit("content", "Fără subtitrări, se transcrie...", 0.2)
        emit("content", "Se descarcă audio...", 0.2)
        audio_path = download_audio(info, job_dir)
        emit("content", "Audio descărcat, se încarcă modelul...", 0.25)
        cues = transcribe_to_cues(transcribe(audio_path, WHISPER_MODEL))
        done("content", "Transcriere completă")

    emit("translate", "Se traduce...", 0.35)
    translated = translate_text([t for _, t in cues], target_lang)
    translated_cues = [(timing, t) for (timing, _), t in zip(cues, translated)]
    done("translate", "Traducere completă")

    srt_path = job_dir / "translated.srt"
    write_srt(translated_cues, srt_path)

    emit("download", "Se descarcă videoclipul...", 0.7)
    video_path = download_video(info, job_dir)
    emit("download", "Se încorporează subtitrările...", 0.85)
    final_path = embed_subtitles(video_path, srt_path, job_dir)
    done("download", "Videoclip gata")

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
