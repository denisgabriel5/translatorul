import html
from pathlib import Path

import whisper
import yt_dlp


SUBTITLE_LANGS = ["en", "es", "fr", "de", "pt", "it", "nl", "ru", "ja", "ko", "zh"]

BASE_YDL_OPTS = {
    "quiet": True,
    "socket_timeout": 30,
    "js_runtimes": {"node": {}},
    "remote_components": {"ejs:github"},
}


def extract_info(url: str) -> dict:
    with yt_dlp.YoutubeDL({**BASE_YDL_OPTS}) as ydl:
        return ydl.extract_info(url, download=False)


def has_manual_subtitles(info: dict) -> bool:
    return bool(info.get("subtitles"))


def predict_audio_path(info: dict, output_dir: Path) -> Path:
    with yt_dlp.YoutubeDL({**BASE_YDL_OPTS}) as ydl:
        raw = ydl.prepare_filename(info)
    return Path(raw).with_suffix(".mp3")


def download_subs(info: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        **BASE_YDL_OPTS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": SUBTITLE_LANGS,
        "subtitlesformat": "vtt",
        "outtmpl": output_template,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(info["webpage_url"], download=True)
        sub_path = Path(ydl.prepare_filename(info))

    langs = list(info.get("subtitles", {}).keys())
    chosen = langs[0]
    sub_file = sub_path.with_name(f"{sub_path.stem}.{chosen}.vtt")

    return sub_file


def download_audio(info: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        **BASE_YDL_OPTS,
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(info["webpage_url"], download=True)
        audio_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")

    return audio_path


def transcribe(audio_path: Path, model_name: str = "base", language: str | None = None) -> dict:
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, verbose=False)
    return result


def read_vtt_cues(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    cues: list[tuple[str, str]] = []
    current_timing = ""
    current_text: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT":
            if current_timing and current_text:
                cues.append((current_timing, " ".join(current_text)))
            current_timing = ""
            current_text = []
        elif " --> " in stripped:
            current_timing = stripped
        else:
            current_text.append(html.unescape(stripped).replace("\u00a0", ""))
    if current_timing and current_text:
        cues.append((current_timing, " ".join(current_text)))
    return cues


def transcribe_to_cues(result: dict) -> list[tuple[str, str]]:
    cues = []
    for seg in result["segments"]:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        cues.append((f"{start} --> {end}", seg["text"].strip()))
    return cues


def write_srt(cues: list[tuple[str, str]], path: Path):
    lines = []
    for i, (timing, text) in enumerate(cues, start=1):
        srt_timing = timing.replace(".", ",")
        lines.append(f"{i}\n{srt_timing}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(cues: list[tuple[str, str]], path: Path):
    lines = ["WEBVTT\n"]
    for timing, text in cues:
        lines.append(f"{timing}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(cues: list[tuple[str, str]], path: Path):
    text = " ".join(text for _, text in cues)
    path.write_text(text, encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp(seconds).replace(",", ".")
