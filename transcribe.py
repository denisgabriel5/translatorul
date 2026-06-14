import html
import os
from pathlib import Path

import yt_dlp


SUBTITLE_LANGS = ["en", "es", "fr", "de", "pt", "it", "nl", "ru", "ja", "ko", "zh"]

# faster-whisper (CTranslate2) model name and compute type. "small" / "int8"
# is a good CPU-only speed/quality balance; bump to "large-v3-turbo" if more
# RAM/CPU is available.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

# Voice-activity detection trims non-speech, which keeps segment timestamps
# from drifting around silence/music -- the main cause of out-of-sync subtitles.
WHISPER_VAD = os.environ.get("WHISPER_VAD", "true").lower() in ("1", "true", "yes")

# Constant nudge (milliseconds) applied to every cue, in case subtitles land a
# touch early/late overall. Negative shows them sooner. Default 0 (no shift).
SUBTITLE_OFFSET = float(os.environ.get("SUBTITLE_OFFSET_MS", "0")) / 1000.0

_whisper_model = None

BASE_YDL_OPTS = {
    "quiet": True,
    "socket_timeout": 30,
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


def _load_whisper_model(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            model_name,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def transcribe(audio_path: Path, model_name: str = WHISPER_MODEL, language: str | None = None) -> dict:
    model = _load_whisper_model(model_name)
    # word_timestamps gives per-word alignment; tightening each cue to its first
    # and last spoken word trims silence padding and improves sync noticeably.
    segments, _info = model.transcribe(
        str(audio_path), language=language, vad_filter=WHISPER_VAD, word_timestamps=True
    )
    out = []
    for seg in segments:
        words = seg.words or []
        start = words[0].start if words else seg.start
        end = words[-1].end if words else seg.end
        start = max(0.0, start + SUBTITLE_OFFSET)
        end = max(start, end + SUBTITLE_OFFSET)
        out.append({"start": start, "end": end, "text": seg.text})
    return {"segments": out}


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
            # VTT timing lines can carry positioning info, e.g.
            # "00:00:01.000 --> 00:00:04.000 align:start position:0%".
            # Keep only the start/end timestamps so the SRT stays valid.
            start, _, rest = stripped.partition(" --> ")
            end = rest.split(maxsplit=1)[0] if rest.strip() else rest
            current_timing = f"{start.strip()} --> {end.strip()}"
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
