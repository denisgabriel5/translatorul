import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from transcribe import (
    download_audio,
    download_subs,
    extract_info,
    has_manual_subtitles,
    predict_audio_path,
    read_vtt_cues,
    transcribe,
    transcribe_to_cues,
    write_srt,
    write_txt,
    write_vtt,
)
from translate import translate_text

console = Console()

console = Console()


def translate_cues(cues: list[tuple[str, str]], target_lang: str) -> list[tuple[str, str]]:
    texts = [text for _, text in cues]
    translated = translate_text(texts, target_lang)
    return [(timing, t) for (timing, _), t in zip(cues, translated)]


def main():
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and translate YouTube videos"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo"],
                        help="faster-whisper model size (default: small)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"),
                        help="Output directory (default: output/)")
    parser.add_argument("--input", "-i", type=Path, default=Path("input"),
                        help="Audio cache directory (default: input/)")
    parser.add_argument("--format", "-f", choices=["txt", "srt", "vtt"],
                        default="srt", help="Output format (default: srt)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g. 'en'). Auto-detected if not set.")
    parser.add_argument("--force-transcribe", action="store_true",
                        help="Skip subtitle check and always use Whisper.")
    parser.add_argument("--target-lang", default="ro",
                        help="Translate to language code (default: ro for Romanian). Pass empty string to skip.")
    args = parser.parse_args()

    info = extract_info(args.url)
    do_translate = args.target_lang not in (None, "")
    stem = predict_audio_path(info, args.output).stem

    # Step 1: get cues (from subs or transcription)
    if not args.force_transcribe and has_manual_subtitles(info):
        console.print(Panel.fit("[bold yellow]Subtitles found[/bold yellow]", border_style="yellow"))
        sub_file = download_subs(info, args.output)
        cues = read_vtt_cues(sub_file)
        sub_file.unlink()
    else:
        console.print(Panel.fit("[bold cyan]Transcribing with Whisper[/bold cyan]", border_style="cyan"))
        audio_path = predict_audio_path(info, args.input)
        if audio_path.exists():
            console.print(f"[blue]Cached[/blue] [bold]{audio_path}[/bold]")
        else:
            console.print("[yellow]Downloading audio...[/yellow]")
            audio_path = download_audio(info, args.input)
            console.print(f"[green]Downloaded[/green] [bold]{audio_path}[/bold]")

        console.print(f"[yellow]Loading model[/yellow] [bold]{args.model}[/bold]...")
        cues = transcribe_to_cues(transcribe(audio_path, args.model, args.language))
        console.print("[green]Transcription complete[/green]")

    # Step 2: translate if requested
    if do_translate:
        console.print(f"[yellow]Translating to {args.target_lang}...[/yellow]")
        cues = translate_cues(cues, args.target_lang)

    # Step 3: write output
    suffix = f".{args.target_lang}" if do_translate else ""
    output_file = args.output / f"{stem}{suffix}.{args.format}"

    if args.format == "srt":
        write_srt(cues, output_file)
    elif args.format == "vtt":
        write_vtt(cues, output_file)
    else:
        write_txt(cues, output_file)

    console.print(f"[green]Saved[/green] [bold]{output_file}[/bold]")

    if args.format == "txt":
        console.print("\n[underline]Transcript[/underline]")
        console.print(" ".join(t for _, t in cues))


if __name__ == "__main__":
    main()
