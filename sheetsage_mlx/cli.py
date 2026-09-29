"""
Command line interface for SheetSage2 MLX.
"""

import argparse
import sys
from pathlib import Path
from .pipeline import transcribe


def main():
    parser = argparse.ArgumentParser(
        prog="sheetsage-mlx",
        description="SheetSage2 MLX - Ultra-fast music transcription on Apple Silicon",
    )
    parser.add_argument("audio", type=str, help="Input audio file (WAV, MP3, M4A, FLAC, etc.)")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("-m", "--model", type=str, default="satoripoyopoyo/SheetSage2-MLX", help="Model name on Hugging Face Hub or local path")
    parser.add_argument("-t", "--max-seconds", type=float, default=None, help="Max duration in seconds to transcribe")
    parser.add_argument("--no-midi", action="store_true", help="Disable MIDI file generation")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML sheet music generation")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🎵 Transcribing: {audio_path.name} ...")
    res = transcribe(
        audio_path=audio_path,
        output_dir=args.output_dir,
        max_seconds=args.max_seconds,
        render_midi=not args.no_midi,
        render_html=not args.no_html,
    )

    stats = res["stats"]
    print("\n✨ Transcription Complete!")
    print(f"  Duration:   {stats['audio_duration_sec']}s")
    print(f"  Speed:      {stats['speed_x']}x Realtime (Total: {stats['elapsed_sec']}s)")
    print(f"  Notes:      {stats['note_count']}")
    print(f"  Output Dir: {res['output_dir']}\n")
    print("Generated Files:")
    for name, p in res["files"].items():
        print(f"  - {name}: {p}")


if __name__ == "__main__":
    main()
