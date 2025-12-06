"""
Command-line interface for UTMOSv2 inference.

Usage:
    utmosv2-predict audio.wav
    utmosv2-predict audio1.wav audio2.wav audio3.wav
    utmosv2-predict *.wav --device cuda
"""

import argparse
import sys
from pathlib import Path

from .predictor import UTMOSv2Predictor


def main() -> int:
    """Main entry point for utmosv2-predict command."""
    parser = argparse.ArgumentParser(
        description="Predict speech quality (MOS) using UTMOSv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    utmosv2-predict audio.wav
    utmosv2-predict audio1.wav audio2.wav
    utmosv2-predict *.wav --device cuda
    utmosv2-predict audio.wav --seed 42
        """,
    )
    parser.add_argument(
        "audio_files",
        type=Path,
        nargs="+",
        help="Audio file(s) to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, mps). Default: cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for models (default: uses TORCH_HOME)",
    )
    parser.add_argument(
        "--ssl-model",
        type=Path,
        default=None,
        help="Path to SSL TorchScript model (optional)",
    )
    parser.add_argument(
        "--fusion-model",
        type=Path,
        default=None,
        help="Path to fusion TorchScript model (optional)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output scores, no headers",
    )
    args = parser.parse_args()

    # Validate audio files
    for audio_path in args.audio_files:
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            return 1

    # Load predictor
    if not args.quiet:
        print("Loading UTMOSv2 model...")

    try:
        predictor = UTMOSv2Predictor(
            ssl_model_path=args.ssl_model,
            fusion_model_path=args.fusion_model,
            cache_dir=args.cache_dir,
            device=args.device,
            auto_download=True,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Models not found. They will be downloaded automatically on first use.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print()

    # Process each file
    for i, audio_path in enumerate(args.audio_files):
        file_seed = (args.seed + i) if args.seed is not None else None

        try:
            score = predictor.predict(audio_path, seed=file_seed)

            if args.quiet:
                print(f"{score:.4f}")
            else:
                print(f"{audio_path}: {score:.4f}")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
