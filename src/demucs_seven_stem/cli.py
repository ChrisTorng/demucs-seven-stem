"""Command-line interface for demucs-seven-stem."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core import DemucsBackend, SeparationConfig, SeparationError, separate_track


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demucs-seven-stem",
        description=(
            "Run Demucs htdemucs_6s, save its six stems, and create a seventh "
            "residual track that reconstructs the pass input when summed."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input audio file(s).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("separated-seven-stem"),
        help="Output root directory (default: separated-seven-stem).",
    )
    parser.add_argument("--model", default="htdemucs_6s", help="Demucs model name.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--shifts",
        type=_non_negative_int,
        default=1,
        help="Random shift count used by Demucs (default: 1).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Chunk overlap from 0 inclusive to 1 exclusive (default: 0.25).",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Optional Demucs segment length in seconds.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable chunked processing; this can require much more VRAM.",
    )
    parser.add_argument(
        "--residual-passes",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Run N additional seven-track separations on the preceding residual. "
            "Use 1 to separate the first residual once."
        ),
    )
    parser.add_argument(
        "--wav-subtype",
        choices=("DOUBLE", "FLOAT"),
        default="DOUBLE",
        help=(
            "Floating-point WAV precision. DOUBLE best preserves reconstruction; "
            "FLOAT uses half the storage (default: DOUBLE)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per-track output directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Demucs progress bars.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = SeparationConfig(
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        split=not args.no_split,
        overlap=args.overlap,
        segment=args.segment,
        residual_passes=args.residual_passes,
        wav_subtype=args.wav_subtype,
        progress=not args.no_progress,
    )

    try:
        config.validate()
        backend = DemucsBackend(config)
        for input_path in args.inputs:
            result = separate_track(
                input_path,
                args.output,
                backend,
                config,
                overwrite=args.overwrite,
            )
            print(result.output_path)
        return 0
    except (SeparationError, ValueError, OSError) as error:
        logging.error("%s", error)
        return 2
    except KeyboardInterrupt:
        logging.error("Cancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
