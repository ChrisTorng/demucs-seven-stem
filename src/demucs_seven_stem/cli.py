"""Command-line interface for demucs-seven-stem."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audio_ops import AudioOperationError, create_audio_output
from .core import DemucsBackend, SeparationConfig, SeparationError, separate_track


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be one or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demucs-seven-stem",
        description=(
            "Run recursive Demucs six-stem separation, accumulate same-name stems, "
            "and create a final residual; or combine sample-aligned audio files."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Separation input file(s), or audio files/directories when "
            "--audio-output is used."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("separated-seven-stem"),
        help="Separation output root directory (default: separated-seven-stem).",
    )
    parser.add_argument(
        "--audio-output",
        type=Path,
        metavar="OUTPUT.wav",
        help=(
            "Audio-operation mode: sum all supplied files/folders into this floating-point WAV. "
            "With --reference, write reference minus the sum instead."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        metavar="REFERENCE",
        help="Reference audio for difference mode: output = reference - sum(inputs).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="In audio-operation mode, search supplied directories recursively.",
    )
    parser.add_argument("--model", default="htdemucs_6s", help="Demucs model name or signature.")
    parser.add_argument(
        "--model-repo",
        type=Path,
        help="Optional local Demucs model repository directory.",
    )
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
        "--passes",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Total recursive separation passes. Pass 1 uses the original; later passes use "
            "the preceding residual (default: 2)."
        ),
    )
    parser.add_argument(
        "--residual-passes",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-accumulated-stems",
        action="store_true",
        help="Do not write the final six same-name stems accumulated across all passes.",
    )
    parser.add_argument(
        "--keep-pass-stems",
        action="store_true",
        help="Keep each pass' six intermediate stems (default: discard them).",
    )
    parser.add_argument(
        "--keep-pass-residuals",
        "--keep-residuals",
        dest="keep_pass_residuals",
        action="store_true",
        help="Keep each pass' residual.wav (default: discard them).",
    )
    parser.add_argument(
        "--no-final-residual",
        action="store_true",
        help="Do not write original minus the six accumulated stems (default: write it).",
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
        help="Replace an existing separation directory or audio output file.",
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


def _resolve_passes(args: argparse.Namespace) -> tuple[int, int | None]:
    if args.passes is not None and args.residual_passes is not None:
        raise ValueError("Use either --passes or legacy --residual-passes, not both.")
    if args.residual_passes is not None:
        return 2, args.residual_passes
    return args.passes if args.passes is not None else 2, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.reference is not None and args.audio_output is None:
            raise AudioOperationError("--reference requires --audio-output.")

        if args.audio_output is not None:
            result = create_audio_output(
                args.inputs,
                args.audio_output,
                reference_path=args.reference,
                recursive=args.recursive,
                wav_subtype=args.wav_subtype,
                overwrite=args.overwrite,
            )
            logging.info(
                "%s: %d file(s), %d Hz, %d channel(s), %d samples, peak %.6g",
                result.operation,
                len(result.input_files),
                result.sample_rate,
                result.channels,
                result.samples,
                result.output_peak,
            )
            print(result.output_path)
            return 0

        passes, residual_passes = _resolve_passes(args)
        config = SeparationConfig(
            model=args.model,
            model_repo=args.model_repo,
            device=args.device,
            shifts=args.shifts,
            split=not args.no_split,
            overlap=args.overlap,
            segment=args.segment,
            passes=passes,
            residual_passes=residual_passes,
            write_accumulated_stems=not args.no_accumulated_stems,
            keep_pass_stems=args.keep_pass_stems,
            keep_pass_residuals=args.keep_pass_residuals,
            write_final_residual=not args.no_final_residual,
            wav_subtype=args.wav_subtype,
            progress=not args.no_progress,
        )
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
    except (AudioOperationError, SeparationError, ValueError, OSError) as error:
        logging.error("%s", error)
        return 2
    except KeyboardInterrupt:
        logging.error("Cancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
