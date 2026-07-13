"""Sample-aligned audio summation and reference-difference operations."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_math import AudioArray, AudioShapeError, as_audio64, peak, quantize_for_wav

LOGGER = logging.getLogger(__name__)
DIRECTORY_AUDIO_EXTENSIONS = frozenset(
    {
        ".aif",
        ".aiff",
        ".au",
        ".caf",
        ".flac",
        ".mp3",
        ".oga",
        ".ogg",
        ".rf64",
        ".snd",
        ".wav",
        ".w64",
    }
)


class AudioOperationError(RuntimeError):
    """Raised when audio files cannot be combined sample-for-sample."""


@dataclass(frozen=True)
class AudioFileData:
    """Decoded audio and format metadata."""

    path: Path
    audio: AudioArray
    sample_rate: int


@dataclass(frozen=True)
class AudioOperationResult:
    """Summary of one summation or reference-difference operation."""

    operation: str
    input_files: tuple[Path, ...]
    reference_path: Path | None
    output_path: Path
    sample_rate: int
    channels: int
    samples: int
    output_peak: float


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_audio_files(
    inputs: Sequence[Path],
    *,
    recursive: bool = False,
    excluded_paths: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Expand explicit files and directories into a deterministic file list.

    Explicit files are accepted regardless of extension and decoded by
    libsndfile. Directory entries are filtered to common audio extensions.
    """

    excluded = {_canonical(path) for path in excluded_paths}
    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        input_path = _canonical(raw_path)
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            iterator = input_path.rglob("*") if recursive else input_path.iterdir()
            candidates = sorted(
                (
                    path.resolve()
                    for path in iterator
                    if path.is_file() and path.suffix.lower() in DIRECTORY_AUDIO_EXTENSIONS
                ),
                key=lambda path: str(path).casefold(),
            )
        else:
            raise AudioOperationError(f"Input path does not exist: {input_path}")

        for candidate in candidates:
            if candidate in excluded or candidate in seen:
                continue
            seen.add(candidate)
            discovered.append(candidate)

    if not discovered:
        raise AudioOperationError("No audio files were found in the supplied inputs.")
    return tuple(discovered)


def read_audio_file(path: Path) -> AudioFileData:
    """Decode one file as channel-first float64 audio."""

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as error:
        raise AudioOperationError(f"Could not decode audio file {path}: {error}") from error

    try:
        audio = as_audio64(data.T)
    except AudioShapeError as error:
        raise AudioOperationError(f"Invalid audio data in {path}: {error}") from error
    return AudioFileData(path=path, audio=audio, sample_rate=int(sample_rate))


def _require_compatible(reference: AudioFileData, candidate: AudioFileData) -> None:
    if candidate.sample_rate != reference.sample_rate:
        raise AudioOperationError(
            f"Sample-rate mismatch: {candidate.path} is {candidate.sample_rate} Hz; "
            f"expected {reference.sample_rate} Hz from {reference.path}."
        )
    if candidate.audio.shape != reference.audio.shape:
        raise AudioOperationError(
            f"Audio-shape mismatch: {candidate.path} is {candidate.audio.shape} "
            f"[channels, samples]; expected {reference.audio.shape} from {reference.path}."
        )


def sum_audio_files(files: Sequence[Path]) -> tuple[AudioArray, int]:
    """Sum compatible files in deterministic order using float64 accumulation."""

    if not files:
        raise AudioOperationError("At least one audio file is required.")

    first = read_audio_file(files[0])
    total = np.zeros_like(first.audio, dtype=np.float64)
    total += first.audio

    for path in files[1:]:
        current = read_audio_file(path)
        _require_compatible(first, current)
        total += current.audio
    return total, first.sample_rate


def _write_float_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    if path.suffix.lower() != ".wav":
        raise AudioOperationError(
            f"Output must use the .wav extension for floating-point output: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        import soundfile as sf

        sf.write(
            str(temporary),
            as_audio64(audio).T,
            sample_rate,
            format="WAV",
            subtype=subtype.upper(),
        )
        temporary.replace(path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise AudioOperationError(f"Could not write output file {path}: {error}") from error


def create_audio_output(
    inputs: Sequence[Path],
    output_path: Path,
    *,
    reference_path: Path | None = None,
    recursive: bool = False,
    wav_subtype: str = "DOUBLE",
    overwrite: bool = False,
) -> AudioOperationResult:
    """Create ``sum(inputs)`` or ``reference - sum(inputs)`` as a float WAV."""

    output_path = _canonical(output_path)
    reference_path = _canonical(reference_path) if reference_path is not None else None
    if output_path.suffix.lower() != ".wav":
        raise AudioOperationError(
            f"Output must use the .wav extension for floating-point output: {output_path}"
        )
    if reference_path is not None and output_path == reference_path:
        raise AudioOperationError("Reference and output paths must be different.")
    files = discover_audio_files(
        inputs,
        recursive=recursive,
        excluded_paths=(
            path for path in (output_path, reference_path) if path is not None
        ),
    )
    LOGGER.info("Combining %d audio file(s)", len(files))
    summed, sample_rate = sum_audio_files(files)

    if reference_path is None:
        operation = "sum"
        output = summed
    else:
        if not reference_path.is_file():
            raise AudioOperationError(f"Reference file does not exist: {reference_path}")
        reference = read_audio_file(reference_path)
        if sample_rate != reference.sample_rate:
            raise AudioOperationError(
                f"Sample-rate mismatch: input sum is {sample_rate} Hz; "
                f"reference {reference_path} is {reference.sample_rate} Hz."
            )
        if summed.shape != reference.audio.shape:
            raise AudioOperationError(
                f"Audio-shape mismatch: input sum is {summed.shape} [channels, samples]; "
                f"reference {reference_path} is {reference.audio.shape}."
            )
        operation = "reference-minus-sum"
        output = reference.audio - summed

    output = quantize_for_wav(output, wav_subtype)
    if output_path.exists():
        if not overwrite:
            raise AudioOperationError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )
    _write_float_wav(output_path, output, sample_rate, wav_subtype)

    return AudioOperationResult(
        operation=operation,
        input_files=files,
        reference_path=reference_path,
        output_path=output_path,
        sample_rate=sample_rate,
        channels=int(output.shape[0]),
        samples=int(output.shape[1]),
        output_peak=peak(output),
    )
