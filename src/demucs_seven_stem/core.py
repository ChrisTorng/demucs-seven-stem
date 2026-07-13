"""Demucs separation pipeline with a reconstruction residual track."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audio_math import (
    AudioArray,
    as_audio64,
    audio_stats,
    peak,
    prepare_level,
    ratio_db,
)

LOGGER = logging.getLogger(__name__)
EXPECTED_SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")


class SeparationError(RuntimeError):
    """Raised when separation cannot be completed safely."""


@dataclass(frozen=True)
class SeparationConfig:
    """Runtime configuration for Demucs and residual recursion."""

    model: str = "htdemucs_6s"
    device: str = "auto"
    shifts: int = 1
    split: bool = True
    overlap: float = 0.25
    segment: int | None = None
    residual_passes: int = 0
    wav_subtype: str = "DOUBLE"
    progress: bool = True

    def validate(self) -> None:
        if self.shifts < 0:
            raise ValueError("shifts must be zero or greater.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be at least 0 and less than 1.")
        if self.segment is not None and self.segment <= 0:
            raise ValueError("segment must be greater than zero.")
        if self.residual_passes < 0:
            raise ValueError("residual_passes must be zero or greater.")
        if self.wav_subtype.upper() not in {"DOUBLE", "FLOAT"}:
            raise ValueError("wav_subtype must be DOUBLE or FLOAT.")


@dataclass(frozen=True)
class TrackResult:
    """Result summary for one input file."""

    input_path: Path
    output_path: Path
    pass_manifests: tuple[dict[str, Any], ...]


class DemucsBackend:
    """Thin, testable wrapper around ``demucs.api.Separator``."""

    def __init__(self, config: SeparationConfig) -> None:
        config.validate()
        try:
            import torch
            from demucs.api import Separator
        except ImportError as error:
            raise SeparationError(
                "Demucs dependencies are unavailable. Install the project with "
                "`python -m pip install -e .`."
            ) from error

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise SeparationError(
                "CUDA was requested, but PyTorch cannot access a CUDA device."
            )

        self.device = device
        self._torch = torch
        self._separator = Separator(
            model=config.model,
            device=device,
            shifts=config.shifts,
            split=config.split,
            overlap=config.overlap,
            segment=config.segment,
            progress=config.progress,
        )
        self.sources = tuple(self._separator.model.sources)
        self.sample_rate = int(self._separator.samplerate)

        if len(self.sources) != 6 or set(self.sources) != set(EXPECTED_SOURCES):
            raise SeparationError(
                f"Model {config.model!r} returned sources {self.sources}; "
                f"this tool requires the six sources {EXPECTED_SOURCES}."
            )

    @staticmethod
    def _to_array(tensor: Any) -> AudioArray:
        return as_audio64(tensor.detach().cpu().numpy())

    def separate_file(self, path: Path) -> tuple[AudioArray, dict[str, AudioArray]]:
        source, stems = self._separator.separate_audio_file(path)
        return self._to_array(source), {
            name: self._to_array(stems[name]) for name in self.sources
        }

    def separate_array(self, audio: np.ndarray) -> dict[str, AudioArray]:
        # Demucs expects float32 [channels, samples] and mutates its local tensor
        # during normalization, so always provide a private contiguous copy.
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(as_audio64(audio).astype(np.float32))
        )
        _, stems = self._separator.separate_tensor(tensor, sr=self.sample_rate)
        return {name: self._to_array(stems[name]) for name in self.sources}


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    try:
        import soundfile as sf
    except ImportError as error:
        raise SeparationError("The soundfile package is required to write WAV files.") from error

    sf.write(
        str(path),
        as_audio64(audio).T,
        sample_rate,
        format="WAV",
        subtype=subtype.upper(),
    )


def _read_wav(path: Path, expected_sample_rate: int) -> AudioArray:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise SeparationError(
            f"Unexpected sample rate in {path}: {sample_rate}, expected {expected_sample_rate}."
        )
    return as_audio64(data.T)


def _verify_written_sum(
    pass_dir: Path,
    source: np.ndarray,
    source_order: tuple[str, ...],
    sample_rate: int,
) -> float:
    source64 = as_audio64(source)
    reconstructed = np.zeros_like(source64)
    for name in (*source_order, "residual"):
        audio = _read_wav(pass_dir / f"{name}.wav", sample_rate)
        if audio.shape != source64.shape:
            raise SeparationError(
                f"Written file {name}.wav has shape {audio.shape}; expected {source64.shape}."
            )
        reconstructed += audio
    return peak(source64 - reconstructed)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pass_manifest(
    *,
    pass_index: int,
    source: np.ndarray,
    stems: dict[str, AudioArray],
    residual: np.ndarray,
    in_memory_error_peak: float,
    written_error_peak: float,
    sample_rate: int,
    wav_subtype: str,
) -> dict[str, Any]:
    return {
        "pass_index": pass_index,
        "input": "original mixture" if pass_index == 0 else "previous pass residual.wav",
        "sample_rate": sample_rate,
        "channels": int(source.shape[0]),
        "samples": int(source.shape[1]),
        "wav_subtype": wav_subtype.upper(),
        "source": audio_stats(source),
        "stems": {name: audio_stats(audio) for name, audio in stems.items()},
        "residual": {
            **audio_stats(residual),
            "rms_relative_to_source_db": ratio_db(residual, source),
        },
        "reconstruction": {
            "in_memory_peak_error": in_memory_error_peak,
            "written_files_peak_error": written_error_peak,
        },
    }


def separate_track(
    input_path: Path,
    output_root: Path,
    backend: DemucsBackend,
    config: SeparationConfig,
    *,
    overwrite: bool = False,
) -> TrackResult:
    """Separate one file into six model stems plus a residual at each pass."""

    config.validate()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise SeparationError(f"Input file does not exist: {input_path}")

    track_dir = output_root.expanduser().resolve() / input_path.stem
    if track_dir.exists():
        if not overwrite:
            raise SeparationError(
                f"Output directory already exists: {track_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(track_dir)
    track_dir.mkdir(parents=True)

    LOGGER.info("Separating %s on %s", input_path, backend.device)
    source, raw_stems = backend.separate_file(input_path)
    source_order = tuple(backend.sources)
    manifests: list[dict[str, Any]] = []

    for pass_index in range(config.residual_passes + 1):
        prepared = prepare_level(
            source,
            raw_stems,
            source_order,
            config.wav_subtype,
        )
        pass_dir = track_dir / f"pass_{pass_index:02d}"
        pass_dir.mkdir()

        for name in source_order:
            _write_wav(
                pass_dir / f"{name}.wav",
                prepared.stems[name],
                backend.sample_rate,
                config.wav_subtype,
            )
        _write_wav(
            pass_dir / "residual.wav",
            prepared.residual,
            backend.sample_rate,
            config.wav_subtype,
        )

        in_memory_error_peak = peak(prepared.reconstruction_error)
        written_error_peak = _verify_written_sum(
            pass_dir,
            source,
            source_order,
            backend.sample_rate,
        )
        manifest = _pass_manifest(
            pass_index=pass_index,
            source=source,
            stems=prepared.stems,
            residual=prepared.residual,
            in_memory_error_peak=in_memory_error_peak,
            written_error_peak=written_error_peak,
            sample_rate=backend.sample_rate,
            wav_subtype=config.wav_subtype,
        )
        _atomic_json(pass_dir / "manifest.json", manifest)
        manifests.append(manifest)

        LOGGER.info(
            "pass_%02d residual %.2f dB RMS relative to source; file-sum peak error %.3g",
            pass_index,
            manifest["residual"]["rms_relative_to_source_db"],
            written_error_peak,
        )

        if pass_index < config.residual_passes:
            # Process exactly the samples written as this pass' seventh track.
            source = prepared.residual
            raw_stems = backend.separate_array(source)

    top_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_directory": str(track_dir),
        "model": config.model,
        "device": backend.device,
        "source_order": list(source_order),
        "configuration": asdict(config),
        "passes": manifests,
    }
    _atomic_json(track_dir / "manifest.json", top_manifest)

    return TrackResult(
        input_path=input_path,
        output_path=track_dir,
        pass_manifests=tuple(manifests),
    )
