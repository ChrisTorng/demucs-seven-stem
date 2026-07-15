"""Recursive Demucs separation with accumulated stems and residuals."""

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
    quantize_for_wav,
    ratio_db,
    rms,
    sum_stems,
)

LOGGER = logging.getLogger(__name__)
EXPECTED_SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")


class SeparationError(RuntimeError):
    """Raised when separation cannot be completed safely."""


@dataclass(frozen=True)
class SeparationConfig:
    """Runtime configuration for recursive separation and output retention."""

    model: str = "htdemucs_6s"
    model_repo: Path | None = None
    device: str = "auto"
    shifts: int = 1
    split: bool = True
    overlap: float = 0.25
    segment: int | None = None
    passes: int = 2
    # Compatibility with versions <= 0.2: total passes = residual_passes + 1.
    residual_passes: int | None = None
    write_accumulated_stems: bool = True
    keep_pass_stems: bool = False
    keep_pass_residuals: bool = False
    write_final_residual: bool = True
    wav_subtype: str = "DOUBLE"
    progress: bool = True

    @property
    def total_passes(self) -> int:
        if self.residual_passes is not None:
            return self.residual_passes + 1
        return self.passes

    def validate(self) -> None:
        if self.shifts < 0:
            raise ValueError("shifts must be zero or greater.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be at least 0 and less than 1.")
        if self.segment is not None and self.segment <= 0:
            raise ValueError("segment must be greater than zero.")
        if self.passes < 1:
            raise ValueError("passes must be one or greater.")
        if self.residual_passes is not None and self.residual_passes < 0:
            raise ValueError("residual_passes must be zero or greater.")
        if self.wav_subtype.upper() not in {"DOUBLE", "FLOAT"}:
            raise ValueError("wav_subtype must be DOUBLE or FLOAT.")


@dataclass(frozen=True)
class TrackResult:
    """Result summary for one input file."""

    input_path: Path
    output_path: Path
    pass_manifests: tuple[dict[str, Any], ...]
    final_manifest: dict[str, Any]


class DemucsBackend:
    """Thin, testable wrapper around the public Demucs Python modules."""

    def __init__(self, config: SeparationConfig) -> None:
        config.validate()
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.audio import AudioFile
            from demucs.pretrained import get_model
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

        # PyTorch >= 2.6 defaults torch.load to weights_only=True. Official Demucs
        # checkpoints contain trusted training metadata and require the legacy mode.
        original_torch_load = torch.load

        def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = compatible_torch_load
        try:
            repo = config.model_repo.expanduser().resolve() if config.model_repo else None
            model = get_model(config.model, repo=repo)
        finally:
            torch.load = original_torch_load

        self.device = device
        self._torch = torch
        self._apply_model = apply_model
        self._audio_file = AudioFile
        self._model = model.to(device).eval()
        self._config = config
        self.sources = tuple(self._model.sources)
        self.sample_rate = int(self._model.samplerate)
        self.audio_channels = int(self._model.audio_channels)

        if len(self.sources) != 6 or set(self.sources) != set(EXPECTED_SOURCES):
            raise SeparationError(
                f"Model {config.model!r} returned sources {self.sources}; "
                f"this tool requires the six sources {EXPECTED_SOURCES}."
            )

    def separate_file(self, path: Path) -> tuple[AudioArray, dict[str, AudioArray]]:
        wav = self._audio_file(path).read(
            streams=0,
            samplerate=self.sample_rate,
            channels=self.audio_channels,
        )
        source = as_audio64(wav.detach().cpu().numpy())
        return source, self.separate_array(source)

    def separate_array(self, audio: np.ndarray) -> dict[str, AudioArray]:
        source = as_audio64(audio)
        mix = self._torch.from_numpy(
            np.ascontiguousarray(source.astype(np.float32))
        ).to(self.device)
        reference = mix.mean(0)
        mean = reference.mean()
        std = reference.std()
        if float(std) == 0.0:
            raise SeparationError(
                "Cannot separate digital silence: input standard deviation is zero."
            )
        normalized = (mix - mean) / std

        with self._torch.inference_mode():
            output = self._apply_model(
                self._model,
                normalized[None],
                shifts=self._config.shifts,
                split=self._config.split,
                overlap=self._config.overlap,
                progress=self._config.progress,
                device=self.device,
                num_workers=0,
                segment=self._config.segment,
            )[0]
        raw = output.detach().cpu().numpy().copy().astype(np.float64)
        restored = raw * float(std) + float(mean)
        return {
            name: as_audio64(restored[index])
            for index, name in enumerate(self.sources)
        }


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    try:
        import soundfile as sf
    except ImportError as error:
        raise SeparationError("The soundfile package is required to write WAV files.") from error

    path.parent.mkdir(parents=True, exist_ok=True)
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
    reconstruction_error_peak: float,
    sample_rate: int,
    wav_subtype: str,
    kept_stems: bool,
    kept_residual: bool,
) -> dict[str, Any]:
    return {
        "pass_index": pass_index,
        "input": "original mixture" if pass_index == 0 else "previous pass residual",
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
            "in_memory_peak_error": reconstruction_error_peak,
        },
        "files": {
            "stems_kept": kept_stems,
            "residual_kept": kept_residual,
        },
    }


def _verify_final_written_sum(
    track_dir: Path,
    source: np.ndarray,
    source_order: tuple[str, ...],
    sample_rate: int,
) -> float:
    reconstructed = np.zeros_like(as_audio64(source))
    for name in source_order:
        reconstructed += _read_wav(track_dir / f"{name}.wav", sample_rate)
    reconstructed += _read_wav(track_dir / "final_residual.wav", sample_rate)
    return peak(as_audio64(source) - reconstructed)


def separate_track(
    input_path: Path,
    output_root: Path,
    backend: DemucsBackend,
    config: SeparationConfig,
    *,
    overwrite: bool = False,
) -> TrackResult:
    """Run recursive passes, accumulate same-name stems, and write selected outputs."""

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

    LOGGER.info(
        "Separating %s on %s for %d pass(es)",
        input_path,
        backend.device,
        config.total_passes,
    )
    original_source, raw_stems = backend.separate_file(input_path)
    pass_source = original_source
    source_order = tuple(backend.sources)
    accumulated = {
        name: np.zeros_like(original_source, dtype=np.float64) for name in source_order
    }
    manifests: list[dict[str, Any]] = []
    last_residual: AudioArray | None = None

    for pass_index in range(config.total_passes):
        prepared = prepare_level(
            pass_source,
            raw_stems,
            source_order,
            config.wav_subtype,
        )
        for name in source_order:
            accumulated[name] += prepared.stems[name]
        last_residual = prepared.residual

        pass_dir: Path | None = None
        if config.keep_pass_stems or config.keep_pass_residuals:
            pass_dir = track_dir / f"pass_{pass_index:02d}"
            pass_dir.mkdir()
            if config.keep_pass_stems:
                for name in source_order:
                    _write_wav(
                        pass_dir / f"{name}.wav",
                        prepared.stems[name],
                        backend.sample_rate,
                        config.wav_subtype,
                    )
            if config.keep_pass_residuals:
                _write_wav(
                    pass_dir / "residual.wav",
                    prepared.residual,
                    backend.sample_rate,
                    config.wav_subtype,
                )

        manifest = _pass_manifest(
            pass_index=pass_index,
            source=pass_source,
            stems=prepared.stems,
            residual=prepared.residual,
            reconstruction_error_peak=peak(prepared.reconstruction_error),
            sample_rate=backend.sample_rate,
            wav_subtype=config.wav_subtype,
            kept_stems=config.keep_pass_stems,
            kept_residual=config.keep_pass_residuals,
        )
        manifests.append(manifest)
        if pass_dir is not None:
            _atomic_json(pass_dir / "manifest.json", manifest)

        LOGGER.info(
            "pass_%02d residual %.2f dB RMS relative to pass input",
            pass_index,
            manifest["residual"]["rms_relative_to_source_db"],
        )

        if pass_index + 1 < config.total_passes:
            # Feed the exact samples implied by the configured output precision.
            pass_source = prepared.residual
            raw_stems = backend.separate_array(pass_source)

    if last_residual is None:  # defensive; validation requires at least one pass
        raise SeparationError("No separation pass was executed.")

    final_accumulated = {
        name: quantize_for_wav(accumulated[name], config.wav_subtype)
        for name in source_order
    }
    final_stem_sum = sum_stems(final_accumulated, source_order, original_source.shape)
    final_residual = quantize_for_wav(
        original_source - final_stem_sum,
        config.wav_subtype,
    )
    final_reconstruction_error = original_source - (final_stem_sum + final_residual)
    residual_difference = final_residual - last_residual

    if config.write_accumulated_stems:
        for name in source_order:
            _write_wav(
                track_dir / f"{name}.wav",
                final_accumulated[name],
                backend.sample_rate,
                config.wav_subtype,
            )
    if config.write_final_residual:
        _write_wav(
            track_dir / "final_residual.wav",
            final_residual,
            backend.sample_rate,
            config.wav_subtype,
        )

    written_reconstruction_error: float | None = None
    if config.write_accumulated_stems and config.write_final_residual:
        written_reconstruction_error = _verify_final_written_sum(
            track_dir,
            original_source,
            source_order,
            backend.sample_rate,
        )

    final_manifest = {
        "accumulated_stems": {
            "written": config.write_accumulated_stems,
            "stats": {
                name: audio_stats(audio) for name, audio in final_accumulated.items()
            },
        },
        "final_residual": {
            "written": config.write_final_residual,
            **audio_stats(final_residual),
            "rms_relative_to_original_db": ratio_db(final_residual, original_source),
        },
        "reconstruction": {
            "in_memory_peak_error": peak(final_reconstruction_error),
            "written_files_peak_error": written_reconstruction_error,
        },
        "last_pass_residual_equivalence": {
            "comparison": "final_residual - last_pass_residual",
            "peak_difference": peak(residual_difference),
            "rms_difference": rms(residual_difference),
            "difference_relative_to_last_residual_db": ratio_db(
                residual_difference,
                last_residual,
            ),
            "mathematically_expected": (
                "Equal except for floating-point accumulation and output-precision quantization."
            ),
        },
    }

    configuration = asdict(config)
    if configuration["model_repo"] is not None:
        configuration["model_repo"] = str(configuration["model_repo"])
    configuration["resolved_total_passes"] = config.total_passes

    top_manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_directory": str(track_dir),
        "model": config.model,
        "device": backend.device,
        "source_order": list(source_order),
        "configuration": configuration,
        "passes": manifests,
        "final": final_manifest,
    }
    _atomic_json(track_dir / "manifest.json", top_manifest)

    return TrackResult(
        input_path=input_path,
        output_path=track_dir,
        pass_manifests=tuple(manifests),
        final_manifest=final_manifest,
    )
