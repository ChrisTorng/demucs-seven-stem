"""Numerically stable audio preparation helpers.

All arrays use Demucs' channel-first layout: ``[channels, samples]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float64]


class AudioShapeError(ValueError):
    """Raised when stems cannot be combined sample-for-sample."""


@dataclass(frozen=True)
class PreparedLevel:
    """Audio for one seven-track separation level."""

    stems: dict[str, AudioArray]
    residual: AudioArray
    reconstructed: AudioArray
    reconstruction_error: AudioArray


def as_audio64(audio: np.ndarray) -> AudioArray:
    """Return a finite, channel-first float64 audio array."""

    result = np.asarray(audio, dtype=np.float64)
    if result.ndim != 2:
        raise AudioShapeError(
            f"Audio must have shape [channels, samples], got {result.shape}."
        )
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise AudioShapeError(f"Audio cannot be empty, got {result.shape}.")
    if not np.isfinite(result).all():
        raise AudioShapeError("Audio contains NaN or infinity.")
    return np.ascontiguousarray(result)


def quantize_for_wav(audio: np.ndarray, subtype: str) -> AudioArray:
    """Model the samples that will be read back from a float WAV file."""

    audio64 = as_audio64(audio)
    normalized_subtype = subtype.upper()
    if normalized_subtype == "DOUBLE":
        return audio64.copy()
    if normalized_subtype == "FLOAT":
        return audio64.astype(np.float32).astype(np.float64)
    raise ValueError("WAV subtype must be DOUBLE or FLOAT.")


def sum_stems(
    stems: Mapping[str, np.ndarray],
    source_order: Sequence[str],
    expected_shape: tuple[int, int],
) -> AudioArray:
    """Sum stems in a deterministic order using float64 accumulation."""

    total = np.zeros(expected_shape, dtype=np.float64)
    for name in source_order:
        try:
            stem = as_audio64(stems[name])
        except KeyError as error:
            raise AudioShapeError(f"Missing stem: {name}") from error
        if stem.shape != expected_shape:
            raise AudioShapeError(
                f"Stem {name!r} has shape {stem.shape}; expected {expected_shape}."
            )
        total += stem
    return total


def prepare_level(
    source: np.ndarray,
    raw_stems: Mapping[str, np.ndarray],
    source_order: Sequence[str],
    wav_subtype: str,
) -> PreparedLevel:
    """Quantize six stems, create the seventh residual, and verify reconstruction.

    The residual is calculated *after* the model stems are converted to their
    eventual WAV precision. Therefore the next recursive pass processes exactly
    the samples stored in ``residual.wav``.
    """

    source64 = as_audio64(source)
    prepared_stems = {
        name: quantize_for_wav(raw_stems[name], wav_subtype) for name in source_order
    }
    stem_sum = sum_stems(prepared_stems, source_order, source64.shape)
    residual = quantize_for_wav(source64 - stem_sum, wav_subtype)

    reconstructed = stem_sum + residual
    error = source64 - reconstructed
    return PreparedLevel(
        stems=prepared_stems,
        residual=residual,
        reconstructed=reconstructed,
        reconstruction_error=error,
    )


def peak(audio: np.ndarray) -> float:
    """Maximum absolute sample value."""

    return float(np.max(np.abs(as_audio64(audio))))


def rms(audio: np.ndarray) -> float:
    """Root mean square across all channels and samples."""

    value = as_audio64(audio)
    return float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))


def amplitude_db(value: float, *, floor_db: float = -300.0) -> float:
    """Convert a non-negative amplitude to dB relative to full scale."""

    if value <= 0.0:
        return floor_db
    return max(20.0 * math.log10(value), floor_db)


def ratio_db(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """RMS ratio in dB, with -300 dB representing digital silence."""

    numerator_rms = rms(numerator)
    denominator_rms = rms(denominator)
    if numerator_rms == 0.0:
        return -300.0
    if denominator_rms == 0.0:
        return math.inf
    return 20.0 * math.log10(numerator_rms / denominator_rms)


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    """Serializable peak and RMS statistics."""

    audio_peak = peak(audio)
    audio_rms = rms(audio)
    return {
        "peak": audio_peak,
        "peak_dbfs": amplitude_db(audio_peak),
        "rms": audio_rms,
        "rms_dbfs": amplitude_db(audio_rms),
    }
