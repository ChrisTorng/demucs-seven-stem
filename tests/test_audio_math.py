from __future__ import annotations

import numpy as np
import pytest

from demucs_seven_stem.audio_math import (
    AudioShapeError,
    prepare_level,
    ratio_db,
)

SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")


def _fixture_audio() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source = np.array(
        [[0.5, -0.25, 0.125, -0.0625], [0.25, 0.0, -0.125, 0.0625]],
        dtype=np.float32,
    ).astype(np.float64)
    stems = {
        name: np.full_like(source, (index + 1) / 128.0)
        for index, name in enumerate(SOURCES)
    }
    return source, stems


def test_double_residual_reconstructs_source_exactly() -> None:
    source, stems = _fixture_audio()
    level = prepare_level(source, stems, SOURCES, "DOUBLE")

    np.testing.assert_array_equal(level.reconstructed, source)
    np.testing.assert_array_equal(level.reconstruction_error, np.zeros_like(source))


def test_float_mode_matches_float_wav_precision() -> None:
    source, stems = _fixture_audio()
    stems["piano"] = stems["piano"] + 1.0 / 10.0
    level = prepare_level(source, stems, SOURCES, "FLOAT")

    assert np.max(np.abs(level.reconstruction_error)) <= np.finfo(np.float32).eps
    assert level.residual.dtype == np.float64


def test_missing_stem_is_rejected() -> None:
    source, stems = _fixture_audio()
    del stems["piano"]

    with pytest.raises(KeyError):
        prepare_level(source, stems, SOURCES, "DOUBLE")


def test_shape_mismatch_is_rejected() -> None:
    source, stems = _fixture_audio()
    stems["piano"] = stems["piano"][:, :-1]

    with pytest.raises(AudioShapeError):
        prepare_level(source, stems, SOURCES, "DOUBLE")


def test_ratio_db_for_identical_signal_is_zero() -> None:
    source, _ = _fixture_audio()
    assert ratio_db(source, source) == pytest.approx(0.0)
