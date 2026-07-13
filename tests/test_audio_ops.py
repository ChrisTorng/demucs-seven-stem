from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demucs_seven_stem.audio_ops import (
    AudioOperationError,
    create_audio_output,
    discover_audio_files,
)

SAMPLE_RATE = 8_000


def _write(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sample_rate, format="WAV", subtype="DOUBLE")


def _read(path: Path) -> np.ndarray:
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    assert sample_rate == SAMPLE_RATE
    return data.T


def test_multiple_files_are_summed_sample_for_sample(tmp_path: Path) -> None:
    first = np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.0]], dtype=np.float64)
    second = np.array([[0.3, -0.1, 0.2], [-0.4, 0.5, 0.25]], dtype=np.float64)
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    output_path = tmp_path / "sum.wav"
    _write(first_path, first)
    _write(second_path, second)

    result = create_audio_output([first_path, second_path], output_path)

    assert result.operation == "sum"
    assert result.input_files == (first_path.resolve(), second_path.resolve())
    np.testing.assert_array_equal(_read(output_path), first + second)


def test_directory_inputs_are_sorted_and_output_is_excluded(tmp_path: Path) -> None:
    folder = tmp_path / "stems"
    audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    _write(folder / "z.wav", audio)
    _write(folder / "A.wav", audio)
    _write(folder / "existing-output.wav", audio)
    (folder / "ignore.txt").write_text("not audio", encoding="utf-8")

    files = discover_audio_files(
        [folder], excluded_paths=[folder / "existing-output.wav"]
    )

    assert [path.name for path in files] == ["A.wav", "z.wav"]


def test_reference_minus_folder_sum_creates_difference(tmp_path: Path) -> None:
    source = np.array([[0.8, -0.4, 0.2], [0.1, 0.3, -0.5]], dtype=np.float64)
    first = source * 0.25
    second = source * 0.5
    folder = tmp_path / "stems"
    reference_path = tmp_path / "source.wav"
    output_path = tmp_path / "difference.wav"
    _write(folder / "first.wav", first)
    _write(folder / "second.wav", second)
    _write(reference_path, source)

    result = create_audio_output(
        [folder], output_path, reference_path=reference_path
    )

    assert result.operation == "reference-minus-sum"
    np.testing.assert_allclose(_read(output_path), source - first - second, atol=1e-15)


def test_recursive_directory_search_is_optional(tmp_path: Path) -> None:
    nested = tmp_path / "stems" / "nested" / "track.wav"
    _write(nested, np.ones((2, 3), dtype=np.float64))

    with pytest.raises(AudioOperationError, match="No audio files"):
        discover_audio_files([tmp_path / "stems"])

    assert discover_audio_files([tmp_path / "stems"], recursive=True) == (
        nested.resolve(),
    )


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    _write(first_path, np.ones((2, 3), dtype=np.float64))
    _write(second_path, np.ones((2, 4), dtype=np.float64))

    with pytest.raises(AudioOperationError, match="Audio-shape mismatch"):
        create_audio_output([first_path, second_path], tmp_path / "sum.wav")


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "sum.wav"
    audio = np.ones((2, 3), dtype=np.float64)
    _write(input_path, audio)
    _write(output_path, np.zeros_like(audio))

    with pytest.raises(AudioOperationError, match="Use --overwrite"):
        create_audio_output([input_path], output_path)

    create_audio_output([input_path], output_path, overwrite=True)
    np.testing.assert_array_equal(_read(output_path), audio)
