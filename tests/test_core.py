from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from demucs_seven_stem.cli import _resolve_passes, build_parser
from demucs_seven_stem.core import EXPECTED_SOURCES, SeparationConfig, separate_track


class FakeBackend:
    device = "cpu"
    sample_rate = 8_000
    sources = EXPECTED_SOURCES

    def __init__(self) -> None:
        self.recursive_inputs: list[np.ndarray] = []

    @staticmethod
    def _initial_audio() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        source = np.array(
            [[0.5, -0.25, 0.125, -0.0625], [0.25, 0.0, -0.125, 0.0625]],
            dtype=np.float64,
        )
        stems = {name: np.zeros_like(source) for name in EXPECTED_SOURCES}
        stems["drums"] = source / 2.0
        return source, stems

    def separate_file(self, path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._initial_audio()

    def separate_array(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        self.recursive_inputs.append(audio.copy())
        stems = {name: np.zeros_like(audio) for name in EXPECTED_SOURCES}
        stems["other"] = audio / 4.0
        return stems


def _read(path: Path) -> np.ndarray:
    data, _ = sf.read(path, dtype="float64", always_2d=True)
    return data.T


def test_default_runs_two_passes_and_writes_only_accumulated_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"fake input; backend does not decode it")
    output_root = tmp_path / "out"
    backend = FakeBackend()
    config = SeparationConfig(wav_subtype="DOUBLE", progress=False)

    result = separate_track(input_path, output_root, backend, config)

    assert config.total_passes == 2
    assert len(result.pass_manifests) == 2
    assert len(backend.recursive_inputs) == 1
    assert not (result.output_path / "pass_00").exists()
    assert not (result.output_path / "pass_01").exists()

    source, _ = backend._initial_audio()
    np.testing.assert_array_equal(_read(result.output_path / "drums.wav"), source / 2.0)
    np.testing.assert_array_equal(_read(result.output_path / "other.wav"), source / 8.0)
    np.testing.assert_array_equal(
        _read(result.output_path / "final_residual.wav"),
        source * 3.0 / 8.0,
    )
    for name in EXPECTED_SOURCES:
        assert (result.output_path / f"{name}.wav").is_file()

    manifest = json.loads((result.output_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["resolved_total_passes"] == 2
    assert manifest["final"]["last_pass_residual_equivalence"]["peak_difference"] == 0.0
    assert manifest["final"]["reconstruction"]["written_files_peak_error"] == 0.0


def test_keep_options_write_each_pass_stems_and_residuals(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"fake")
    config = SeparationConfig(
        passes=2,
        keep_pass_stems=True,
        keep_pass_residuals=True,
        wav_subtype="DOUBLE",
        progress=False,
    )

    result = separate_track(input_path, tmp_path / "out", FakeBackend(), config)

    for pass_index in (0, 1):
        pass_dir = result.output_path / f"pass_{pass_index:02d}"
        assert (pass_dir / "manifest.json").is_file()
        assert (pass_dir / "residual.wav").is_file()
        for name in EXPECTED_SOURCES:
            assert (pass_dir / f"{name}.wav").is_file()


def test_output_switches_can_disable_accumulated_and_final_audio(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"fake")
    config = SeparationConfig(
        passes=1,
        write_accumulated_stems=False,
        write_final_residual=False,
        wav_subtype="DOUBLE",
        progress=False,
    )

    result = separate_track(input_path, tmp_path / "out", FakeBackend(), config)

    assert (result.output_path / "manifest.json").is_file()
    assert not (result.output_path / "drums.wav").exists()
    assert not (result.output_path / "final_residual.wav").exists()


def test_cli_defaults_to_two_total_passes_and_supports_legacy_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["song.wav"])
    assert _resolve_passes(args) == (2, None)

    legacy = parser.parse_args(["song.wav", "--residual-passes", "2"])
    assert _resolve_passes(legacy) == (2, 2)
    assert SeparationConfig(residual_passes=2).total_passes == 3
