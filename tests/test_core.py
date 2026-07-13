from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from demucs_seven_stem.core import (
    EXPECTED_SOURCES,
    SeparationConfig,
    separate_track,
)


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


def test_recursive_separation_writes_seven_tracks_per_pass(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"fake input; backend does not decode it")
    output_root = tmp_path / "out"
    backend = FakeBackend()
    config = SeparationConfig(
        residual_passes=1,
        wav_subtype="DOUBLE",
        progress=False,
    )

    result = separate_track(input_path, output_root, backend, config)

    assert result.output_path == output_root / "song"
    assert len(result.pass_manifests) == 2
    assert len(backend.recursive_inputs) == 1

    for pass_index in (0, 1):
        pass_dir = result.output_path / f"pass_{pass_index:02d}"
        assert pass_dir.is_dir()
        for name in (*EXPECTED_SOURCES, "residual"):
            assert (pass_dir / f"{name}.wav").is_file()
        manifest = json.loads((pass_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["reconstruction"]["written_files_peak_error"] == 0.0

    pass0_residual, sample_rate = sf.read(
        result.output_path / "pass_00" / "residual.wav",
        dtype="float64",
        always_2d=True,
    )
    assert sample_rate == backend.sample_rate
    np.testing.assert_array_equal(backend.recursive_inputs[0], pass0_residual.T)
