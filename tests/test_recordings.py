from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from hearweave import load_recording


class RecordingAdapterTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        sample_rate_hz = 16_000
        left = np.array([0, 16384, -16384], dtype=np.int16)
        right = np.array([0, 8192, -8192], dtype=np.int16)
        wavfile.write(root / "left.wav", sample_rate_hz, left)
        wavfile.write(root / "right.wav", sample_rate_hz, right)
        manifest = {
            "schema_version": 1,
            "recording_id": "fixture-001",
            "sample_rate_hz": sample_rate_hz,
            "geometry_name": "test-headset",
            "channels": [
                {"label": "left", "file": "left.wav", "position_m": [-0.08, 0, 0]},
                {"label": "right", "file": "right.wav", "position_m": [0.08, 0, 0]},
            ],
            "evidence_scope": "synthetic test fixture",
        }
        path = root / "recording.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_loads_channels_in_manifest_order_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = load_recording(self._fixture(Path(directory)))
        self.assertEqual(recording.recording_id, "fixture-001")
        self.assertEqual(recording.signals.shape, (2, 3))
        self.assertEqual(recording.geometry.labels, ("left", "right"))
        self.assertAlmostEqual(recording.signals[0, 1], 0.5)
        self.assertEqual(recording.metadata["evidence_scope"], "synthetic test fixture")

    def test_rejects_sample_rate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            wavfile.write(root / "right.wav", 8_000, np.zeros(3, dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "sample rate mismatch"):
                load_recording(path)

    def test_rejects_channel_path_outside_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["channels"][0]["file"] = "../private.wav"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes the manifest directory"):
                load_recording(path)


if __name__ == "__main__":
    unittest.main()
