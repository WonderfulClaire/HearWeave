import unittest

import numpy as np

from hearweave import (
    apply_microphone_mismatch,
    delay_and_sum,
    glasses_4mic,
    scan_azimuth_energy,
    simulate_plane_wave,
    srp_phat,
)
from hearweave.metrics import snr_db
from hearweave.simulation import speech_like_probe


def circular_error_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


class ProcessingTests(unittest.TestCase):
    def test_scan_recovers_simulated_direction(self) -> None:
        sample_rate_hz = 16_000
        target = 42.0
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.7)
        signals = simulate_plane_wave(
            probe,
            geometry,
            sample_rate_hz,
            target,
            snr_db=24.0,
            rng=np.random.default_rng(3),
        )
        estimate, scores, grid = scan_azimuth_energy(signals, geometry, sample_rate_hz)
        self.assertEqual(scores.shape, grid.shape)
        self.assertLessEqual(circular_error_degrees(estimate, target), 4.0)

    def test_delay_and_sum_improves_independent_noise(self) -> None:
        sample_rate_hz = 16_000
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.8)
        signals = simulate_plane_wave(
            probe,
            geometry,
            sample_rate_hz,
            20.0,
            snr_db=1.0,
            rng=np.random.default_rng(9),
        )
        enhanced = delay_and_sum(signals, geometry, sample_rate_hz, 20.0)
        margin = 64
        input_snr = snr_db(probe[margin:-margin], signals[0, margin:-margin])
        output_snr = snr_db(probe[margin:-margin], enhanced[margin:-margin])
        self.assertGreater(output_snr, input_snr + 2.0)

    def test_srp_phat_recovers_simulated_direction(self) -> None:
        sample_rate_hz = 16_000
        target = -75.0
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.7)
        signals = simulate_plane_wave(
            probe,
            geometry,
            sample_rate_hz,
            target,
            snr_db=10.0,
            rng=np.random.default_rng(5),
        )
        estimate, scores, grid = srp_phat(signals, geometry, sample_rate_hz)
        self.assertEqual(scores.shape, grid.shape)
        self.assertLessEqual(circular_error_degrees(estimate, target), 4.0)

    def test_srp_phat_survives_microphone_mismatch(self) -> None:
        sample_rate_hz = 16_000
        target = 42.0
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.7)
        signals = simulate_plane_wave(
            probe,
            geometry,
            sample_rate_hz,
            target,
            snr_db=18.0,
            rng=np.random.default_rng(11),
        )
        degraded = apply_microphone_mismatch(
            signals,
            sample_rate_hz,
            gain_std_db=2.0,
            delay_jitter_std_s=8e-6,
            rng=np.random.default_rng(12),
        )
        estimate, _, _ = srp_phat(degraded, geometry, sample_rate_hz)
        self.assertLessEqual(circular_error_degrees(estimate, target), 8.0)

    def test_mismatch_changes_channels_but_keeps_shape(self) -> None:
        signals = np.tile(np.sin(np.linspace(0, 20, 4_000)), (4, 1))
        degraded = apply_microphone_mismatch(
            signals, 16_000, rng=np.random.default_rng(1)
        )
        self.assertEqual(degraded.shape, signals.shape)
        self.assertFalse(np.allclose(degraded, signals))

    def test_mismatch_rejects_negative_spread(self) -> None:
        with self.assertRaises(ValueError):
            apply_microphone_mismatch(np.zeros((2, 100)), 16_000, gain_std_db=-1.0)


if __name__ == "__main__":
    unittest.main()
