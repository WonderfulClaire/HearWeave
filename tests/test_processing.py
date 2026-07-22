import unittest

import numpy as np

from hearweave import delay_and_sum, glasses_4mic, scan_azimuth_energy, simulate_plane_wave
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


if __name__ == "__main__":
    unittest.main()
