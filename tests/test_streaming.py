import unittest

import numpy as np

from hearweave import StreamingDelayAndSum, delay_and_sum, glasses_4mic, stream_blocks
from hearweave.simulation import simulate_plane_wave, speech_like_probe


class StreamingTests(unittest.TestCase):
    def test_streaming_matches_offline_after_latency(self) -> None:
        sample_rate_hz = 16_000
        azimuth = 35.0
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.6)
        signals = simulate_plane_wave(
            probe, geometry, sample_rate_hz, azimuth, snr_db=None
        )
        offline = delay_and_sum(signals, geometry, sample_rate_hz, azimuth)

        streamer = StreamingDelayAndSum(geometry, sample_rate_hz, azimuth)
        blocks = [streamer.process_block(block) for block in stream_blocks(signals, 256)]
        streamed = np.concatenate(blocks)[: signals.shape[1]]

        latency = streamer.latency_samples
        # Ignore edges where offline zero-padding and streaming warm-up differ.
        margin = 32
        aligned_streamed = streamed[latency + margin : -margin]
        aligned_offline = offline[margin : -latency - margin]
        error = np.max(np.abs(aligned_streamed - aligned_offline))
        self.assertLess(error, 1e-6)

    def test_streaming_is_block_size_invariant(self) -> None:
        sample_rate_hz = 16_000
        geometry = glasses_4mic()
        probe = speech_like_probe(sample_rate_hz, 0.4)
        signals = simulate_plane_wave(probe, geometry, sample_rate_hz, -60.0, snr_db=None)

        outputs = []
        for block_size in (128, 512):
            streamer = StreamingDelayAndSum(geometry, sample_rate_hz, -60.0)
            blocks = [
                streamer.process_block(block) for block in stream_blocks(signals, block_size)
            ]
            outputs.append(np.concatenate(blocks)[: signals.shape[1]])
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-9)

    def test_streaming_latency_is_wearable_scale(self) -> None:
        streamer = StreamingDelayAndSum(glasses_4mic(), 16_000, 90.0)
        self.assertLessEqual(streamer.latency_samples, 10)

    def test_stream_blocks_pads_final_block(self) -> None:
        signals = np.ones((2, 500))
        blocks = list(stream_blocks(signals, 256))
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1].shape, (2, 256))
        self.assertEqual(float(blocks[1][:, 244:].sum()), 0.0)

    def test_invalid_block_shape_is_rejected(self) -> None:
        streamer = StreamingDelayAndSum(glasses_4mic(), 16_000, 0.0)
        with self.assertRaises(ValueError):
            streamer.process_block(np.zeros((3, 128)))


if __name__ == "__main__":
    unittest.main()
