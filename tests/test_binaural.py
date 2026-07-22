import unittest

import numpy as np

from hearweave.binaural import binaural_coherence_enhance


class BinauralTests(unittest.TestCase):
    def test_binaural_enhancer_preserves_shape_and_finite_values(self) -> None:
        rng = np.random.default_rng(2)
        shared = np.sin(2 * np.pi * 440 * np.arange(8_000) / 16_000)
        left = shared + 0.3 * rng.normal(size=shared.size)
        right = shared + 0.3 * rng.normal(size=shared.size)
        result = binaural_coherence_enhance(left, right, 16_000)
        self.assertEqual(result.shape, (2, shared.size))
        self.assertTrue(np.isfinite(result).all())

    def test_binaural_enhancer_rejects_mismatched_channels(self) -> None:
        with self.assertRaises(ValueError):
            binaural_coherence_enhance(np.zeros(10), np.zeros(11), 16_000)


if __name__ == "__main__":
    unittest.main()
