import unittest

import numpy as np

from hearweave.geometry import ArrayGeometry, glasses_4mic, relative_arrival_delays


class GeometryTests(unittest.TestCase):
    def test_glasses_geometry_is_centered_and_wearable_scale(self) -> None:
        geometry = glasses_4mic()
        self.assertEqual(geometry.microphone_count, 4)
        np.testing.assert_allclose(geometry.center_m, 0.0)
        self.assertLess(0.14, geometry.aperture_m)
        self.assertGreater(0.18, geometry.aperture_m)

    def test_arrival_delays_are_non_negative(self) -> None:
        delays = relative_arrival_delays(glasses_4mic(), 35.0)
        self.assertAlmostEqual(float(delays.min()), 0.0)
        self.assertLess(float(delays.max()), 0.001)

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArrayGeometry("broken", np.zeros((4, 2)), ("a", "b", "c", "d"))


if __name__ == "__main__":
    unittest.main()
