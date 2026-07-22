"""Microphone-array geometry presets for smart wearable devices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ArrayGeometry:
    """A microphone array expressed in metres in a right-handed xyz frame."""

    name: str
    positions_m: FloatArray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape (microphones, 3)")
        if positions.shape[0] < 2:
            raise ValueError("an array needs at least two microphones")
        if len(self.labels) != positions.shape[0]:
            raise ValueError("labels must match the microphone count")
        if not np.isfinite(positions).all():
            raise ValueError("positions_m must contain finite values")
        object.__setattr__(self, "positions_m", positions)

    @property
    def microphone_count(self) -> int:
        return self.positions_m.shape[0]

    @property
    def center_m(self) -> FloatArray:
        return self.positions_m.mean(axis=0)

    @property
    def centered_positions_m(self) -> FloatArray:
        return self.positions_m - self.center_m

    @property
    def aperture_m(self) -> float:
        delta = self.positions_m[:, None, :] - self.positions_m[None, :, :]
        return float(np.linalg.norm(delta, axis=-1).max())


def glasses_4mic() -> ArrayGeometry:
    """Four microphones distributed across two smart-glasses temples."""

    return ArrayGeometry(
        name="glasses-4mic",
        positions_m=np.array(
            [
                [-0.075, 0.040, 0.000],
                [-0.075, -0.040, 0.000],
                [0.075, 0.040, 0.000],
                [0.075, -0.040, 0.000],
            ]
        ),
        labels=("left-front", "left-rear", "right-front", "right-rear"),
    )


def asymmetric_earbuds_6mic() -> ArrayGeometry:
    """A deliberately asymmetric six-microphone binaural earbud layout."""

    return ArrayGeometry(
        name="asymmetric-earbuds-6mic",
        positions_m=np.array(
            [
                [-0.090, 0.012, 0.004],
                [-0.087, -0.008, -0.004],
                [-0.082, 0.002, 0.012],
                [0.090, 0.010, 0.003],
                [0.086, -0.011, -0.006],
                [0.081, 0.004, 0.009],
            ]
        ),
        labels=("L-outer", "L-inner", "L-top", "R-outer", "R-inner", "R-top"),
    )


def direction_vector(azimuth_deg: float, elevation_deg: float = 0.0) -> FloatArray:
    """Convert azimuth/elevation in degrees to a 3-D unit vector."""

    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def relative_arrival_delays(
    geometry: ArrayGeometry,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    sound_speed_m_s: float = 343.0,
) -> FloatArray:
    """Return non-negative far-field arrival delays relative to the first arrival."""

    if sound_speed_m_s <= 0:
        raise ValueError("sound_speed_m_s must be positive")
    direction = direction_vector(azimuth_deg, elevation_deg)
    delays = -(geometry.centered_positions_m @ direction) / sound_speed_m_s
    return delays - delays.min()
