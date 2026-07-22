"""Time-difference and scan-based direction-of-arrival helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .beamforming import delay_and_sum
from .geometry import ArrayGeometry

FloatArray = NDArray[np.float64]


def gcc_phat(
    signal: ArrayLike,
    reference: ArrayLike,
    sample_rate_hz: int,
    *,
    max_tau_s: float | None = None,
    interpolation: int = 8,
) -> tuple[float, FloatArray, FloatArray]:
    """Estimate pairwise time delay with generalized cross-correlation PHAT."""

    first = np.asarray(signal, dtype=float)
    second = np.asarray(reference, dtype=float)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("signal and reference must be one-dimensional")
    size = first.size + second.size
    cross_spectrum = np.fft.rfft(first, n=size) * np.conj(np.fft.rfft(second, n=size))
    cross_spectrum /= np.maximum(np.abs(cross_spectrum), 1e-12)
    correlation = np.fft.irfft(cross_spectrum, n=interpolation * size)
    max_shift = interpolation * size // 2
    if max_tau_s is not None:
        max_shift = min(max_shift, int(interpolation * sample_rate_hz * max_tau_s))
    correlation = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
    shifts = np.arange(-max_shift, max_shift + 1)
    shift = int(shifts[np.argmax(np.abs(correlation))])
    tau_s = shift / (interpolation * sample_rate_hz)
    lags_s = shifts / (interpolation * sample_rate_hz)
    return float(tau_s), correlation, lags_s


def scan_azimuth_energy(
    signals: ArrayLike,
    geometry: ArrayGeometry,
    sample_rate_hz: int,
    azimuth_grid_deg: ArrayLike | None = None,
) -> tuple[float, FloatArray, FloatArray]:
    """Scan candidate azimuths using delay-and-sum output energy."""

    grid = np.asarray(
        np.arange(-180.0, 181.0, 2.0) if azimuth_grid_deg is None else azimuth_grid_deg,
        dtype=float,
    )
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("azimuth_grid_deg must be a non-empty vector")
    scores = np.array(
        [
            np.mean(delay_and_sum(signals, geometry, sample_rate_hz, float(angle)) ** 2)
            for angle in grid
        ]
    )
    peak = float(grid[int(np.argmax(scores))])
    normalized = scores / (scores.max() + 1e-12)
    return peak, normalized, grid
