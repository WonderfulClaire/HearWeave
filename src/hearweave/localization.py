"""Time-difference and scan-based direction-of-arrival helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import stft

from .beamforming import delay_and_sum
from .geometry import ArrayGeometry, relative_arrival_delays

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


def srp_phat(
    signals: ArrayLike,
    geometry: ArrayGeometry,
    sample_rate_hz: int,
    azimuth_grid_deg: ArrayLike | None = None,
    *,
    n_fft: int = 1024,
    band_hz: tuple[float, float] | None = None,
    coherence_power: float = 2.0,
) -> tuple[float, FloatArray, FloatArray]:
    """Steered-response-power localization with PHAT weighting.

    The scene is split into STFT frames; every microphone pair's cross-spectrum
    is PHAT-whitened per time-frequency bin (magnitude discarded, phase kept)
    and averaged over frames. For every candidate azimuth the whitened
    cross-spectra are phase-aligned with the far-field delay model and
    accumulated. Two guards keep small wearable apertures honest:

    - ``band_hz`` restricts the scan to frequencies whose phase is spatially
      unambiguous. The default upper edge is ``sound_speed / aperture``
      (about 2 kHz for a 17 cm glasses frame) because higher frequencies
      alias across the widest microphone pair and drag the peak sideways.
    - ``coherence_power`` re-weights each bin by the magnitude of its
      frame-averaged whitened cross-spectrum raised to this power. Bins
      dominated by independent noise average towards zero and are therefore
      suppressed without any explicit noise estimate; ``0.0`` disables this.

    Returns ``(peak_azimuth_deg, normalized_scores, grid_deg)`` in the same
    format as :func:`scan_azimuth_energy` so the two baselines are drop-in
    interchangeable.
    """

    array = np.asarray(signals, dtype=float)
    if array.ndim != 2:
        raise ValueError("signals must have shape (microphones, samples)")
    if array.shape[0] != geometry.microphone_count:
        raise ValueError("signal channel count does not match geometry")
    if coherence_power < 0:
        raise ValueError("coherence_power must be non-negative")
    grid = np.asarray(
        np.arange(-180.0, 181.0, 2.0) if azimuth_grid_deg is None else azimuth_grid_deg,
        dtype=float,
    )
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("azimuth_grid_deg must be a non-empty vector")
    if band_hz is None:
        band_hz = (100.0, 343.0 / geometry.aperture_m)
    low_hz, high_hz = float(band_hz[0]), float(band_hz[1])
    if not 0 <= low_hz < high_hz:
        raise ValueError("band_hz must satisfy 0 <= low < high")

    frequencies, _, spectra = stft(
        array, fs=sample_rate_hz, nperseg=n_fft, noverlap=n_fft // 2, axis=-1
    )
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not band.any():
        raise ValueError("band_hz selects no STFT bins; widen the band or raise n_fft")
    frequencies = frequencies[band]
    pairs = [
        (first, second)
        for first in range(geometry.microphone_count)
        for second in range(first + 1, geometry.microphone_count)
    ]
    whitened = {}
    for first, second in pairs:
        cross = spectra[first, band] * np.conj(spectra[second, band])
        cross /= np.maximum(np.abs(cross), 1e-12)
        averaged = cross.mean(axis=-1)
        whitened[(first, second)] = averaged * np.abs(averaged) ** coherence_power

    scores = np.zeros(grid.size)
    for index, angle in enumerate(grid):
        delays = relative_arrival_delays(geometry, float(angle))
        power = 0.0
        for first, second in pairs:
            pair_delay = delays[first] - delays[second]
            steering = np.exp(2j * np.pi * frequencies * pair_delay)
            power += float(np.real(np.sum(whitened[(first, second)] * steering)))
        scores[index] = power

    peak = float(grid[int(np.argmax(scores))])
    scores -= scores.min()
    normalized = scores / (scores.max() + 1e-12)
    return peak, normalized, grid
