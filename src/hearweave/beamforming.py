"""Reference beamformers with explicit array geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import istft, stft

from .geometry import ArrayGeometry, relative_arrival_delays
from .simulation import fractional_delay

FloatArray = NDArray[np.float64]


def _validate_signals(signals: ArrayLike, geometry: ArrayGeometry) -> FloatArray:
    array = np.asarray(signals, dtype=float)
    if array.ndim != 2:
        raise ValueError("signals must have shape (microphones, samples)")
    if array.shape[0] != geometry.microphone_count:
        raise ValueError("signal channel count does not match geometry")
    return array


def delay_and_sum(
    signals: ArrayLike,
    geometry: ArrayGeometry,
    sample_rate_hz: int,
    look_azimuth_deg: float,
    *,
    look_elevation_deg: float = 0.0,
) -> FloatArray:
    """Align a far-field look direction and average the microphone channels."""

    array = _validate_signals(signals, geometry)
    delays = relative_arrival_delays(geometry, look_azimuth_deg, look_elevation_deg)
    aligned = np.vstack(
        [
            fractional_delay(channel, -float(delay), sample_rate_hz)
            for channel, delay in zip(array, delays, strict=True)
        ]
    )
    return aligned.mean(axis=0)


def mvdr_beamform(
    signals: ArrayLike,
    geometry: ArrayGeometry,
    sample_rate_hz: int,
    look_azimuth_deg: float,
    *,
    n_fft: int = 512,
    diagonal_loading: float = 1e-3,
) -> FloatArray:
    """Frequency-domain MVDR reference implementation.

    Covariance is estimated over all frames. For evaluation work, callers should
    provide a dedicated noise estimate or segment the scene explicitly.
    """

    array = _validate_signals(signals, geometry)
    if diagonal_loading <= 0:
        raise ValueError("diagonal_loading must be positive")
    frequencies, _, spectra = stft(
        array,
        fs=sample_rate_hz,
        nperseg=n_fft,
        noverlap=n_fft // 2,
        axis=-1,
        boundary="zeros",
    )
    delays = relative_arrival_delays(geometry, look_azimuth_deg)
    steering = np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :])
    output = np.zeros((frequencies.size, spectra.shape[-1]), dtype=np.complex128)
    microphone_count = geometry.microphone_count

    for frequency_index in range(frequencies.size):
        snapshot = spectra[:, frequency_index, :]
        covariance = snapshot @ snapshot.conj().T / max(snapshot.shape[1], 1)
        scale = max(float(np.trace(covariance).real / microphone_count), 1e-12)
        covariance += diagonal_loading * scale * np.eye(microphone_count)
        steering_vector = steering[frequency_index]
        inverse_steering = np.linalg.solve(covariance, steering_vector)
        denominator = steering_vector.conj() @ inverse_steering
        weights = inverse_steering / (denominator + 1e-12)
        output[frequency_index] = weights.conj() @ snapshot

    _, enhanced = istft(
        output,
        fs=sample_rate_hz,
        nperseg=n_fft,
        noverlap=n_fft // 2,
        input_onesided=True,
    )
    result = np.asarray(enhanced[: array.shape[1]], dtype=float)
    if result.size < array.shape[1]:
        result = np.pad(result, (0, array.shape[1] - result.size))
    return result
