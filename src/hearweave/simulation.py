"""Deterministic far-field simulation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import ArrayGeometry, relative_arrival_delays

FloatArray = NDArray[np.float64]


def fractional_delay(signal: ArrayLike, delay_s: float, sample_rate_hz: int) -> FloatArray:
    """Delay a one-dimensional signal with linear interpolation and zero padding."""

    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    time = np.arange(samples.size, dtype=float) / sample_rate_hz
    return np.interp(time - delay_s, time, samples, left=0.0, right=0.0)


def simulate_plane_wave(
    signal: ArrayLike,
    geometry: ArrayGeometry,
    sample_rate_hz: int,
    azimuth_deg: float,
    *,
    elevation_deg: float = 0.0,
    snr_db: float | None = 20.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Simulate a far-field source arriving at every microphone.

    The model is intentionally lightweight: free-field propagation, fractional
    delay, equal gain, and optional independent white noise per channel.
    """

    clean = np.asarray(signal, dtype=float)
    delays = relative_arrival_delays(geometry, azimuth_deg, elevation_deg)
    channels = np.vstack(
        [fractional_delay(clean, float(delay), sample_rate_hz) for delay in delays]
    )
    if snr_db is None:
        return channels
    generator = rng or np.random.default_rng(0)
    power = float(np.mean(clean**2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    return channels + generator.normal(scale=np.sqrt(noise_power), size=channels.shape)


def speech_like_probe(sample_rate_hz: int = 16_000, duration_s: float = 1.5) -> FloatArray:
    """Generate a deterministic, speech-like broadband probe without external data."""

    time = np.arange(int(sample_rate_hz * duration_s), dtype=float) / sample_rate_hz
    envelope = 0.45 + 0.55 * np.sin(2 * np.pi * 2.1 * time) ** 2
    sweep = np.sin(2 * np.pi * (180 * time + 0.5 * 520 * time**2))
    harmonics = 0.45 * np.sin(2 * np.pi * 420 * time) + 0.2 * np.sin(2 * np.pi * 910 * time)
    probe = envelope * (sweep + harmonics)
    return probe / (np.max(np.abs(probe)) + 1e-12)
