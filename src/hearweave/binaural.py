"""Small binaural cooperation primitives for reference experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import istft, stft

FloatArray = NDArray[np.float64]


def binaural_coherence_enhance(
    left: ArrayLike,
    right: ArrayLike,
    sample_rate_hz: int,
    *,
    n_fft: int = 512,
    coherence_floor: float = 0.25,
    minimum_gain: float = 0.15,
) -> FloatArray:
    """Apply a shared coherence mask while preserving two output channels.

    This is a compact educational baseline, not a perceptually validated
    hearing-aid algorithm.
    """

    left_signal = np.asarray(left, dtype=float)
    right_signal = np.asarray(right, dtype=float)
    if left_signal.shape != right_signal.shape or left_signal.ndim != 1:
        raise ValueError("left and right must be one-dimensional and equally sized")
    _, _, left_spectrum = stft(
        left_signal, fs=sample_rate_hz, nperseg=n_fft, noverlap=n_fft // 2
    )
    _, _, right_spectrum = stft(
        right_signal, fs=sample_rate_hz, nperseg=n_fft, noverlap=n_fft // 2
    )
    cross = np.mean(left_spectrum * np.conj(right_spectrum), axis=1)
    left_power = np.mean(np.abs(left_spectrum) ** 2, axis=1)
    right_power = np.mean(np.abs(right_spectrum) ** 2, axis=1)
    coherence = np.abs(cross) / np.sqrt(left_power * right_power + 1e-12)
    mask = np.clip(
        (coherence - coherence_floor) / max(1.0 - coherence_floor, 1e-12),
        minimum_gain,
        1.0,
    )[:, None]
    outputs = []
    for spectrum in (left_spectrum, right_spectrum):
        _, enhanced = istft(
            spectrum * mask,
            fs=sample_rate_hz,
            nperseg=n_fft,
            noverlap=n_fft // 2,
        )
        enhanced = np.asarray(enhanced[: left_signal.size], dtype=float)
        outputs.append(np.pad(enhanced, (0, left_signal.size - enhanced.size)))
    return np.vstack(outputs)
