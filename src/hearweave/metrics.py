"""Small, dependency-light signal quality metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def snr_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return signal-to-residual ratio in decibels."""

    target = np.asarray(reference, dtype=float)
    output = np.asarray(estimate, dtype=float)
    if target.shape != output.shape:
        raise ValueError("reference and estimate must have equal shapes")
    residual = output - target
    return float(10.0 * np.log10((np.sum(target**2) + 1e-12) / (np.sum(residual**2) + 1e-12)))


def si_sdr_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return scale-invariant signal-to-distortion ratio in decibels."""

    target = np.asarray(reference, dtype=float)
    output = np.asarray(estimate, dtype=float)
    if target.shape != output.shape:
        raise ValueError("reference and estimate must have equal shapes")
    target = target - np.mean(target)
    output = output - np.mean(output)
    projection = np.dot(output, target) * target / (np.dot(target, target) + 1e-12)
    noise = output - projection
    return float(10.0 * np.log10((np.sum(projection**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))
