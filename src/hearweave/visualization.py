"""Publication-friendly visualizations for wearable-array experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .geometry import ArrayGeometry, direction_vector, relative_arrival_delays


def plot_geometry(geometry: ArrayGeometry, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a top-down microphone layout."""

    axes = ax or plt.subplots(figsize=(6, 4))[1]
    positions = geometry.positions_m
    axes.scatter(positions[:, 0] * 100, positions[:, 1] * 100, s=90, color="#00d4ff")
    for label, position in zip(geometry.labels, positions, strict=True):
        axes.annotate(
            label,
            (position[0] * 100, position[1] * 100),
            xytext=(5, 6),
            textcoords="offset points",
        )
    axes.axhline(0, color="#aab2c0", lw=0.7)
    axes.axvline(0, color="#aab2c0", lw=0.7)
    axes.set_aspect("equal")
    axes.set_xlabel("x / cm")
    axes.set_ylabel("y / cm")
    axes.set_title(f"{geometry.name} · aperture {geometry.aperture_m * 100:.1f} cm")
    axes.grid(alpha=0.2)
    return axes


def beam_pattern(
    geometry: ArrayGeometry,
    look_azimuth_deg: float,
    frequency_hz: float = 2_000.0,
    azimuth_grid_deg: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized narrowband delay-and-sum beampattern."""

    grid = np.asarray(
        np.linspace(-180.0, 180.0, 721) if azimuth_grid_deg is None else azimuth_grid_deg,
        dtype=float,
    )
    look_delays = relative_arrival_delays(geometry, look_azimuth_deg)
    response = []
    for angle in grid:
        arrival = relative_arrival_delays(geometry, float(angle))
        phase = np.exp(-2j * np.pi * frequency_hz * (arrival - look_delays))
        response.append(abs(np.mean(phase)))
    values = np.asarray(response)
    return grid, values / (values.max() + 1e-12)


def plot_beam_pattern(
    geometry: ArrayGeometry,
    look_azimuth_deg: float,
    frequency_hz: float = 2_000.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the normalized narrowband beampattern on a polar axis."""

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    grid, response = beam_pattern(geometry, look_azimuth_deg, frequency_hz)
    ax.plot(np.deg2rad(grid), response, color="#6f5cff", lw=2)
    look = direction_vector(look_azimuth_deg)
    ax.plot([np.arctan2(look[1], look[0])], [1.0], "o", color="#ffb000")
    ax.set_title(f"DAS pattern · {frequency_hz / 1000:.1f} kHz")
    ax.set_rmax(1.0)
    return ax


def save_localization_plot(
    grid_deg: ArrayLike,
    scores: ArrayLike,
    estimate_deg: float,
    target_deg: float,
    output: str | Path,
) -> None:
    """Save a direction scan plot."""

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(grid_deg, scores, color="#6f5cff", lw=2)
    ax.axvline(target_deg, color="#00a87a", ls="--", label=f"target {target_deg:.0f}°")
    ax.axvline(estimate_deg, color="#ff8b00", ls=":", label=f"estimate {estimate_deg:.0f}°")
    ax.set(
        xlabel="azimuth / degree",
        ylabel="normalized scan energy",
        xlim=(-180, 180),
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
