"""Generate a reproducible static overview and lightweight animated walkthrough."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hearweave import delay_and_sum, glasses_4mic, srp_phat
from hearweave.visualization import beam_pattern

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "blue": "#0F4D92",
    "green": "#2E8B72",
    "red": "#B64342",
    "gray": "#767676",
    "light": "#E8EEF5",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.4,
            "legend.frameon": False,
        }
    )


def _load_scene() -> tuple[np.ndarray, int, float]:
    scene = np.load(ROOT / "datasets" / "simulated_glasses_scene.npz")
    return (
        np.asarray(scene["microphone_signals"], dtype=float),
        int(scene["sample_rate_hz"]),
        float(scene["target_azimuth_deg"]),
    )


def _build_figure() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    figure = plt.figure(figsize=(12, 6.75), facecolor="white")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.25, 1.15), hspace=0.42, wspace=0.35)
    axes = {
        "geometry": figure.add_subplot(grid[:, 0]),
        "waveform": figure.add_subplot(grid[0, 1:]),
        "scan": figure.add_subplot(grid[1, 1]),
        "beam": figure.add_subplot(grid[1, 2], projection="polar"),
    }
    figure.suptitle(
        "HearWeave · from wearable microphones to an inspectable beam",
        x=0.04,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#272727",
    )
    figure.text(
        0.04,
        0.925,
        "Deterministic synthetic scene · 4-mic smart-glasses geometry · no recorded speech",
        color=PALETTE["gray"],
    )
    return figure, axes


def _draw_fixed_panels(
    axes: dict[str, plt.Axes], signals: np.ndarray, sample_rate_hz: int
) -> None:
    geometry = glasses_4mic()
    positions = geometry.positions_m * 100
    axis = axes["geometry"]
    axis.add_patch(plt.Rectangle((-8.7, -5.5), 17.4, 11.0, color=PALETTE["light"], zorder=0))
    axis.plot([-8.5, 8.5], [0, 0], color=PALETTE["gray"], lw=5, solid_capstyle="round")
    axis.scatter(positions[:, 0], positions[:, 1], s=95, color=PALETTE["blue"], zorder=3)
    labeled_positions = zip(geometry.labels, positions, strict=True)
    for index, (label, position) in enumerate(labeled_positions, start=1):
        axis.annotate(
            f"M{index} · {label}",
            position[:2],
            xytext=(7, 7 if position[1] > 0 else -14),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(xlim=(-11, 11), ylim=(-8, 8), xlabel="x / cm", ylabel="y / cm")
    axis.set_aspect("equal")
    axis.set_title("1 · Device geometry", loc="left", fontweight="bold")
    axis.grid(alpha=0.15)

    axis = axes["waveform"]
    samples = min(signals.shape[1], int(0.045 * sample_rate_hz))
    time_ms = 1_000 * np.arange(samples) / sample_rate_hz
    offsets = np.arange(signals.shape[0])[::-1] * 2.0
    scale = max(float(np.max(np.abs(signals[:, :samples]))), 1e-12)
    for index, (channel, offset) in enumerate(zip(signals[:, :samples], offsets, strict=True)):
        axis.plot(time_ms, channel / scale + offset, color=PALETTE["blue"], lw=1.1)
        axis.text(time_ms[-1] + 0.5, offset, f"M{index + 1}", va="center", fontsize=9)
    axis.set(
        xlim=(time_ms[0], time_ms[-1] + 3),
        xlabel="time / ms",
        yticks=[],
        title="2 · The same probe arrives at four microphones with different delays",
    )
    axis.title.set_fontweight("bold")
    axis.title.set_ha("left")
    axis.title.set_position((0, 1.0))
    axis.grid(axis="x", alpha=0.15)


def generate(output: Path, *, frames: int = 48, fps: int = 12) -> dict[str, float]:
    """Write the static PNG and animated GIF to *output*."""
    _style()
    output.mkdir(parents=True, exist_ok=True)
    signals, sample_rate_hz, target_deg = _load_scene()
    geometry = glasses_4mic()
    estimate_deg, scores, scan_grid = srp_phat(signals, geometry, sample_rate_hz)
    _ = delay_and_sum(signals, geometry, sample_rate_hz, estimate_deg)
    figure, axes = _build_figure()
    _draw_fixed_panels(axes, signals, sample_rate_hz)

    scan_axis = axes["scan"]
    scan_axis.plot(scan_grid, scores, color=PALETTE["blue"], lw=2.4)
    scan_axis.axvline(target_deg, color=PALETTE["green"], ls="--", lw=1.8, label="synthetic truth")
    candidate_line = scan_axis.axvline(-180, color=PALETTE["red"], lw=1.8, label="candidate")
    estimate_marker, = scan_axis.plot([], [], "o", color=PALETTE["red"], ms=7)
    scan_axis.set(
        xlim=(-180, 180),
        ylim=(0, 1.05),
        xlabel="azimuth / degree",
        ylabel="normalized SRP-PHAT score",
        title="3 · Find the direction",
    )
    scan_axis.title.set_fontweight("bold")
    scan_axis.title.set_ha("left")
    scan_axis.title.set_position((0, 1.0))
    scan_axis.grid(alpha=0.15)
    scan_axis.legend(loc="upper left", fontsize=8)

    beam_axis = axes["beam"]
    beam_line, = beam_axis.plot([], [], color=PALETTE["blue"], lw=2.4)
    look_marker, = beam_axis.plot([], [], "o", color=PALETTE["red"], ms=7)
    beam_axis.set_rmax(1.0)
    beam_axis.set_rticks([0.5, 1.0])
    beam_axis.set_title("4 · Inspect the 2 kHz DAS beam", fontweight="bold", pad=18)
    status = figure.text(0.68, 0.025, "", ha="center", color=PALETTE["gray"], fontsize=9)

    candidates = np.linspace(-180, estimate_deg, frames)

    def update(frame_index: int) -> tuple[object, ...]:
        candidate = float(candidates[frame_index])
        pattern_grid, response = beam_pattern(geometry, candidate)
        candidate_line.set_xdata([candidate, candidate])
        score = float(np.interp(candidate, scan_grid, scores))
        estimate_marker.set_data([candidate], [score])
        beam_line.set_data(np.deg2rad(pattern_grid), response)
        look_marker.set_data([np.deg2rad(candidate)], [1.0])
        if frame_index == frames - 1:
            status.set_text(
                f"Lock: estimate {estimate_deg:.0f}° · synthetic truth {target_deg:.0f}°"
            )
        else:
            status.set_text(
                f"Azimuth candidate: {candidate:+.0f}° · "
                "visual explanation, not a runtime trace"
            )
        return candidate_line, estimate_marker, beam_line, look_marker, status

    update(frames - 1)
    figure.savefig(output / "hearweave_showcase.png", dpi=220, bbox_inches="tight")
    animation = FuncAnimation(figure, update, frames=frames, interval=1_000 / fps, blit=False)
    animation.save(output / "hearweave_showcase.gif", writer=PillowWriter(fps=fps), dpi=110)
    plt.close(figure)
    return {"target_azimuth_deg": target_deg, "estimated_azimuth_deg": estimate_deg}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", type=Path, default=ROOT / "docs" / "assets")
    command.add_argument("--frames", type=int, default=48)
    command.add_argument("--fps", type=int, default=12)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.frames < 2 or args.fps <= 0:
        raise SystemExit("--frames must be >= 2 and --fps must be positive")
    print(generate(args.output, frames=args.frames, fps=args.fps))


if __name__ == "__main__":
    main()
