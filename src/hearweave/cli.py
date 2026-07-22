"""Command-line demonstration for HearWeave."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from .beamforming import delay_and_sum
from .geometry import glasses_4mic
from .localization import scan_azimuth_energy
from .metrics import snr_db
from .simulation import simulate_plane_wave, speech_like_probe
from .visualization import plot_beam_pattern, plot_geometry, save_localization_plot


def run_demo(output_directory: Path, seed: int = 7) -> dict[str, float]:
    output_directory.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 16_000
    target_azimuth_deg = 35.0
    geometry = glasses_4mic()
    clean = speech_like_probe(sample_rate_hz)
    channels = simulate_plane_wave(
        clean,
        geometry,
        sample_rate_hz,
        target_azimuth_deg,
        snr_db=4.0,
        rng=np.random.default_rng(seed),
    )
    estimate, scores, grid = scan_azimuth_energy(channels, geometry, sample_rate_hz)
    enhanced = delay_and_sum(channels, geometry, sample_rate_hz, target_azimuth_deg)
    margin = 64
    input_snr = snr_db(clean[margin:-margin], channels[0, margin:-margin])
    output_snr = snr_db(clean[margin:-margin], enhanced[margin:-margin])

    figure = plt.figure(figsize=(11, 4.2), constrained_layout=True)
    plot_geometry(geometry, figure.add_subplot(1, 2, 1))
    polar_axes = figure.add_subplot(1, 2, 2, projection="polar")
    plot_beam_pattern(geometry, target_azimuth_deg, ax=polar_axes)
    figure.savefig(output_directory / "array_and_beam_pattern.png", dpi=180)
    plt.close(figure)
    save_localization_plot(
        grid,
        scores,
        estimate,
        target_azimuth_deg,
        output_directory / "localization_scan.png",
    )

    peak = max(np.max(np.abs(enhanced)), 1e-12)
    pcm = np.int16(np.clip(enhanced / peak, -1, 1) * 32767)
    wavfile.write(output_directory / "enhanced.wav", sample_rate_hz, pcm)
    np.savez_compressed(
        output_directory / "simulated_glasses_scene.npz",
        sample_rate_hz=sample_rate_hz,
        target_azimuth_deg=target_azimuth_deg,
        microphone_positions_m=geometry.positions_m,
        clean_reference=clean,
        microphone_signals=channels,
    )
    return {
        "target_azimuth_deg": target_azimuth_deg,
        "estimated_azimuth_deg": estimate,
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HearWeave smart-glasses demo")
    parser.add_argument("--output", type=Path, default=Path("hearweave-demo"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metrics = run_demo(args.output, args.seed)
    print("HearWeave demo complete")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}")
    print(f"  artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
