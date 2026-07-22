"""Regenerate the deterministic README assets and sample dataset."""

from pathlib import Path
from shutil import copy2

from hearweave.cli import run_demo


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    generated = root / "build" / "generated-demo"
    metrics = run_demo(generated)
    assets = root / "docs" / "assets"
    dataset = root / "datasets"
    assets.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(parents=True, exist_ok=True)
    copy2(generated / "array_and_beam_pattern.png", assets / "array_and_beam_pattern.png")
    copy2(generated / "localization_scan.png", assets / "localization_scan.png")
    copy2(generated / "simulated_glasses_scene.npz", dataset / "simulated_glasses_scene.npz")
    print(metrics)


if __name__ == "__main__":
    main()
