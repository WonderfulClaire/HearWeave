# HearWeave Agent Guide

## Product
- Python toolkit for smart wearable microphone-array research.
- Prefer explicit geometry, units, assumptions, and deterministic examples.
- Never present synthetic results as real-device evidence.

## Verify
- Install: `python -m pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `python -m unittest discover -s tests -v`
- Demo: `hearweave-demo --output build/demo`

## Architecture
- `geometry.py`: device layouts and propagation delays.
- `simulation.py`: deterministic far-field scenes.
- `beamforming.py`: enhancement baselines.
- `localization.py`: TDOA and direction scan.
- `binaural.py`: two-ear cooperation baseline.
- `visualization.py`: reproducible plots.

## Rules
- New algorithms need a synthetic regression test and documented limitations.
- Keep coordinate systems and units in public APIs.
- Do not commit private audio or unlicensed datasets.
- Avoid accuracy, clinical, or product-readiness claims without a reproducible protocol.
