# Changelog

## [Unreleased]

### Changed

- Use the end-user wheel installation path in the README and tutorial.
- Document Python-version, old-pip, and console-script troubleshooting.
- Build and install the project wheel in a clean CI environment before running the demo.

## [0.2.0] - 2026-07-25

### Added

- SRP-PHAT azimuth localization (`srp_phat`) with wearable-aware defaults:
  aperture-derived band limiting and frame-coherence weighting
- Microphone gain/timing mismatch simulation (`apply_microphone_mismatch`)
- Streaming block-based delay-and-sum (`StreamingDelayAndSum`,
  `stream_blocks`) with sample-exact integer latency, verified against the
  offline reference in tests
- `docs/TUTORIAL.md`: seven-step end-to-end walkthrough, every block verified
- Expanded `docs/ALGORITHMS.md`: derivations, parameter cheat sheet,
  complexity, and failure modes for every algorithm
- CLI demo now also reports the SRP-PHAT estimate
- Nine new regression tests (16 total)

## [0.1.0] - 2026-07-22

### Added

- Smart-glasses and asymmetric earbud microphone geometries
- Far-field simulation and deterministic synthetic sample scene
- Delay-and-sum and MVDR reference beamformers
- GCC-PHAT, azimuth scan, and binaural coherence baseline
- Metrics, visualizations, CLI demo, tests, CI, and algorithm documentation
