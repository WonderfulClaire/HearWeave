# Algorithms and assumptions

HearWeave provides readable baselines for smart wearable array experiments.

## Geometry and simulation

Coordinates are expressed in metres in a right-handed Cartesian frame. Plane-wave simulation uses far-field relative delays, equal gain, linear-interpolation fractional delay, and optional independent white noise. It does not model head shadowing, pinna cues, room impulse responses, device scattering, clock drift, or microphone mismatch.

## Delay-and-sum

Channels are aligned with predicted far-field delays and averaged. The reference implementation prioritizes clarity over streaming efficiency.

## MVDR

The frequency-domain MVDR implementation estimates spatial covariance over all available frames and applies diagonal loading. Serious evaluations should estimate noise covariance from a defined noise segment and report STFT, loading, and steering assumptions.

## Direction scan

`scan_azimuth_energy` steers delay-and-sum across an explicit angle grid and returns normalized output energy. It is a compact localization baseline, not a substitute for a validated SRP-PHAT implementation.

## Binaural coherence mask

The shared mask retains frequency regions with stronger interaural coherence while preserving two output channels. It is an educational baseline and has not been perceptually or clinically validated.

## Evaluation checklist

- Declare array coordinates and units.
- Separate synthetic, replayed, and real-device results.
- Report room, distance, SNR, source count, and microphone mismatch.
- Use multiple random seeds and confidence intervals.
- Do not interpret the included sample scene as benchmark evidence.
