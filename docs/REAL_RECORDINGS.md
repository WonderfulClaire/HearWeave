# Real-recording adapter and evaluation protocol

HearWeave accepts real multichannel recordings through an explicit JSON manifest plus one mono WAV file per microphone. The adapter solves a narrow but important problem: it makes channel order, geometry, sample rate, and evidence metadata reviewable instead of guessing them from filenames.

It does **not** turn an arbitrary recording into a benchmark. Consent, licensing, ground truth, synchronization, calibration, scene control, and train/test separation remain the evaluator's responsibility.

## Manifest contract

Start from [`datasets/recording_manifest.example.json`](../datasets/recording_manifest.example.json). Required fields are:

- `schema_version`: currently `1`;
- `recording_id`: a stable non-personal identifier;
- `sample_rate_hz`: positive integer shared by every channel;
- `geometry_name`: name for the measured layout;
- `channels`: ordered channel objects containing a unique `label`, relative mono-WAV `file`, and `[x, y, z]` `position_m` in metres.

Any additional top-level fields are preserved in `recording.metadata`. Use them for non-identifying device revision, scene, clocking, calibration, ground-truth method, dataset version, or evidence scope. Do not store names, contact details, API keys, consent forms, or private filesystem paths in a public manifest.

## Load and inspect

```python
from hearweave import load_recording

recording = load_recording("my-capture/recording.json")
print(recording.recording_id)
print(recording.signals.shape)       # microphones × samples
print(recording.sample_rate_hz)
print(recording.geometry.labels)     # exactly the manifest order
print(recording.geometry.positions_m)
print(recording.metadata)
```

The loader rejects missing files, absolute or escaping paths, stereo files, unsupported sample types, sample-rate disagreement, unequal lengths, duplicate labels, and invalid positions. Integer PCM is converted to floating point near `[-1, 1]`; floating-point WAV values are preserved.

## Minimum defensible evaluation record

Before reporting a real-device number, record at least:

1. device revision, measured microphone coordinates, channel labels, and clock topology;
2. capture date, room/scene identifier, source and interferer placement, and ground-truth method;
3. sample rate, synchronization and calibration procedure, clipping/dropout checks;
4. participant consent or dataset license in a private, access-controlled record;
5. immutable dataset version and split policy, with no test-scene tuning;
6. metric definition, baseline, repetitions, exclusions, software commit, and environment;
7. whether audio and metadata may be redistributed; private recordings stay outside Git.

For a first hardware smoke test, inspect waveform length and clipping, run localization and DAS with the manifest geometry, and save results in BeamBench's tidy CSV contract. Call the result a device smoke test until multiple controlled scenes and repetitions exist.

## Current boundary

This adapter assumes already synchronized, equal-length mono channels. It does not estimate clock drift, resample channels, align independent devices, read interleaved multichannel containers, anonymize speech, or verify consent. Those operations must happen in a traceable preprocessing stage before loading.
