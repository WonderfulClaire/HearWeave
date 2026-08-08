# Tutorial: from zero to a wearable beamforming experiment

This walkthrough builds a complete smart-glasses experiment in seven short
steps, entirely from synthetic data — nothing to download, every number
reproducible. Each step is a copy-pasteable block; run them in one Python
session (or paste them into a notebook cell by cell).

Companion reading: [Algorithms and assumptions](ALGORITHMS.md) explains the
mathematics behind every function used here.

## 0. Install and smoke-test

```bash
git clone https://github.com/WonderfulClaire/HearWeave.git
cd HearWeave
python --version  # HearWeave requires Python 3.10+
python -m pip install --upgrade pip
python -m pip install .
hearweave-demo --output demo-output   # writes plots + wav + npz
```

If the demo prints an estimated azimuth near 35° and an SNR improvement of a
few dB, your environment works.

## 1. Pick a device geometry

```python
from hearweave import glasses_4mic, asymmetric_earbuds_6mic

geometry = glasses_4mic()
print(geometry.name)                  # glasses-4mic
print(geometry.microphone_count)      # 4
print(geometry.aperture_m)            # 0.17 — widest mic pair, in metres
print(geometry.labels)                # ('left-front', 'left-rear', ...)
print(geometry.positions_m)           # (4, 3) array, metres, x=right y=front
```

The aperture matters more than the microphone count: it sets both the lowest
frequency with usable spatial resolution (≈ c/4D ≈ 500 Hz here) and the
highest unambiguous frequency (≈ c/D ≈ 2 kHz). Keep those two numbers in
mind — they explain most "why is my beamformer not helping" surprises.

To try your own device, construct an `ArrayGeometry` directly:

```python
import numpy as np
from hearweave import ArrayGeometry

my_device = ArrayGeometry(
    name="my-headband-3mic",
    positions_m=np.array([[-0.08, 0.02, 0.0], [0.0, 0.09, 0.0], [0.08, 0.02, 0.0]]),
    labels=("left", "front", "right"),
)
```

## 2. Simulate a scene you fully control

```python
import numpy as np
from hearweave import simulate_plane_wave
from hearweave.simulation import speech_like_probe

sample_rate_hz = 16_000
target_azimuth_deg = 35.0            # front-left of the wearer

clean = speech_like_probe(sample_rate_hz, duration_s=1.5)
channels = simulate_plane_wave(
    clean, geometry, sample_rate_hz, target_azimuth_deg,
    snr_db=4.0,                       # a challenging but realistic input SNR
    rng=np.random.default_rng(7),     # fixed seed => identical rerun
)
print(channels.shape)                 # (4, 24000) — mics x samples
```

Because the scene is synthetic you keep the ground truth (`clean`, the true
azimuth) — which is exactly what lets you *measure* improvement later instead
of eyeballing spectrograms.

## 3. Localize: where is the talker?

```python
from hearweave import scan_azimuth_energy, srp_phat

est_scan, scores_scan, grid = scan_azimuth_energy(channels, geometry, sample_rate_hz)
est_srp,  scores_srp,  _    = srp_phat(channels, geometry, sample_rate_hz)

print(f"energy scan : {est_scan:6.1f}°")   # within a few degrees of 35
print(f"SRP-PHAT    : {est_srp:6.1f}°")    # usually the sharper of the two
```

Both return `(peak, normalized_scores, grid)` so you can overlay them:

```python
import matplotlib.pyplot as plt

plt.plot(grid, scores_scan, label="energy scan")
plt.plot(grid, scores_srp, label="SRP-PHAT")
plt.axvline(target_azimuth_deg, ls="--", color="k", label="truth")
plt.xlabel("azimuth / °"); plt.legend(); plt.show()
```

A broad single hump for the energy scan and a narrow SRP-PHAT peak on top of
it is the healthy pattern. Twin SRP peaks mirrored around ±90° indicate
front–back ambiguity — your geometry has too little front–back spread.

## 4. Enhance: point a beam at the estimate

```python
from hearweave import delay_and_sum, mvdr_beamform
from hearweave.metrics import snr_db

enhanced_das  = delay_and_sum(channels, geometry, sample_rate_hz, est_srp)
enhanced_mvdr = mvdr_beamform(channels, geometry, sample_rate_hz, est_srp)

margin = 64                                    # skip filter warm-up edges
ref = clean[margin:-margin]
print("input SNR :", snr_db(ref, channels[0, margin:-margin]))
print("DAS SNR   :", snr_db(ref, enhanced_das[margin:-margin]))
print("MVDR SNR  :", snr_db(ref, enhanced_mvdr[margin:-margin]))
```

Expected on this scene: input around −2.5 dB, DAS around +11 dB, MVDR close
behind. Two honest footnotes on that gap: `snr_db` counts the channel-0
propagation delay as "noise", so the input number is pessimistic and part of
the improvement is mere re-alignment; the true array gain against
independent noise is bounded by 10·log₁₀(4) = 6 dB. And MVDR with all-frame
covariance cannot beat DAS here — its advantage appears with *directional*
interferers, which white noise does not provide. Steer 20° off on purpose
and watch both numbers drop; that sensitivity is why localization quality
matters.

## 5. Stress-test with microphone mismatch

Real production microphones differ in gain and timing. Never report clean
numbers only:

```python
from hearweave import apply_microphone_mismatch

errors = []
for seed in range(10):
    degraded = apply_microphone_mismatch(
        channels, sample_rate_hz,
        gain_std_db=2.0, delay_jitter_std_s=10e-6,
        rng=np.random.default_rng(seed),
    )
    est, _, _ = srp_phat(degraded, geometry, sample_rate_hz)
    errors.append(abs((est - target_azimuth_deg + 180) % 360 - 180))

print(f"SRP-PHAT error over 10 mismatch draws: "
      f"median {np.median(errors):.1f}°, worst {max(errors):.1f}°")
```

This ten-line loop is the difference between "works in my notebook" and a
robustness claim you can defend.

## 6. Go block-based (the wearable reality)

Wearable DSP receives audio in 2–16 ms callbacks, not as whole files.
`StreamingDelayAndSum` is the stateful counterpart of `delay_and_sum`:

```python
from hearweave import StreamingDelayAndSum, stream_blocks

streamer = StreamingDelayAndSum(geometry, sample_rate_hz, look_azimuth_deg=est_srp)
blocks = [streamer.process_block(b) for b in stream_blocks(channels, block_size=256)]
streamed = np.concatenate(blocks)[: channels.shape[1]]

print("latency:", streamer.latency_samples, "samples",
      f"({1000 * streamer.latency_samples / sample_rate_hz:.2f} ms)")

# Identical to offline output, just shifted by the reported latency:
L = streamer.latency_samples
print("max deviation:", np.max(np.abs(streamed[L + 32 : -32]
                                      - enhanced_das[32 : -L - 32])))
```

Block-size invariance is guaranteed (same output for 128 or 512-sample
blocks), so choose the block size your audio stack dictates and prototype the
control logic — e.g. re-running SRP-PHAT once per second and calling
`streamer.reset()` when the look direction jumps.

## 7. Make the figures

```python
from hearweave.visualization import plot_geometry, plot_beam_pattern, save_localization_plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(11, 4), constrained_layout=True)
plot_geometry(geometry, fig.add_subplot(1, 2, 1))
plot_beam_pattern(geometry, est_srp, ax=fig.add_subplot(1, 2, 2, projection="polar"))
fig.savefig("my_experiment.png", dpi=180)

save_localization_plot(grid, scores_srp, est_srp, target_azimuth_deg, "my_scan.png")
```

Check the polar beam pattern at 500 Hz, 2 kHz, and 4 kHz before drawing
conclusions: at 500 Hz it is nearly a circle (no directivity), at 4 kHz
grating lobes appear. The clean-looking 2 kHz pattern in the README is the
*best* frequency for this aperture, not the typical one.

## Where to go next

- Swap `glasses_4mic()` for `asymmetric_earbuds_6mic()` — every step above
  works unchanged; watch how the left/right split changes the scan shape.
- Replace the probe with your own mono recording (`numpy` array at 16 kHz)
  — the simulation and metrics do not care where the samples come from.
- Feed a *directional* interferer by summing two `simulate_plane_wave`
  scenes from different azimuths, then compare DAS vs MVDR again — this is
  where MVDR earns its complexity.
- Read the [evaluation checklist](ALGORITHMS.md#11-evaluation-checklist)
  before publishing any number.
