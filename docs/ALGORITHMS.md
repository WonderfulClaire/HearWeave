# Algorithms and assumptions

This document explains every algorithm shipped in HearWeave: the model behind
it, the exact mathematics implemented in the source, how to choose parameters
on a wearable device, what it costs, and where it breaks. Read it before using
any result in a paper or product comparison.

Notation used throughout:

| Symbol | Meaning |
| --- | --- |
| $M$ | number of microphones |
| $N$ | number of time samples |
| $f_s$ | sample rate in Hz |
| $c$ | speed of sound, 343 m/s at 20 °C |
| $\mathbf{p}_m \in \mathbb{R}^3$ | position of microphone $m$ in metres |
| $\theta, \varphi$ | azimuth and elevation of the source, in degrees |
| $\mathbf{u}(\theta,\varphi)$ | unit vector pointing *towards* the source |
| $x_m[n]$ | signal captured by microphone $m$ |
| $X_m(f, t)$ | STFT of $x_m$ at frequency bin $f$, frame $t$ |

## 1. Coordinate system and geometry

Coordinates are metres in a right-handed Cartesian frame centred on the head:
**+x is the wearer's right, +y is forward, +z is up**. Azimuth is measured in
the horizontal plane and elevation from that plane, so

$$
\mathbf{u}(\theta,\varphi) =
\begin{pmatrix}
\cos\varphi \cos\theta \\
\cos\varphi \sin\theta \\
\sin\varphi
\end{pmatrix}.
$$

`ArrayGeometry` validates shape, finiteness, and channel labels at
construction time so that unit errors fail loudly instead of skewing results.
`aperture_m` is the largest inter-microphone distance
$D = \max_{i,j} \lVert \mathbf{p}_i - \mathbf{p}_j \rVert$; it controls both
the low-frequency resolution limit and the high-frequency aliasing limit of
every spatial algorithm below.

### Far-field delay model

A plane wave from direction $\mathbf{u}$ reaches microphone $m$ earlier the
further the microphone lies *towards* the source, so the arrival time relative
to the array centre is

$$
\tau_m = -\frac{\mathbf{p}_m^{\mathsf{T}} \mathbf{u}}{c}.
$$

`relative_arrival_delays` subtracts the minimum so all returned delays are
non-negative (the first-hit microphone has delay 0). The far-field assumption
is valid when the source distance $r \gg D^2 f / c$ (Fraunhofer criterion);
for a 17 cm glasses frame at 2 kHz that is anything beyond roughly 0.2 m, so
conversation-distance sources are safely far-field. **Near-field talkers
closer than about 30 cm (the wearer's own mouth!) violate this model** — own
voice needs a spherical-wavefront extension, which is on the roadmap.

## 2. Simulation

### Fractional delay

`fractional_delay` shifts a signal by $\tau$ seconds using linear
interpolation with zero padding. Linear interpolation is a first-order
low-pass: it attenuates content near Nyquist by up to 6 dB when the
fractional part approaches 0.5 samples. That is acceptable for 16 kHz speech
prototyping; use a windowed-sinc interpolator if you need flat response above
6 kHz.

### Plane-wave scenes

`simulate_plane_wave` applies the far-field delays, equal gain per channel,
and optional independent white Gaussian noise at a specified SNR relative to
the clean probe power:

$$
x_m[n] = s\!\left(n/f_s - \tau_m\right) + v_m[n], \qquad
v_m \sim \mathcal{N}\!\left(0, \; \sigma^2\right), \quad
\sigma^2 = \frac{P_s}{10^{\mathrm{SNR}/10}}.
$$

Deliberately **not** modelled: room reverberation, head shadowing / HRTFs,
device scattering, source directivity, clock drift between left and right
devices. Independent white noise is the friendliest possible noise field —
diffuse babble is spatially correlated at low frequencies and will reduce
every gain figure measured here.

### Microphone mismatch

`apply_microphone_mismatch` degrades a simulated scene with per-channel gain
and timing errors:

$$
\tilde{x}_m[n] = g_m \, x_m\!\left[n - \delta_m f_s\right], \qquad
20\log_{10} g_m \sim \mathcal{N}(0, \sigma_g^2), \quad
\delta_m \sim \mathcal{N}(0, \sigma_\tau^2).
$$

Defaults ($\sigma_g = 1$ dB, $\sigma_\tau = 10\,\mu s$) match typical MEMS
production tolerance; $\pm 3$ dB and tens of microseconds are realistic for
uncalibrated units across two independently-clocked earbuds. Use this to
report robustness curves instead of clean-only numbers: run the same
algorithm over several mismatch draws (`rng=np.random.default_rng(k)`) and
show the error distribution.

## 3. Delay-and-sum beamforming

Align every channel to the look direction and average:

$$
y[n] = \frac{1}{M} \sum_{m=1}^{M} x_m\!\left[n + \tau_m f_s\right].
$$

For a target exactly in the look direction with independent noise of equal
power per channel, the signal adds coherently (amplitude $\times M$) and the
noise adds incoherently (power $\times M$), so the theoretical SNR gain — the
*white noise gain* — is

$$
G_{\mathrm{WNG}} = 10 \log_{10} M \;\; \text{dB},
$$

i.e. 6 dB for the 4-mic glasses preset. The shipped regression test asserts a
conservative +2 dB to stay robust across seeds. Two structural limits:

- **Low frequency**: wavelengths much longer than the aperture produce nearly
  identical phases on all microphones, so the beam degenerates to
  omnidirectional below roughly $f \approx c / (4D)$ (≈ 500 Hz for glasses).
- **High frequency**: once $f > c / D_{\min}$ for some microphone pair
  spacing, grating lobes appear — directions other than the look direction
  also sum coherently. Inspect `beam_pattern` at several frequencies before
  trusting a geometry.

Complexity is $O(MN)$ per look direction. The offline reference prioritizes
clarity; `StreamingDelayAndSum` (section 7) is the block-based counterpart.

## 4. MVDR beamforming

The minimum-variance distortionless-response beamformer minimizes output
power subject to unity gain in the look direction. With spatial covariance
$\mathbf{R}(f)$ and steering vector $\mathbf{d}(f)$
(where $d_m = e^{-j 2\pi f \tau_m}$):

$$
\mathbf{w}(f) = \frac{\mathbf{R}^{-1}(f)\, \mathbf{d}(f)}
                     {\mathbf{d}^{\mathsf{H}}(f)\, \mathbf{R}^{-1}(f)\, \mathbf{d}(f)},
\qquad
Y(f,t) = \mathbf{w}^{\mathsf{H}}(f) \, \mathbf{X}(f,t).
$$

Implementation notes, in the order they matter in practice:

1. **Covariance source.** The reference estimates $\mathbf{R}$ from *all*
   frames, which mixes target and noise. This makes the demo self-contained
   but under-uses MVDR: with target-contaminated covariance the beamformer
   trades target cancellation against noise reduction. For real evaluations
   estimate $\mathbf{R}$ from a noise-only segment, or use a speech-presence
   mask, and say so in your report.
2. **Diagonal loading.** $\mathbf{R} \leftarrow \mathbf{R} + \lambda
   \tfrac{\operatorname{tr}(\mathbf{R})}{M}\mathbf{I}$ with $\lambda$ =
   `diagonal_loading` (default $10^{-3}$). Loading bounds the white-noise gain
   and is *the* robustness lever against steering error and microphone
   mismatch: raise it to $10^{-2}$–$10^{-1}$ when the look direction is
   uncertain by a few degrees; lower it towards $10^{-4}$ only with calibrated
   arrays and accurate steering.
3. **STFT resolution.** `n_fft=512` at 16 kHz gives 32 ms frames — enough
   frequency resolution for the narrowband approximation to hold across a
   20 cm aperture. Halving it degrades low-frequency behaviour first.

Complexity per frequency bin is $O(M^3 + M T)$ ($T$ frames) for the solve and
apply; total $O(F (M^3 + M T))$. For wearable $M \le 8$ the $M^3$ term is
negligible next to the STFTs.

## 5. GCC-PHAT pairwise delay

The generalized cross-correlation with phase transform whitens the cross
spectrum before the inverse transform:

$$
R_{12}(\tau) = \int
\frac{X_1(f) X_2^{*}(f)}{\left| X_1(f) X_2^{*}(f) \right|}
\, e^{j 2 \pi f \tau} \, df,
\qquad
\hat{\tau} = \arg\max_\tau \left| R_{12}(\tau) \right|.
$$

Discarding magnitude makes the correlation peak sharp and robust to spectral
colouring and mild reverberation, at the cost of amplifying phase noise in
low-SNR bands. The implementation zero-pads to avoid circular wrap-around,
upsamples the correlation by `interpolation` (default 8×, i.e. delay
resolution $1/(8 f_s) \approx 8\,\mu s$), and optionally restricts the search
to $|\tau| \le$ `max_tau_s`. **Always pass** `max_tau_s = aperture / c` when
you know the pair spacing — it removes physically impossible peaks for free.

A single pair constrains the direction to a cone (front–back ambiguity in the
plane); combining pairs is exactly what SRP-PHAT does.

## 6. Azimuth localization: energy scan and SRP-PHAT

### Energy scan (`scan_azimuth_energy`)

Steer delay-and-sum over a grid and pick the azimuth with the largest output
energy. Simple, deterministic and a good sanity check, but the spatial
response of a small array is broad, so the peak is easily dragged by noise:
expect a few degrees of bias even in clean scenes. Complexity
$O(G \cdot M N)$ for $G$ grid points — the most expensive baseline here.

### SRP-PHAT (`srp_phat`)

Steered response power with PHAT weighting accumulates whitened cross-spectra
of all $M(M-1)/2$ pairs over a grid of candidate directions:

$$
P(\theta) = \sum_{i<j} \; \sum_{f \in \mathcal{B}} \; w_{ij}(f) \,
e^{\,j 2 \pi f\, (\tau_i(\theta) - \tau_j(\theta))},
\qquad
w_{ij}(f) = \overline{
\frac{X_i(f,t) X_j^{*}(f,t)}{\left| X_i(f,t) X_j^{*}(f,t) \right|}
}^{\,t}.
$$

Two implementation details are essential for wearable-sized apertures, and
both were tuned on the deterministic scenes in `tests/`:

- **Band limiting ($\mathcal{B}$).** Above $f_{\mathrm{alias}} \approx c / D$
  the widest pair's phase wraps and contributes coherent energy to *wrong*
  directions (spatial aliasing). The default band is
  $[100 \text{ Hz}, \; c / D]$ — about 2 kHz for the glasses preset. If you
  widen it, verify the scan plot for ghost peaks rather than trusting the
  argmax.
- **Coherence weighting.** After averaging over frames, the magnitude
  $|w_{ij}(f)| \in [0,1]$ is itself a phase-consistency measure: bins
  dominated by independent noise average towards 0, bins with a stable
  direction stay near 1. Multiplying each bin by $|w_{ij}(f)|^{\gamma}$
  ($\gamma$ = `coherence_power`, default 2) suppresses unreliable bins
  without any explicit noise estimate. Set $\gamma = 0$ to recover textbook
  SRP-PHAT.

Complexity $O(G \cdot M^2 F)$ on top of one STFT — much cheaper than the
energy scan for the same grid, because there is no per-angle time-domain
resynthesis.

### Choosing between them

| | Energy scan | SRP-PHAT |
| --- | --- | --- |
| Robust to spectral colouring | no | yes (PHAT) |
| Robust to gain mismatch | partly | yes (magnitude discarded) |
| Very low SNR (< 0 dB) | degrades gracefully | phase noise dominates |
| Cost per grid point | high ($MN$) | low ($M^2 F$) |
| Typical clean-scene error (glasses preset) | ≤ 4° | ≤ 2° |

Both report the grid argmax; interpolate the peak parabolic-style if you need
sub-grid resolution. Front–back confusion is geometry-dependent: the glasses
preset has enough y-spread to resolve it, a pure left–right earbud pair does
not.

## 7. Streaming delay-and-sum

`StreamingDelayAndSum` processes fixed-size blocks with a per-channel history
buffer. Causality forbids advancing channels, so every channel is *delayed*
by $L - \tau_m f_s$ samples, where the shared integer latency is

$$
L = \left\lceil \max_m \tau_m f_s \right\rceil + 1 .
$$

The output equals the offline reference delayed by exactly $L$ samples, which
the regression test verifies sample-by-sample ($< 10^{-6}$ absolute error)
and independently of block size. For wearable apertures $L$ stays under 10
samples at 16 kHz (< 0.7 ms) — negligible against typical hearing-aid latency
budgets of 5–10 ms. Memory is $O(ML)$; per-block cost is $O(M B)$ for block
size $B$. `reset()` clears state after a look-direction change; expect one
latency-length transient.

`stream_blocks` cuts an offline recording into constant-size blocks
(zero-padding the tail) so offline scenes can drive block-based code exactly
like a real-time audio callback would.

## 8. Binaural coherence mask

For left/right signals the magnitude-squared-coherence-like statistic

$$
\Gamma(f) = \frac{\left| \overline{X_L(f,t) X_R^{*}(f,t)} \right|}
{\sqrt{\overline{|X_L|^2} \; \overline{|X_R|^2}}}
$$

is near 1 for a compact frontal source and lower for diffuse noise. The
baseline maps $\Gamma$ through a floor/gain ramp and applies the *same* real
mask to both ears, preserving interaural time and level differences — the
cues the wearer's brain uses for spatial hearing. It is an educational
baseline: a single time-invariant mask per frequency, no perceptual or
clinical validation. Do not present its output as hearing-aid performance.

## 9. Metrics

- `snr_db` — signal-to-residual ratio $10\log_{10}(\lVert s \rVert^2 /
  \lVert \hat{s} - s \rVert^2)$. Sensitive to gain and delay: align and scale
  before use (the demo trims warm-up margins for exactly this reason).
- `si_sdr_db` — scale-invariant SDR: project the estimate onto the reference,
  then compare projection to residual. Insensitive to gain, still sensitive
  to misalignment. Report SI-SDR *improvement* over the noisy input, not the
  absolute number.

Neither metric is perceptual. For listening claims use PESQ/STOI/HASPI on
real recordings — outside this repository's scope by design.

## 10. Parameter cheat sheet

| Parameter | Default | Raise it when… | Lower it when… |
| --- | --- | --- | --- |
| `diagonal_loading` (MVDR) | 1e-3 | steering/geometry uncertain, mismatch present | array calibrated, direction known |
| `n_fft` (MVDR/SRP/binaural) | 512/1024 | need finer frequency detail, long stationary scenes | latency budget tight, short scenes |
| `coherence_power` (SRP) | 2.0 | heavy noise, want sharper peak | very short scenes (few frames to average) |
| `band_hz` upper edge (SRP) | $c/D$ | never above $c/D$ without checking scan plots | strong low-frequency interference |
| `interpolation` (GCC-PHAT) | 8 | need finer delay resolution | compute-bound |
| `gain_std_db` / `delay_jitter_std_s` | 1 dB / 10 µs | stress-testing robustness | modelling calibrated arrays |
| block size (streaming) | caller's | throughput matters | latency matters |

## 11. Evaluation checklist

- Declare array coordinates, units, and the coordinate convention.
- Separate synthetic, replayed, and real-device results — never average them.
- Report room, source distance, SNR, noise type, source count, and mismatch
  parameters alongside every number.
- Use multiple random seeds and report confidence intervals, not best runs.
- State covariance estimation policy (all-frames vs noise-only) for MVDR.
- State the frequency band and grid resolution for localization results.
- Do not interpret the included sample scene as benchmark evidence.
