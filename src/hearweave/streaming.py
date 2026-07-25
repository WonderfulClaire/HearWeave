"""Block-based streaming interfaces for wearable-friendly processing.

Wearable DSP runs on short fixed-size blocks (2 ms to 16 ms), not on whole
recordings. This module provides a stateful streaming counterpart to the
offline reference beamformer so that latency and block-boundary behaviour can
be prototyped honestly: the streaming output should match the offline output
except for the modelled algorithmic delay.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import ArrayGeometry, relative_arrival_delays

FloatArray = NDArray[np.float64]


class StreamingDelayAndSum:
    """Stateful block-wise delay-and-sum beamformer.

    The far-field alignment delays for the look direction are converted to a
    per-channel fractional-sample delay, implemented with a small per-channel
    history buffer so that block boundaries introduce no discontinuities. The
    total algorithmic latency is exactly :attr:`latency_samples` samples
    (an integer, so ``streaming[latency:]`` can be compared sample-by-sample
    with the offline reference), which for wearable apertures under 20 cm
    stays below 10 samples at 16 kHz.

    Example::

        streamer = StreamingDelayAndSum(geometry, 16_000, look_azimuth_deg=35.0)
        for block in stream_blocks(signals, block_size=256):
            enhanced_block = streamer.process_block(block)
    """

    def __init__(
        self,
        geometry: ArrayGeometry,
        sample_rate_hz: int,
        look_azimuth_deg: float,
        *,
        look_elevation_deg: float = 0.0,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.geometry = geometry
        self.sample_rate_hz = sample_rate_hz
        delays_s = relative_arrival_delays(geometry, look_azimuth_deg, look_elevation_deg)
        # Advancing every channel is impossible causally; instead every
        # channel is delayed by (latency - own_delay) so the summed output is
        # the offline-aligned signal at a constant integer latency.
        delay_samples = delays_s * sample_rate_hz
        self.latency_samples = int(np.ceil(float(delay_samples.max()))) + 1
        self._channel_delays = self.latency_samples - delay_samples
        self._history = np.zeros((geometry.microphone_count, self.latency_samples))

    def process_block(self, block: ArrayLike) -> FloatArray:
        """Process one block of shape ``(microphones, block_samples)``."""

        samples = np.asarray(block, dtype=float)
        if samples.ndim != 2 or samples.shape[0] != self.geometry.microphone_count:
            raise ValueError("block must have shape (microphones, block_samples)")
        block_size = samples.shape[1]
        extended = np.concatenate([self._history, samples], axis=1)
        positions = np.arange(block_size, dtype=float) + self.latency_samples
        aligned = np.empty((self.geometry.microphone_count, block_size))
        for channel_index in range(self.geometry.microphone_count):
            source_positions = positions - self._channel_delays[channel_index]
            aligned[channel_index] = np.interp(
                source_positions,
                np.arange(extended.shape[1], dtype=float),
                extended[channel_index],
            )
        self._history = extended[:, -self.latency_samples :]
        return aligned.mean(axis=0)

    def reset(self) -> None:
        """Clear the internal history, e.g. when the look direction changes."""

        self._history = np.zeros_like(self._history)


def stream_blocks(
    signals: ArrayLike,
    block_size: int,
) -> Iterator[FloatArray]:
    """Yield fixed-size blocks from an offline multichannel recording.

    The final partial block is zero-padded to ``block_size`` so consumers can
    assume a constant block length, mirroring real-time audio callbacks.
    """

    array = np.asarray(signals, dtype=float)
    if array.ndim != 2:
        raise ValueError("signals must have shape (microphones, samples)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total = array.shape[1]
    for start in range(0, total, block_size):
        block = array[:, start : start + block_size]
        if block.shape[1] < block_size:
            block = np.pad(block, ((0, 0), (0, block_size - block.shape[1])))
        yield block
