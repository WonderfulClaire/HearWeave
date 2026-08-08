"""Validated adapters for real or synthetic multichannel WAV recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from .geometry import ArrayGeometry

FloatArray = NDArray[np.float64]
SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MultichannelRecording:
    """One validated recording with channels ordered to match its geometry."""

    recording_id: str
    signals: FloatArray
    sample_rate_hz: int
    geometry: ArrayGeometry
    metadata: dict[str, Any]


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be a non-empty string")
    return value.strip()


def _resolve_channel_file(root: Path, value: object, index: int) -> Path:
    relative = Path(_required_text(value, f"channels[{index}].file"))
    if relative.is_absolute():
        raise ValueError(f"channels[{index}].file must be relative to the manifest")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"channels[{index}].file escapes the manifest directory") from error
    return resolved


def _pcm_to_float(samples: np.ndarray) -> FloatArray:
    if np.issubdtype(samples.dtype, np.floating):
        output = samples.astype(float)
    elif np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype)
        output = samples.astype(float) / max(abs(info.min), info.max)
    elif samples.dtype == np.uint8:
        output = (samples.astype(float) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample dtype: {samples.dtype}")
    if not np.isfinite(output).all():
        raise ValueError("WAV samples must be finite")
    return output


def load_recording(manifest_path: str | Path) -> MultichannelRecording:
    """Load a manifest and its mono WAV channels in declared geometry order.

    The manifest is the authority for channel order, positions, and sample rate.
    Channel paths must stay below the manifest directory; files with implicit or
    conflicting layout information are rejected rather than guessed.
    """

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid recording manifest JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("recording manifest must contain a JSON object")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SUPPORTED_SCHEMA_VERSION}")

    recording_id = _required_text(payload.get("recording_id"), "recording_id")
    sample_rate_hz = payload.get("sample_rate_hz")
    valid_sample_rate = (
        isinstance(sample_rate_hz, int)
        and not isinstance(sample_rate_hz, bool)
        and sample_rate_hz > 0
    )
    if not valid_sample_rate:
        raise ValueError("'sample_rate_hz' must be a positive integer")
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) < 2:
        raise ValueError("'channels' must contain at least two channel objects")

    labels: list[str] = []
    positions: list[list[float]] = []
    signals: list[FloatArray] = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            raise ValueError(f"channels[{index}] must be an object")
        labels.append(_required_text(channel.get("label"), f"channels[{index}].label"))
        position = np.asarray(channel.get("position_m"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"channels[{index}].position_m must contain three finite metres")
        positions.append(position.tolist())
        channel_path = _resolve_channel_file(path.parent, channel.get("file"), index)
        if not channel_path.is_file():
            raise FileNotFoundError(f"channel WAV not found: {channel_path}")
        actual_rate, samples = wavfile.read(channel_path)
        if actual_rate != sample_rate_hz:
            raise ValueError(
                f"sample rate mismatch for {channel_path.name}: "
                f"manifest={sample_rate_hz}, wav={actual_rate}"
            )
        if samples.ndim != 1:
            raise ValueError(f"channel WAV must be mono: {channel_path.name}")
        signals.append(_pcm_to_float(samples))

    if len(set(labels)) != len(labels):
        raise ValueError("channel labels must be unique")
    lengths = {signal.size for signal in signals}
    if len(lengths) != 1:
        raise ValueError("all channel WAV files must have equal sample counts")

    geometry = ArrayGeometry(
        name=_required_text(payload.get("geometry_name"), "geometry_name"),
        positions_m=np.asarray(positions, dtype=float),
        labels=tuple(labels),
    )
    reserved = {
        "schema_version",
        "recording_id",
        "sample_rate_hz",
        "geometry_name",
        "channels",
    }
    metadata = {key: value for key, value in payload.items() if key not in reserved}
    return MultichannelRecording(
        recording_id=recording_id,
        signals=np.vstack(signals),
        sample_rate_hz=sample_rate_hz,
        geometry=geometry,
        metadata=metadata,
    )
