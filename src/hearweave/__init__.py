"""HearWeave: spatial-audio building blocks for smart wearable devices."""

from .beamforming import delay_and_sum, mvdr_beamform
from .binaural import binaural_coherence_enhance
from .geometry import ArrayGeometry, asymmetric_earbuds_6mic, glasses_4mic
from .localization import gcc_phat, scan_azimuth_energy, srp_phat
from .metrics import si_sdr_db, snr_db
from .simulation import apply_microphone_mismatch, simulate_plane_wave
from .streaming import StreamingDelayAndSum, stream_blocks

__all__ = [
    "ArrayGeometry",
    "StreamingDelayAndSum",
    "apply_microphone_mismatch",
    "asymmetric_earbuds_6mic",
    "binaural_coherence_enhance",
    "delay_and_sum",
    "gcc_phat",
    "glasses_4mic",
    "mvdr_beamform",
    "scan_azimuth_energy",
    "si_sdr_db",
    "simulate_plane_wave",
    "snr_db",
    "srp_phat",
    "stream_blocks",
]

__version__ = "0.2.0"
