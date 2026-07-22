"""HearWeave: spatial-audio building blocks for smart wearable devices."""

from .beamforming import delay_and_sum, mvdr_beamform
from .binaural import binaural_coherence_enhance
from .geometry import ArrayGeometry, asymmetric_earbuds_6mic, glasses_4mic
from .localization import gcc_phat, scan_azimuth_energy
from .metrics import si_sdr_db, snr_db
from .simulation import simulate_plane_wave

__all__ = [
    "ArrayGeometry",
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
]

__version__ = "0.1.0"
