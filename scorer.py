"""NeuroChat EEG engagement scorer.

Implements the scoring pipeline from Baradari et al., "NeuroChat: A
Neuroadaptive AI Chatbot for Customizing Learning Experiences" (CUI '25).

Pipeline (paper §3.2):
    1. Bandpass 1-30 Hz + 60 Hz notch filter
    2. Segment into 1-second epochs (250 ms hop)
    3. Welch PSD -> band powers for theta (4-7), alpha (7-11), beta (11-20)
    4. Engagement index E = beta / (alpha + theta)  (Pope et al., 1995)
    5. 15-second sliding window mean
    6. Min-max normalize with per-user calibration (relax -> E_min,
       mental effort task -> E_max)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (7.0, 11.0)
BETA_BAND = (11.0, 20.0)

EPOCH_SEC = 1.0
HOP_SEC = 0.25
WINDOW_SEC = 15.0


def _design_filter(fs: float) -> np.ndarray:
    """Bandpass 1-30 Hz cascaded with a 60 Hz notch, returned as SOS."""
    nyq = 0.5 * fs
    bp_sos = butter(4, [1.0 / nyq, 30.0 / nyq], btype="band", output="sos")
    notch_b, notch_a = iirnotch(w0=60.0 / nyq, Q=30.0)
    notch_sos = tf2sos(notch_b, notch_a)
    return np.vstack([bp_sos, notch_sos])


def preprocess(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase bandpass + notch filter.

    eeg: shape (n_samples,) or (n_channels, n_samples).
    """
    sos = _design_filter(fs)
    return sosfiltfilt(sos, eeg, axis=-1)


def band_power(epoch: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Integrated PSD (Welch) in [lo, hi] Hz, averaged across channels."""
    epoch = np.atleast_2d(epoch)
    nperseg = min(epoch.shape[-1], int(fs))
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)
    mask = (freqs >= lo) & (freqs < hi)
    # Trapezoid integration over the band, then mean across channels.
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(psd[..., mask], freqs[mask], axis=-1).mean())


def engagement_index(epoch: np.ndarray, fs: float) -> float:
    """Raw Pope engagement index E = beta / (alpha + theta)."""
    theta = band_power(epoch, fs, *THETA_BAND)
    alpha = band_power(epoch, fs, *ALPHA_BAND)
    beta = band_power(epoch, fs, *BETA_BAND)
    denom = alpha + theta
    if denom <= 0:
        return 0.0
    return beta / denom


def _epoch_scores(eeg: np.ndarray, fs: float) -> list[float]:
    """Run the filter + sliding epoch + engagement pipeline on a segment."""
    filtered = preprocess(eeg, fs)
    filtered = np.atleast_2d(filtered)
    n_samples = filtered.shape[-1]
    epoch_len = int(round(EPOCH_SEC * fs))
    hop = int(round(HOP_SEC * fs))
    if n_samples < epoch_len:
        return []
    scores: list[float] = []
    for start in range(0, n_samples - epoch_len + 1, hop):
        epoch = filtered[..., start : start + epoch_len]
        scores.append(engagement_index(epoch, fs))
    return scores


def calibrate(
    relax_eeg: np.ndarray, active_eeg: np.ndarray, fs: float
) -> tuple[float, float]:
    """Compute (E_min, E_max) from the two 2-minute calibration tasks."""
    relax_scores = _epoch_scores(relax_eeg, fs)
    active_scores = _epoch_scores(active_eeg, fs)
    if not relax_scores or not active_scores:
        raise ValueError("Calibration segments too short for a 1-second epoch.")
    e_min = float(np.mean(relax_scores))
    e_max = float(np.mean(active_scores))
    if e_max <= e_min:
        raise ValueError(
            f"Calibration failed: E_max ({e_max:.4f}) <= E_min ({e_min:.4f})."
        )
    return e_min, e_max


def score_window(
    eeg_window: np.ndarray, fs: float, e_min: float, e_max: float
) -> float:
    """One-shot score for a ~15 s window of EEG, clipped to [0, 1]."""
    scores = _epoch_scores(eeg_window, fs)
    if not scores:
        return 0.0
    raw = float(np.mean(scores))
    return float(np.clip((raw - e_min) / (e_max - e_min), 0.0, 1.0))


@dataclass
class NeuroChatScorer:
    """Stateful scorer for streaming EEG.

    Hold calibration constants and a ring buffer of the most recent
    ``window_sec`` seconds of samples. Call ``update`` with new samples as
    they arrive; read ``latest`` (or call ``current_score``) whenever the
    LLM request is about to be sent.
    """

    fs: float
    e_min: float
    e_max: float
    n_channels: int = 1
    window_sec: float = WINDOW_SEC

    def __post_init__(self) -> None:
        self._maxlen = int(round(self.window_sec * self.fs))
        self._buffers: list[deque[float]] = [
            deque(maxlen=self._maxlen) for _ in range(self.n_channels)
        ]
        self.latest: float = 0.0

    def update(self, samples: np.ndarray) -> float:
        """Append new samples (shape (n_channels, n) or (n,)) and rescore."""
        samples = np.atleast_2d(samples)
        if samples.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {samples.shape[0]}."
            )
        for ch, row in zip(self._buffers, samples):
            ch.extend(row.tolist())
        if len(self._buffers[0]) < int(round(EPOCH_SEC * self.fs)):
            return self.latest
        window = np.array([list(ch) for ch in self._buffers])
        self.latest = score_window(window, self.fs, self.e_min, self.e_max)
        return self.latest

    def current_score(self) -> float:
        return self.latest
