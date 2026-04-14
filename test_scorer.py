"""Synthetic-signal tests for the NeuroChat engagement scorer.

Strategy: build fake EEG as pure sinusoids in specific bands + Gaussian noise,
and check that the engagement index E = beta / (alpha + theta) behaves as the
paper describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from scorer import (
    NeuroChatScorer,
    band_power,
    calibrate,
    engagement_index,
    preprocess,
    score_window,
)

FS = 256.0  # Muse 2 sampling rate (paper §3.2.1).
RNG = np.random.default_rng(0)


def synth(freq: float, seconds: float, fs: float = FS, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t) + 0.05 * RNG.standard_normal(t.size)


def test_band_power_peaks_in_correct_band():
    sig = synth(15.0, seconds=4.0)  # beta band
    filt = preprocess(sig, FS)
    beta = band_power(filt, FS, 11, 20)
    alpha = band_power(filt, FS, 7, 11)
    theta = band_power(filt, FS, 4, 7)
    assert beta > alpha * 10
    assert beta > theta * 10


def test_high_beta_signal_yields_high_engagement():
    beta_sig = synth(15.0, seconds=4.0, amp=2.0)
    theta_sig = synth(5.0, seconds=4.0, amp=2.0)
    e_beta = engagement_index(preprocess(beta_sig, FS), FS)
    e_theta = engagement_index(preprocess(theta_sig, FS), FS)
    assert e_beta > e_theta
    assert e_beta > 1.0
    assert e_theta < 0.5


def test_calibration_normalizes_to_unit_range():
    relax = synth(5.0, seconds=120.0, amp=2.0)  # theta-dominant -> low E
    active = synth(15.0, seconds=120.0, amp=2.0)  # beta-dominant -> high E
    e_min, e_max = calibrate(relax, active, FS)
    assert e_max > e_min

    low_window = synth(5.0, seconds=15.0, amp=2.0)
    high_window = synth(15.0, seconds=15.0, amp=2.0)
    low = score_window(low_window, FS, e_min, e_max)
    high = score_window(high_window, FS, e_min, e_max)
    assert 0.0 <= low < 0.25
    assert 0.75 < high <= 1.0


def test_multichannel_preprocessing_runs():
    # Four Muse channels: AF7, AF8, TP9, TP10.
    sig = np.stack([synth(15.0, seconds=4.0) for _ in range(4)])
    filt = preprocess(sig, FS)
    assert filt.shape == sig.shape
    e = engagement_index(filt, FS)
    assert e > 1.0


def test_streaming_scorer_converges_to_window_score():
    relax = synth(5.0, seconds=120.0, amp=2.0)
    active = synth(15.0, seconds=120.0, amp=2.0)
    e_min, e_max = calibrate(relax, active, FS)

    scorer = NeuroChatScorer(fs=FS, e_min=e_min, e_max=e_max, n_channels=1)
    beta_stream = synth(15.0, seconds=20.0, amp=2.0)
    # Feed in 250 ms chunks, as the paper updates every second anyway.
    chunk = int(0.25 * FS)
    for start in range(0, beta_stream.size - chunk, chunk):
        scorer.update(beta_stream[start : start + chunk])
    assert scorer.current_score() > 0.75


def test_calibration_rejects_degenerate_range():
    relax = synth(10.0, seconds=120.0)
    active = synth(10.0, seconds=120.0)
    with pytest.raises(ValueError):
        calibrate(relax, active, FS)


def test_short_window_returns_zero_not_crash():
    # Less than one epoch's worth of samples.
    tiny = synth(15.0, seconds=0.5)
    assert score_window(tiny, FS, e_min=0.1, e_max=1.0) == 0.0
