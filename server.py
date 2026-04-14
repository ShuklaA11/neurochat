"""NeuroChat scoring sidecar.

Small FastAPI service that holds a stateful ``NeuroChatScorer`` and exposes
it over HTTP so the Open WebUI filter (running inside Docker) can fetch the
current engagement score to inject into each prompt.

Endpoints
---------
GET  /health              -> {"ok": true, "calibrated": bool, "score": float}
POST /calibrate           -> compute (E_min, E_max) from two EEG segments
POST /samples             -> push new EEG samples into the rolling buffer
GET  /score               -> latest normalized engagement score in [0, 1]
POST /demo/start          -> start a background thread feeding synthetic EEG
POST /demo/stop           -> stop the demo feeder

Usage
-----
    pip install fastapi uvicorn numpy scipy
    python -m uvicorn server:app --host 0.0.0.0 --port 8765

From the Muse reader (future), POST to /samples periodically. For now, hit
/demo/start to get a live-looking score without hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scorer import NeuroChatScorer, calibrate


def _mixed_signal(beta_amp: float, theta_amp: float, fs: float, seconds: float, seed: int) -> np.ndarray:
    """Beta + theta sines plus noise, matching the demo feeder's shape."""
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(seed)
    return (
        beta_amp * np.sin(2 * np.pi * 15.0 * t)
        + theta_amp * np.sin(2 * np.pi * 5.0 * t)
        + 0.05 * rng.standard_normal(t.size)
    )


def _synth_calibration(fs: float) -> tuple[float, float]:
    """Generate 120 s relax/active segments whose band ratios match what the
    demo feeder actually produces, so normalized scores span [0, 1]."""
    relax = _mixed_signal(beta_amp=0.5, theta_amp=2.0, fs=fs, seconds=120.0, seed=7)
    active = _mixed_signal(beta_amp=2.0, theta_amp=0.5, fs=fs, seconds=120.0, seed=8)
    return calibrate(relax, active, fs)

DEFAULT_FS = 256.0
DEFAULT_CHANNELS = 4  # Muse 2: AF7, AF8, TP9, TP10.

app = FastAPI(title="NeuroChat Scorer", version="0.1.0")

_state_lock = threading.Lock()
_scorer: Optional[NeuroChatScorer] = None
_demo_thread: Optional[threading.Thread] = None
_demo_stop = threading.Event()


def _ensure_scorer(
    fs: float = DEFAULT_FS,
    e_min: float = 0.3,
    e_max: float = 1.5,
    n_channels: int = DEFAULT_CHANNELS,
) -> NeuroChatScorer:
    """Initialize a scorer with placeholder calibration if none exists yet."""
    global _scorer
    with _state_lock:
        if _scorer is None:
            _scorer = NeuroChatScorer(
                fs=fs, e_min=e_min, e_max=e_max, n_channels=n_channels
            )
        return _scorer


class CalibrateRequest(BaseModel):
    fs: float = DEFAULT_FS
    n_channels: int = DEFAULT_CHANNELS
    relax_eeg: list  # shape (n_channels, n_samples) or (n_samples,)
    active_eeg: list


class CalibrateResponse(BaseModel):
    e_min: float
    e_max: float


class SamplesRequest(BaseModel):
    samples: list  # shape (n_channels, n) or (n,)


class ScoreResponse(BaseModel):
    score: float
    calibrated: bool


@app.get("/health")
def health() -> dict:
    s = _scorer
    return {
        "ok": True,
        "calibrated": s is not None,
        "score": float(s.latest) if s else 0.0,
        "fs": s.fs if s else None,
        "n_channels": s.n_channels if s else None,
    }


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate_endpoint(req: CalibrateRequest) -> CalibrateResponse:
    global _scorer
    relax = np.asarray(req.relax_eeg, dtype=float)
    active = np.asarray(req.active_eeg, dtype=float)
    try:
        e_min, e_max = calibrate(relax, active, req.fs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _state_lock:
        _scorer = NeuroChatScorer(
            fs=req.fs,
            e_min=e_min,
            e_max=e_max,
            n_channels=req.n_channels,
        )
    return CalibrateResponse(e_min=e_min, e_max=e_max)


@app.post("/samples")
def push_samples(req: SamplesRequest) -> dict:
    scorer = _ensure_scorer()
    arr = np.asarray(req.samples, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] != scorer.n_channels:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {scorer.n_channels} channels, "
                f"got shape {arr.shape}."
            ),
        )
    score = scorer.update(arr)
    return {"score": float(score), "buffered_samples": len(scorer._buffers[0])}


@app.get("/score", response_model=ScoreResponse)
def get_score() -> ScoreResponse:
    s = _scorer
    if s is None:
        return ScoreResponse(score=0.5, calibrated=False)
    return ScoreResponse(score=float(s.latest), calibrated=True)


def _demo_loop() -> None:
    """Feed synthetic EEG that drifts between beta-dominant and theta-dominant.

    Lets you eyeball the filter responding to score changes without a Muse.
    """
    scorer = _ensure_scorer()
    fs = scorer.fs
    chunk_sec = 0.25
    n = int(chunk_sec * fs)
    t0 = time.time()
    rng = np.random.default_rng(1)
    while not _demo_stop.is_set():
        elapsed = time.time() - t0
        # Slowly oscillate the "engagement" target between 0 and 1 over 60 s.
        target = 0.5 + 0.5 * np.sin(2 * np.pi * elapsed / 60.0)
        beta_amp = 0.5 + 1.5 * target
        theta_amp = 0.5 + 1.5 * (1.0 - target)
        t = (np.arange(n) + int(elapsed * fs)) / fs
        beta_wave = beta_amp * np.sin(2 * np.pi * 15.0 * t)
        theta_wave = theta_amp * np.sin(2 * np.pi * 5.0 * t)
        sample = beta_wave + theta_wave + 0.05 * rng.standard_normal(n)
        block = np.tile(sample, (scorer.n_channels, 1))
        scorer.update(block)
        time.sleep(chunk_sec)


@app.post("/demo/start")
def demo_start() -> dict:
    global _scorer, _demo_thread
    with _state_lock:
        e_min, e_max = _synth_calibration(DEFAULT_FS)
        _scorer = NeuroChatScorer(
            fs=DEFAULT_FS,
            e_min=e_min,
            e_max=e_max,
            n_channels=DEFAULT_CHANNELS,
        )
    if _demo_thread and _demo_thread.is_alive():
        return {"running": True, "e_min": e_min, "e_max": e_max}
    _demo_stop.clear()
    _demo_thread = threading.Thread(target=_demo_loop, daemon=True)
    _demo_thread.start()
    return {"running": True, "e_min": e_min, "e_max": e_max}


@app.post("/demo/stop")
def demo_stop() -> dict:
    _demo_stop.set()
    return {"running": False}
