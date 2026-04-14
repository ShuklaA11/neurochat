# NeuroChat

EEG-based engagement scoring sidecar for neuroadaptive chatbots.

NeuroChat computes a normalized cognitive-engagement score in real time from raw EEG (e.g., Muse 2) and exposes it over HTTP so a conversational UI (Open WebUI, custom frontends) can inject the current engagement level into LLM prompts — letting the model adapt its responses to the user's cognitive state.

The scoring pipeline implements the Pope et al. (1995) engagement index `β / (α + θ)` with per-user min–max calibration, following the method described in Baradari et al., *NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences* (CUI '25).

## How it works

1. Raw EEG samples arrive via `POST /samples` (or the built-in synthetic demo feeder).
2. Signals are bandpass-filtered (1–30 Hz) with a 60 Hz notch filter.
3. 1-second epochs are analyzed with Welch's method to estimate power in θ (4–8 Hz), α (8–13 Hz), and β (13–30 Hz) bands.
4. The engagement index `E = β / (α + θ)` is computed per epoch and smoothed with a 15-second rolling mean.
5. A per-user calibration (two short relax/active segments) maps the raw index to a normalized `[0, 1]` score via min–max scaling.
6. `GET /score` returns the latest normalized score on demand.

## Quickstart

```bash
git clone https://github.com/ShuklaA11/neurochat.git
cd neurochat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the sidecar (default port 8765)
python -m uvicorn server:app --host 0.0.0.0 --port 8765

# In another terminal — kick off the synthetic EEG demo (no hardware needed)
curl -X POST http://localhost:8765/demo/start
curl http://localhost:8765/score
```

Run the tests:

```bash
pytest
```

## Endpoints

| Method | Path            | Description                                                       |
| ------ | --------------- | ----------------------------------------------------------------- |
| GET    | `/health`       | Service status, calibration flag, and latest score                |
| POST   | `/calibrate`    | Compute `E_min`/`E_max` from relax and active EEG segments        |
| POST   | `/samples`      | Push new EEG samples (`n_channels × n` matrix) into the buffer    |
| GET    | `/score`        | Return the latest normalized engagement score in `[0, 1]`         |
| POST   | `/demo/start`   | Start a synthetic EEG feeder (oscillating β/θ mix, no hardware)   |
| POST   | `/demo/stop`    | Stop the synthetic feeder                                         |

Default sampling rate is 256 Hz with 4 channels (AF7, AF8, TP9, TP10 for Muse 2). These can be overridden in the scorer constructor.

## Status

Functional prototype. The HTTP API and scoring pipeline are complete and unit-tested against synthetic signals. Live Muse ingestion is not yet wired in — use `/demo/start` to produce a live-looking score without hardware.

## Citation

If you use this work, please cite the original paper:

> Baradari, A., et al. (2025). *NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences*. In Proceedings of the ACM Conference on Conversational User Interfaces (CUI '25).

The engagement index itself is from:

> Pope, A. T., Bogart, E. H., & Bartolome, D. S. (1995). Biocybernetic system evaluates indices of operator engagement in automated task. *Biological Psychology*, 40(1–2), 187–195.

## License

MIT — see [LICENSE](LICENSE).
