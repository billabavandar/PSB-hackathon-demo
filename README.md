# Continuous Identity Trust

**Shifting from gatekeepers to real-time behavioural inference.**

Point-in-time login is a front door: once a user is in, they're implicitly trusted
for the whole session. That's the exact gap that mid-session **account takeover**,
session hijacking, and replay bots walk through. Continuous Identity Trust treats
trust as a **continuously inferred state** instead of a one-time gate — monitoring
behavioural biometrics in the background and stepping up authentication *only* when
the risk is mathematically real, so genuine users never feel it.

This repo is a self-contained, runnable **prototype** of that idea. It demonstrates
the full decision loop with zero infrastructure.

> 📊 The pitch deck is in [`pitch.html`](pitch.html) (open in any browser).
> 📓 The live walkthrough is in [`demo.ipynb`](demo.ipynb).

---

## The loop, demonstrated

```
 Telemetry            Features            Inference           Feedback loop
 (Wasm agent)   -->   (Polars)      -->   (ML risk score) --> (step-up auth)
 mouse + keys         7 behavioural       continuous risk     fires mid-session
 kinematics           physics features    state (EMA)         the instant risk spikes
```

1. **Telemetry** — high-frequency mouse kinematics + keystroke timing. *(Synthetic
   here so the demo is self-contained; in production it streams from the Wasm client
   agent over WebSockets.)*
2. **Features** — Polars micro-batch aggregation turns each window into seven
   interpretable behavioural-physics features (path efficiency, heading variance,
   cadence jitter, keystroke timing variance, …).
3. **Inference** — a gradient-boosted model scores each window; an exponential moving
   average smooths it into the **continuous risk state** of the session.
4. **Feedback loop** — when the smoothed risk crosses the threshold, the engine raises
   **step-up authentication** automatically, mid-session.

The headline scenario: a session that **starts as a legitimate human, passes login,
and is hijacked by a script halfway through.** Static auth never sees it. This engine
catches it ~1.5 seconds after the behaviour turns malicious.

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # install everything into a local .venv

uv run quickstart.py    # headless end-to-end demo (no Jupyter needed)

uv run jupyter lab demo.ipynb   # full visual walkthrough with charts
```

Headless output looks like:

```
held-out ROC-AUC : 1.0000
Replaying an account-takeover session (human -> bot at 50%)...
  takeover at      :   9.8s
  STEP-UP fired at :  11.2s  (latency 1.4s)
```

---

## Project layout

```
pitch.html        the hackathon pitch deck (reveal.js)
demo.ipynb        full visual walkthrough — the main deliverable
quickstart.py     headless end-to-end run
src/cit/
  telemetry.py    synthetic behavioural telemetry (human / bot / takeover)
  features.py     Polars feature engineering over event windows
  engine.py       risk model + streaming SessionMonitor + step-up logic
```

## Prototype → production

The decision core — features, model, continuous-risk state machine, and step-up
feedback loop — is **real** in this prototype. Productionising is a matter of swapping
the data source and scaling the runtime:

| Tier          | Prototype             | Production (per pitch)              |
| ------------- | --------------------- | ----------------------------------- |
| Telemetry     | synthetic generator   | Wasm (Rust/C++) client agent        |
| Ingestion     | in-process Polars     | async Python + WebSockets + Polars  |
| Inference     | scikit-learn GBM      | PyTorch / XGBoost, Numba features   |
| Session state | `SessionMonitor`      | Redis in-memory risk state          |
| Action        | `stepped_up` flag     | API call forcing step-up auth       |

---

*Hackathon prototype. Telemetry is synthetic; everything downstream of ingestion is
the real engine.*
