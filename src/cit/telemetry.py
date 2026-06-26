"""Synthetic behavioral telemetry generator.

In production this data would arrive from the Wasm client agent described in the
pitch (mouse kinematics + keystroke dynamics streamed over WebSockets). For the
prototype we synthesize statistically realistic sessions so the rest of the
pipeline — features, scoring, step-up — can run end-to-end with zero infra.

Two population types are modelled:

* ``human``  — smooth, curved mouse paths with natural velocity jitter and
  human keystroke timing (~150-250ms between keys, with variance).
* ``bot``    — straight-line teleport-like movements, near-constant velocity,
  and robotic, low-variance keystroke timing (scripted typing / replay).

An ``account_takeover`` session starts as a human and switches to bot behaviour
partway through — this is the mid-session ATO the engine is designed to catch.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Each telemetry "event" is one sampled pointer/keystroke observation.
EVENT_SCHEMA = ["session_id", "t_ms", "x", "y", "dt_ms", "key_interval_ms", "label"]


def _human_segment(rng: np.random.Generator, n: int, t0: float) -> dict:
    """A burst of human-like pointer + typing events."""
    # Curved path via a random Bezier-ish control point.
    start = rng.uniform(0, 1920, size=2)
    end = rng.uniform(0, 1920, size=2)
    ctrl = (start + end) / 2 + rng.normal(0, 300, size=2)
    ts = np.linspace(0, 1, n)
    xs = (1 - ts) ** 2 * start[0] + 2 * (1 - ts) * ts * ctrl[0] + ts**2 * end[0]
    ys = (1 - ts) ** 2 * start[1] + 2 * (1 - ts) * ts * ctrl[1] + ts**2 * end[1]
    # Natural hand jitter.
    xs += rng.normal(0, 4, size=n)
    ys += rng.normal(0, 4, size=n)
    # Variable sampling cadence (humans aren't metronomes).
    dt = np.clip(rng.normal(28, 9, size=n), 8, None)
    t = t0 + np.cumsum(dt)
    # Keystroke inter-key intervals: ~190ms mean, high variance.
    key_iv = np.clip(rng.normal(190, 55, size=n), 40, None)
    return {"x": xs, "y": ys, "t": t, "dt": dt, "key_iv": key_iv}


def _bot_segment(rng: np.random.Generator, n: int, t0: float) -> dict:
    """A burst of robotic pointer + typing events.

    Different automation frameworks inject different amounts of fake jitter, so
    each bot session draws its own ``sophistication`` level. Crude bots are easy;
    sophisticated ones add enough timing noise to overlap the human population —
    which is what keeps the evaluation honest rather than a toy.
    """
    # Different automation frameworks inject different amounts of fake jitter.
    sophistication = rng.uniform(0.0, 1.0)
    # Endpoints in roughly opposite regions -> a long, clearly straight traverse.
    start = rng.uniform(0, 600, size=2)
    end = rng.uniform(1320, 1920, size=2)
    ts = np.linspace(0, 1, n)
    pos_jitter = 0.5 + 5.0 * sophistication
    xs = start[0] + (end[0] - start[0]) * ts + rng.normal(0, pos_jitter, size=n)
    ys = start[1] + (end[1] - start[1]) * ts + rng.normal(0, pos_jitter, size=n)
    # Cadence is more regular than a human's; sophisticated bots add timing
    # jitter that creeps toward the human range (so the classes overlap).
    dt = np.clip(rng.normal(18, 1.5 + 9.0 * sophistication, size=n), 8, None)
    t = t0 + np.cumsum(dt)
    key_iv = np.clip(rng.normal(85, 12 + 45 * sophistication, size=n), 30, None)
    return {"x": xs, "y": ys, "t": t, "dt": dt, "key_iv": key_iv}


def generate_session(
    session_id: str,
    kind: str = "human",
    n_events: int = 600,
    takeover_at: float = 0.5,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate one session worth of telemetry events.

    kind: ``"human"``, ``"bot"`` or ``"account_takeover"``.
    takeover_at: fraction of the session at which an ATO flips to bot behaviour.
    """
    rng = np.random.default_rng(seed)

    if kind == "human":
        seg = _human_segment(rng, n_events, t0=0.0)
        label = np.zeros(n_events, dtype=np.int8)
    elif kind == "bot":
        seg = _bot_segment(rng, n_events, t0=0.0)
        label = np.ones(n_events, dtype=np.int8)
    elif kind == "account_takeover":
        cut = int(n_events * takeover_at)
        h = _human_segment(rng, cut, t0=0.0)
        b = _bot_segment(rng, n_events - cut, t0=float(h["t"][-1]))
        seg = {k: np.concatenate([h[k], b[k]]) for k in h}
        label = np.concatenate(
            [np.zeros(cut, dtype=np.int8), np.ones(n_events - cut, dtype=np.int8)]
        )
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    return pl.DataFrame(
        {
            "session_id": session_id,
            "t_ms": seg["t"],
            "x": seg["x"],
            "y": seg["y"],
            "dt_ms": seg["dt"],
            "key_interval_ms": seg["key_iv"],
            "label": label,
        }
    )


def generate_dataset(
    n_human: int = 120,
    n_bot: int = 120,
    n_events: int = 400,
    seed: int = 7,
) -> pl.DataFrame:
    """A labelled training corpus of complete human and bot sessions.

    Mirrors the "historically partitioned Parquet datasets of normal behaviour"
    in the pitch — here it's the baseline the model learns from.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_human):
        frames.append(
            generate_session(f"h{i:04d}", "human", n_events, seed=int(rng.integers(1e9)))
        )
    for i in range(n_bot):
        frames.append(
            generate_session(f"b{i:04d}", "bot", n_events, seed=int(rng.integers(1e9)))
        )
    return pl.concat(frames)
