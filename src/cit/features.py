"""Polars feature engineering over raw telemetry windows.

The ingestion tier in the pitch uses Polars for "rapid, micro-batch
aggregations of the data stream in real time". This module is exactly that:
it turns a window of raw pointer/keystroke events into the handful of
behavioural-physics features the model scores on.

The features are deliberately interpretable — each one separates human hand
kinematics from scripted/replayed automation:

* ``speed_mean / speed_std`` — pointer velocity and its variability.
* ``accel_std``              — jerkiness; humans accelerate/decelerate erratically.
* ``path_efficiency``        — straight-line distance / path length. Bots ~1.0.
* ``angle_std``              — heading-change variance; bots move in straight lines.
* ``dt_std``                 — sampling cadence jitter; bots are metronomic.
* ``key_interval_std``       — keystroke timing variance; bots type uniformly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FEATURE_COLS = [
    "speed_mean",
    "speed_std",
    "accel_std",
    "path_efficiency",
    "angle_std",
    "dt_std",
    "key_interval_std",
]


def features_from_window(window: pl.DataFrame) -> dict[str, float]:
    """Compute the feature vector for a single window of events."""
    x = window["x"].to_numpy()
    y = window["y"].to_numpy()
    dt = np.clip(window["dt_ms"].to_numpy(), 1e-3, None)
    key_iv = window["key_interval_ms"].to_numpy()

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    speed = step / dt[1:]
    accel = np.diff(speed)

    # Heading change between consecutive steps.
    angles = np.arctan2(dy, dx)
    dangle = np.diff(np.unwrap(angles)) if len(angles) > 1 else np.array([0.0])

    path_len = step.sum() + 1e-6
    straight = np.hypot(x[-1] - x[0], y[-1] - y[0])

    return {
        "speed_mean": float(speed.mean()),
        "speed_std": float(speed.std()),
        "accel_std": float(accel.std()) if accel.size else 0.0,
        "path_efficiency": float(straight / path_len),
        "angle_std": float(dangle.std()) if dangle.size else 0.0,
        "dt_std": float(dt.std()),
        "key_interval_std": float(key_iv.std()),
    }


def windows_from_session(
    session: pl.DataFrame, window: int = 50, stride: int = 25
) -> pl.DataFrame:
    """Slice a session into overlapping windows and featurize each one.

    Returns one row per window with the feature columns, the window's end time,
    and the majority label (used for evaluation / ATO-onset visualisation).
    """
    n = session.height
    rows = []
    for start in range(0, max(1, n - window + 1), stride):
        w = session.slice(start, window)
        if w.height < window // 2:
            continue
        feat = features_from_window(w)
        feat["session_id"] = session["session_id"][0]
        feat["t_end_ms"] = float(w["t_ms"][-1])
        feat["window_idx"] = start // stride
        feat["label"] = int(round(w["label"].mean()))
        rows.append(feat)
    return pl.DataFrame(rows)


def build_feature_table(
    dataset: pl.DataFrame, window: int = 50, stride: int = 25
) -> pl.DataFrame:
    """Featurize every session in a labelled dataset into a training table."""
    frames = [
        windows_from_session(g, window, stride)
        for _, g in dataset.group_by("session_id", maintain_order=True)
    ]
    return pl.concat(frames)
