"""A tiny "edge" model that can run inside the Wasm agent / the browser.

The gradient-boosted model in ``engine.py`` is the real inference brain, but a
tree ensemble is awkward to ship into a 20 kB Wasm module. So we also fit a plain
logistic-regression scorer on the exact same features. It's a bit less accurate,
but it's just a scale + a dot product + a sigmoid, which means the client agent
can produce a first-pass risk score locally, with no round-trip, before the
server ever sees the stream.

``export_edge_model`` dumps everything the browser needs (feature order, the
standardiser, the weights) into a small JSON file that ``live.html`` loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS


def train_edge_model(feature_table: pl.DataFrame, seed: int = 0):
    """Fit the logistic edge scorer. Returns (scaler, clf)."""
    X = feature_table.select(FEATURE_COLS).to_numpy()
    y = feature_table["label"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def export_edge_model(
    scaler: StandardScaler,
    clf: LogisticRegression,
    path: str | Path,
) -> dict:
    """Write the edge model to JSON for the in-browser scorer.

    Scoring in the browser is then:  z = (x - mean) / scale ;  p = sigmoid(w.z + b)
    """
    payload = {
        "features": FEATURE_COLS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "note": "logistic edge model; p_bot = sigmoid(coef . ((x-mean)/scale) + intercept)",
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2))
    return payload
