"""The continuous risk engine: model + live session state + step-up logic.

This is the "Inference & State" tier from the pitch, minus the infra. A
gradient-boosted classifier learns the human/bot boundary from behavioural
features; at inference time each incoming window produces an instantaneous
risk probability, which is fed through an exponential moving average to form
the smoothed, continuous "risk state" of the session (the value that would
live in Redis in production).

When the smoothed risk crosses ``step_up_threshold`` the engine raises a
step-up-authentication signal — the active feedback loop the pitch describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, features_from_window


@dataclass
class RiskModel:
    """Trained behavioural classifier + scaler."""

    scaler: StandardScaler
    clf: GradientBoostingClassifier

    def score_windows(self, feature_table: pl.DataFrame) -> np.ndarray:
        """Return P(bot) for each window row in a feature table."""
        X = feature_table.select(FEATURE_COLS).to_numpy()
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def score_one(self, feature_vec: dict[str, float]) -> float:
        X = np.array([[feature_vec[c] for c in FEATURE_COLS]])
        return float(self.clf.predict_proba(self.scaler.transform(X))[0, 1])


def train_model(feature_table: pl.DataFrame, seed: int = 0) -> RiskModel:
    """Fit the risk model on a labelled feature table."""
    X = feature_table.select(FEATURE_COLS).to_numpy()
    y = feature_table["label"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed
    )
    clf.fit(scaler.transform(X), y)
    return RiskModel(scaler=scaler, clf=clf)


@dataclass
class SessionMonitor:
    """Stateful, streaming risk tracker for a single live session.

    Feed it raw event windows one at a time (as a Wasm agent would stream them);
    it maintains the EMA-smoothed risk state and flips ``stepped_up`` the first
    time risk crosses the threshold.
    """

    model: RiskModel
    ema_alpha: float = 0.3
    step_up_threshold: float = 0.6
    risk: float = 0.0
    stepped_up: bool = False
    step_up_t_ms: float | None = None
    history: list[dict] = field(default_factory=list)

    def update(self, window: pl.DataFrame) -> dict:
        """Process one window, update risk state, return the latest snapshot."""
        feat = features_from_window(window)
        instant = self.model.score_one(feat)
        # Exponential moving average -> smoothed continuous risk.
        self.risk = self.ema_alpha * instant + (1 - self.ema_alpha) * self.risk
        t_end = float(window["t_ms"][-1])

        if not self.stepped_up and self.risk >= self.step_up_threshold:
            self.stepped_up = True
            self.step_up_t_ms = t_end

        snap = {
            "t_end_ms": t_end,
            "instant_risk": instant,
            "risk_state": self.risk,
            "stepped_up": self.stepped_up,
        }
        self.history.append(snap)
        return snap

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.history)


def replay_session(
    model: RiskModel,
    session: pl.DataFrame,
    window: int = 50,
    stride: int = 25,
    **monitor_kwargs,
) -> SessionMonitor:
    """Stream a whole session through a fresh monitor, window by window."""
    mon = SessionMonitor(model=model, **monitor_kwargs)
    n = session.height
    for start in range(0, max(1, n - window + 1), stride):
        w = session.slice(start, window)
        if w.height < window // 2:
            continue
        mon.update(w)
    return mon
