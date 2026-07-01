"""The continuous risk engine: model + live session state + step-up logic.

This is the "Inference & State" tier from the pitch, minus the infra. A
gradient-boosted classifier learns the human/bot boundary from behavioural
features; at inference time each incoming window produces an instantaneous
risk probability, which is fed through an exponential moving average to form
the smoothed, continuous "risk state" of the session (the value that would
live in Redis in production).

When the smoothed risk crosses ``step_up_threshold`` the engine raises a
step-up-authentication signal — the active feedback loop the pitch describes.
It also emits *reason codes* so the step-up isn't a black box: we can say which
behavioural features drifted away from the user's human baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, features_from_window

# Plain-language description of what each feature drifting "toward bot" means,
# used when we surface reason codes at step-up time.
_REASON_TEXT = {
    "speed_mean": "pointer speed",
    "speed_std": "speed consistency",
    "accel_std": "acceleration smoothness",
    "path_efficiency": "path straightness",
    "angle_std": "movement direction changes",
    "dt_std": "sampling cadence regularity",
    "key_interval_std": "keystroke timing regularity",
}


@dataclass
class RiskModel:
    """Trained behavioural classifier + scaler + a human baseline.

    The baseline (per-feature mean/std over genuine human windows, plus the
    direction each feature moves when a session turns bot-like) is what lets us
    explain a decision instead of just returning a number.
    """

    scaler: StandardScaler
    clf: GradientBoostingClassifier
    human_mean: np.ndarray
    human_std: np.ndarray
    bot_direction: np.ndarray  # +1 if a higher value is more bot-like, else -1

    def score_windows(self, feature_table: pl.DataFrame) -> np.ndarray:
        """Return P(bot) for each window row in a feature table."""
        X = feature_table.select(FEATURE_COLS).to_numpy()
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def score_one(self, feature_vec: dict[str, float]) -> float:
        X = np.array([[feature_vec[c] for c in FEATURE_COLS]])
        return float(self.clf.predict_proba(self.scaler.transform(X))[0, 1])

    def explain(self, feature_vec: dict[str, float], top_k: int = 3) -> list[dict]:
        """Which features look least human right now, and by how much.

        For each feature we measure how many standard deviations it sits from the
        human baseline, signed so that positive means "more bot-like". The biggest
        positive deviations are the reason codes.
        """
        x = np.array([feature_vec[c] for c in FEATURE_COLS])
        z = (x - self.human_mean) / self.human_std
        toward_bot = z * self.bot_direction
        order = np.argsort(toward_bot)[::-1]
        reasons = []
        for i in order[:top_k]:
            if toward_bot[i] <= 0.5:  # still within normal human range, skip
                continue
            reasons.append(
                {
                    "feature": FEATURE_COLS[i],
                    "what": _REASON_TEXT[FEATURE_COLS[i]],
                    "sigma": float(toward_bot[i]),
                }
            )
        return reasons


def train_model(feature_table: pl.DataFrame, seed: int = 0) -> RiskModel:
    """Fit the risk model on a labelled feature table."""
    X = feature_table.select(FEATURE_COLS).to_numpy()
    y = feature_table["label"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed
    )
    clf.fit(scaler.transform(X), y)

    # Human baseline + which way each feature leans when things go bot-like.
    human = X[y == 0]
    bot = X[y == 1]
    human_mean = human.mean(axis=0)
    human_std = human.std(axis=0) + 1e-9
    bot_direction = np.sign(bot.mean(axis=0) - human_mean)

    return RiskModel(
        scaler=scaler,
        clf=clf,
        human_mean=human_mean,
        human_std=human_std,
        bot_direction=bot_direction,
    )


@dataclass
class SessionMonitor:
    """Stateful, streaming risk tracker for a single live session.

    Feed it raw event windows one at a time (as a Wasm agent would stream them);
    it maintains the EMA-smoothed risk state and flips ``stepped_up`` the first
    time risk crosses the threshold. When that happens it snapshots the reason
    codes for the window that tipped it over.
    """

    model: RiskModel
    ema_alpha: float = 0.15
    step_up_threshold: float = 0.6
    risk: float = 0.0
    stepped_up: bool = False
    step_up_t_ms: float | None = None
    step_up_reasons: list[dict] = field(default_factory=list)
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
            self.step_up_reasons = self.model.explain(feat)

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
