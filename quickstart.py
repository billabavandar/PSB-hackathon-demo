"""Headless end-to-end demo — run the whole engine without Jupyter.

    uv run quickstart.py

Generates a behavioural corpus, trains the risk model, reports held-out
accuracy, then streams a mid-session account-takeover through the engine and
shows where step-up authentication fires (and *why*). Also re-exports the
edge model that the in-browser live demo uses.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from cit import (
    build_feature_table,
    export_edge_model,
    generate_dataset,
    generate_session,
    replay_session,
    train_edge_model,
    train_model,
)


def main() -> None:
    print("Continuous Identity Trust — headless demo\n" + "=" * 42)

    # 1. Behavioural corpus -> Polars feature windows.
    dataset = generate_dataset(n_human=120, n_bot=120, n_events=400, seed=7)
    feat = build_feature_table(dataset)
    print(f"telemetry events : {dataset.height:,}")
    print(f"feature windows  : {feat.height:,} over {feat['session_id'].n_unique()} sessions")

    # 2. Session-level split + train.
    sids = feat["session_id"].unique().to_list()
    rng = np.random.default_rng(0)
    rng.shuffle(sids)
    cut = int(len(sids) * 0.3)
    test = feat.filter(pl.col("session_id").is_in(sids[:cut]))
    train = feat.filter(pl.col("session_id").is_in(sids[cut:]))
    model = train_model(train)
    auc = roc_auc_score(test["label"].to_numpy(), model.score_windows(test))
    print(f"held-out ROC-AUC : {auc:.4f}")

    # 3. Stream a hijacked session and catch the takeover.
    print("\nReplaying an account-takeover session (human -> bot at 50%)...")
    ato = generate_session("ato_demo", "account_takeover", n_events=700,
                           takeover_at=0.5, seed=42)
    mon = replay_session(model, ato, ema_alpha=0.15, step_up_threshold=0.6)
    takeover_t = float(ato["t_ms"][350]) / 1000
    if mon.stepped_up:
        up = mon.step_up_t_ms / 1000
        print(f"  takeover at      : {takeover_t:5.1f}s")
        print(f"  STEP-UP fired at : {up:5.1f}s  (latency {up - takeover_t:.1f}s)")
        print(f"  final risk state : {mon.risk:.2f}")
        print("  reasons          :")
        for r in mon.step_up_reasons:
            print(f"    - {r['what']} ({r['sigma']:.1f}sigma off the human baseline)")
    else:
        print("  takeover NOT detected")

    # 4. Re-export the edge model used by the in-browser demo (web/live.html).
    sc, clf = train_edge_model(feat)
    export_edge_model(sc, clf, "web/model.json")
    print("\nedge model written to web/model.json")
    print("Open demo.ipynb for the visual walkthrough, or serve web/live.html to play with it live.")


if __name__ == "__main__":
    main()
