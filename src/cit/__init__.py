"""Continuous Identity Trust — behavioural risk engine prototype.

A self-contained demo of continuous, zero-friction authentication: synthetic
behavioural telemetry -> Polars feature engineering -> ML risk scoring ->
step-up-auth feedback loop.
"""

from .edge import export_edge_model, train_edge_model
from .engine import RiskModel, SessionMonitor, replay_session, train_model
from .features import FEATURE_COLS, build_feature_table, windows_from_session
from .telemetry import generate_dataset, generate_session

__all__ = [
    "generate_session",
    "generate_dataset",
    "windows_from_session",
    "build_feature_table",
    "FEATURE_COLS",
    "train_model",
    "replay_session",
    "RiskModel",
    "SessionMonitor",
    "train_edge_model",
    "export_edge_model",
]
