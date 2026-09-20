"""Chronological back-testing (rolling-origin validation).

A random train/validation split would let the model interpolate between a hub's
neighbouring days, which is nothing like real forecasting (extrapolating forward).
Each window here mimics the real task exactly: fit only on data whose *target dates*
lie before the window, then forecast the next ``horizon`` days from the window's
origin and compare with what actually happened.

::

    history ----------------------|---------- window (H days) --------|
    training targets end here ->  | cutoff = origin (last known day)  | scored against actuals
    (targets <= cutoff - H)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loaders import RawData
from src.data.schema import DATE_COL, ENTITY_COL, TARGET_COL
from src.evaluation.metrics import apply_closed_hub_rule, regression_report, rmsle
from src.features.builder import FeatureTables
from src.models.baselines import baseline_predictions
from src.models.dispatch import AnyModel, fit as fit_model, is_ensemble, predict as predict_demand
from src.models.ensemble import predict_members


class LeakageError(AssertionError):
    """A training example's target date reaches into the period being predicted."""


@dataclass
class BacktestWindow:
    name: str
    cutoff: pd.Timestamp            # origin: the last day whose data the model may see
    horizon: int
    max_train_origin: pd.Timestamp  # last origin allowed in training (targets stay < start)

    @property
    def start(self) -> pd.Timestamp:
        return self.cutoff + pd.Timedelta(days=1)

    @property
    def end(self) -> pd.Timestamp:
        return self.cutoff + pd.Timedelta(days=self.horizon)

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "cutoff": self.cutoff.date().isoformat(),
            "start": self.start.date().isoformat(), "end": self.end.date().isoformat(),
            "horizon": self.horizon,
        }


def make_backtest_windows(train_max: pd.Timestamp, horizon: int, n_windows: int) -> List[BacktestWindow]:
    windows = []
    for i in range(n_windows):
        cutoff = train_max - pd.Timedelta(days=horizon * (i + 1))
        name = "primary" if i == 0 else f"earlier_{i}"
        windows.append(BacktestWindow(name, cutoff, horizon, cutoff - pd.Timedelta(days=horizon)))
    return windows


def training_origins(tables: FeatureTables, max_origin: pd.Timestamp, cfg: Config) -> pd.DatetimeIndex:
    first = tables.target_panel.index.min() + pd.Timedelta(days=cfg.forecast.min_history_days)
    return pd.date_range(first, max_origin, freq=f"{cfg.forecast.origin_stride_days}D")


def assert_no_temporal_leakage(train_df: pd.DataFrame, window: BacktestWindow) -> None:
    """Every training example must have Date = Origin + h with h >= 1, and its target
    date must fall strictly before the window being predicted."""
    if not (train_df[DATE_COL] > train_df["Origin"]).all():
        raise LeakageError("a training row has target date <= its origin")
    latest = train_df[DATE_COL].max()
    if latest >= window.start:
        raise LeakageError(
            f"training target dates reach {latest.date()} which is inside/after the "
            f"validation window starting {window.start.date()}"
        )


@dataclass
class BacktestResult:
    window: BacktestWindow
    model: AnyModel
    rows: pd.DataFrame              # window rows + prediction + baselines + actual
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, Dict[str, float]]
    n_train_rows: int
    member_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)   # ensemble members, same rows


def run_backtest_window(window: BacktestWindow, tables: FeatureTables, cfg: Config,
                        keep_train_sample: int = 0, seed: int = 0):
    """Train on data before ``window`` and score the window. Returns (result, train_sample)."""
    horizons = range(1, window.horizon + 1)
    origins = training_origins(tables, window.max_train_origin, cfg)
    train_df = tables.make_dataset(origins, horizons, require_label=True)
    assert_no_temporal_leakage(train_df, window)

    model = fit_model(train_df, cfg)
    n_train = len(train_df)

    train_sample = None
    if keep_train_sample:
        train_sample = train_df.sample(min(keep_train_sample, n_train), random_state=seed)
    del train_df

    rows = tables.make_dataset([window.cutoff], horizons, require_label=True)
    rows["actual"] = rows[TARGET_COL].to_numpy(dtype=float)
    rows["predicted"] = apply_closed_hub_rule(predict_demand(model, rows), rows["IsOpen"])
    member_metrics: Dict[str, Dict[str, float]] = {}
    if is_ensemble(model):
        # per-member scores on the very same rows (members are predicted once more; cheap relative to fitting)
        for name, p in predict_members(model, rows).items():
            member_metrics[name] = regression_report(rows["actual"], apply_closed_hub_rule(p, rows["IsOpen"]), rows["IsOpen"])
            rows[f"pred_{name}"] = apply_closed_hub_rule(p, rows["IsOpen"])

    baselines = baseline_predictions(rows, tables.target_panel, window.cutoff)
    for name, pred in baselines.items():
        rows[f"baseline_{name}"] = pred

    is_open = rows["IsOpen"].to_numpy()
    metrics = regression_report(rows["actual"], rows["predicted"], is_open)
    base_metrics = {n: regression_report(rows["actual"], p, is_open) for n, p in baselines.items()}
    result = BacktestResult(window, model, rows, metrics, base_metrics, n_train, member_metrics)
    return result, train_sample


def training_performance(model: AnyModel, train_sample: pd.DataFrame) -> Dict[str, float]:
    """In-sample accuracy on a random sample of training rows (for the train/val gap)."""
    pred = apply_closed_hub_rule(predict_demand(model, train_sample), train_sample["IsOpen"])
    return regression_report(train_sample[TARGET_COL], pred, train_sample["IsOpen"])
