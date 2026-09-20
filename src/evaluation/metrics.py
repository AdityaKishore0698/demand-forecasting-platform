"""Forecast accuracy metrics.

Definitions (also surfaced in the dashboard):
* RMSLE - all hub-days. Log-scale error; the metric the model is selected on.
* MAE, RMSE, WAPE, RMSPE, MAPE - OPEN hub-days only. Closed days are deterministic
  zeros (predicted exactly), so including them would dilute the error, and
  percentage metrics are undefined for a zero actual.
* RMSPE / MAPE additionally skip the (very few) open days whose actual is 0.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def apply_closed_hub_rule(pred, is_open) -> np.ndarray:
    """A closed hub sells nothing. ``IsOpen`` is known in advance, so forcing the
    forecast to exactly 0 on closed days is safe (no leakage) and removes error."""
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    return np.where(np.asarray(is_open) == 0, 0.0, pred)


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: total absolute error / total actual."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_pred - y_true)) / denom) if denom > 0 else float("nan")


def rmspe(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = y_true > 0
    return float(np.sqrt(np.mean(((y_true[m] - y_pred[m]) / y_true[m]) ** 2))) if m.any() else float("nan")


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = y_true > 0
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m]))) if m.any() else float("nan")


def regression_report(y_true, y_pred, is_open) -> Dict[str, float]:
    y_true, y_pred, is_open = (np.asarray(a) for a in (y_true, y_pred, is_open))
    open_mask = is_open == 1
    yt, yp = y_true[open_mask].astype(float), y_pred[open_mask].astype(float)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "wape": wape(yt, yp),
        "rmspe": rmspe(yt, yp),
        "mape": mape(yt, yp),
        "bias": float(yp.sum() / yt.sum() - 1.0) if yt.sum() > 0 else float("nan"),
        "n_rows": int(len(y_true)),
        "n_open_rows": int(open_mask.sum()),
    }
