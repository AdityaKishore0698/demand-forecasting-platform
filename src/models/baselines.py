"""Naive forecasting baselines used to put model error in context.

A forecast model is only meaningful relative to simple rules of thumb, so every
back-test reports these on exactly the same rows. All use only data up to the
window's origin (the last known day) - the same information the model has.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data.schema import DATE_COL, ENTITY_COL
from src.evaluation.metrics import apply_closed_hub_rule

BASELINE_LABELS = {
    "global_mean": "Global average",
    "hub_mean": "Store average",
    "last_value": "Last known day",
    "seasonal_naive": "Same weekday, last week",
    "rolling_7d_mean": "7-day rolling average",
}


def baseline_predictions(rows: pd.DataFrame, panel: pd.DataFrame, origin: pd.Timestamp) -> Dict[str, np.ndarray]:
    """``rows`` are the model rows of ONE window (single origin) incl. history features."""
    history = panel.loc[:origin]
    global_mean = float(np.nanmean(history.to_numpy()))

    hub_mean = rows["OV_expanding_mean"].to_numpy(dtype=float)
    last_value = rows["OV_lag1"].to_numpy(dtype=float)
    roll7 = rows["OV_rollmean7"].to_numpy(dtype=float)

    # most recent same-weekday observation on or before the origin
    offset = (origin.dayofweek - rows[DATE_COL].dt.dayofweek.to_numpy()) % 7
    lookup_dates = origin - pd.to_timedelta(offset, unit="D")
    date_idx = history.index.get_indexer(lookup_dates)
    hub_idx = history.columns.get_indexer(rows[ENTITY_COL])
    seasonal = np.full(len(rows), np.nan)
    ok = (date_idx >= 0) & (hub_idx >= 0)
    seasonal[ok] = history.to_numpy()[date_idx[ok], hub_idx[ok]]

    raw = {
        "global_mean": np.full(len(rows), global_mean),
        "hub_mean": hub_mean,
        "last_value": last_value,
        "seasonal_naive": seasonal,
        "rolling_7d_mean": roll7,
    }
    is_open = rows["IsOpen"].to_numpy()
    out = {}
    for name, pred in raw.items():
        pred = np.where(np.isnan(pred), global_mean, pred)   # e.g. lookup landed in a history gap
        out[name] = apply_closed_hub_rule(pred, is_open)
    return out
