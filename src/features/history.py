"""History-based features, anchored to a forecast ORIGIN (the last known day).

For every candidate origin date ``o`` these use only observations with date <= o.
Because a 42-day forecast cannot use "yesterday's value" for day 30 (that value is
itself unknown), all history features are frozen at the origin and the model is told
how far past the origin it is predicting (the horizon step ``h``).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _stack_named(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    long_frames = []
    for name, wide_df in frames.items():
        s = wide_df.stack(future_stack=True)
        s.name = name
        long_frames.append(s)
    out = pd.concat(long_frames, axis=1)
    out.index.names = ["Date", "HubID"]
    return out.reset_index()


def compute_origin_features(panel: pd.DataFrame, prefix: str, lag_days: List[int],
                            rolling_windows: List[int]) -> pd.DataFrame:
    """Lag / rolling / expanding statistics as of each date (treated as an origin).

    ``lag1`` is the value AT the origin itself (i.e. one day before the first
    forecast day); ``lagK`` is the value K-1 days before the origin.
    Rolling windows end at (and include) the origin; ``min_periods = max(3, w // 2)``
    yields NaN, rather than a noisy estimate, when a window is mostly empty.
    """
    feats: Dict[str, pd.DataFrame] = {}
    for k in lag_days:
        feats[f"{prefix}_lag{k}"] = panel.shift(k - 1)
    for w in rolling_windows:
        min_p = max(3, w // 2)
        feats[f"{prefix}_rollmean{w}"] = panel.rolling(window=w, min_periods=min_p).mean()
        feats[f"{prefix}_rollstd{w}"] = panel.rolling(window=w, min_periods=min_p).std()
    feats[f"{prefix}_expanding_mean"] = panel.expanding(min_periods=3).mean()
    return _stack_named(feats)


def compute_hub_weekday_origin_features(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """For each (origin, hub, weekday w): the hub's mean demand on weekday ``w`` using
    only occurrences of ``w`` on or before the origin.

    Masking the panel to weekday-``w`` rows and taking an expanding mean recomputes
    "mean of the w-days seen so far" at every date, so non-``w`` dates simply repeat
    the most recent value (nothing new entered) - exactly "as of the origin".
    The result is joined later on the TARGET row's own weekday, not the origin's.
    """
    weekday_of_date = pd.Series(panel.index.dayofweek + 1, index=panel.index)
    frames = []
    for w in range(1, 8):
        mask = pd.DataFrame(
            np.broadcast_to((weekday_of_date == w).values[:, None], panel.shape),
            index=panel.index, columns=panel.columns,
        )
        expanding_w_mean = panel.where(mask).expanding(min_periods=1).mean()
        s = expanding_w_mean.stack(future_stack=True)
        df_w = s.reset_index()
        df_w.columns = ["Origin", "HubID", f"{prefix}_hubweekday_mean"]
        df_w["Weekday"] = w
        frames.append(df_w)
    return pd.concat(frames, ignore_index=True)
