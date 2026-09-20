"""Small deterministic synthetic dataset with the same schema as the real one.

It has the properties the pipeline must handle: closed-on-Sunday hubs, promo uplift,
zero demand on closed days, a hub with a missing block of history, NaN hub metadata,
and a train-only column that must never be used as a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import RawData

WEEKDAY_FACTOR = {1: 1.15, 2: 1.0, 3: 0.95, 4: 0.97, 5: 1.05, 6: 0.85, 7: 1.10}


def make_synthetic_raw(n_hubs: int = 6, n_train_days: int = 330, horizon: int = 42, seed: int = 0) -> RawData:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2014-01-01", periods=n_train_days + horizon, freq="D")
    hubs = np.arange(1, n_hubs + 1)
    grid = pd.MultiIndex.from_product([dates, hubs], names=["Date", "HubID"]).to_frame(index=False)

    grid["Weekday"] = grid["Date"].dt.dayofweek + 1
    day_idx = (grid["Date"] - dates[0]).dt.days
    # every hub closed on Sundays except hub 1
    grid["IsOpen"] = np.where((grid["Weekday"] == 7) & (grid["HubID"] != 1), 0, 1)
    grid["PromoActive"] = (((day_idx + grid["HubID"]) // 7) % 2 == 0).astype(int) * (grid["Weekday"] <= 5)
    grid["RegionalHoliday"] = np.where(day_idx.isin([120, 121, 250]), 1, 0)
    grid["SchoolClosureFlag"] = np.where((day_idx >= 180) & (day_idx < 215), 1, 0)

    base = pd.Series(rng.uniform(3000, 9000, n_hubs), index=hubs)
    demand = (
        grid["HubID"].map(base)
        * grid["Weekday"].map(WEEKDAY_FACTOR)
        * (1 + 0.35 * grid["PromoActive"])
        * (1 - 0.15 * grid["SchoolClosureFlag"])
        * rng.lognormal(0.0, 0.08, len(grid))
    )
    grid["OrderVolume"] = np.round(demand * grid["IsOpen"]).astype(int)
    grid["AppSessions"] = np.round(grid["OrderVolume"] * 0.1).astype(int)   # train-only column

    train_end = dates[n_train_days - 1]
    train = grid[grid["Date"] <= train_end].copy()
    # hub 3: 40-day hole in the history (rows simply absent)
    hole = (train["HubID"] == 3) & (day_idx[train.index] >= 100) & (day_idx[train.index] < 140)
    train = train[~hole].reset_index(drop=True)

    future = grid[grid["Date"] > train_end].drop(columns=["OrderVolume", "AppSessions"]).reset_index(drop=True)
    future.insert(0, "Id", np.arange(1, len(future) + 1))

    hub_meta = pd.DataFrame({
        "HubID": hubs,
        "HubFormat": [(i % 4) + 1 for i in range(n_hubs)],
        "AssortmentTier": [(i % 3) + 1 for i in range(n_hubs)],
        "CompetitorDistance": [1200.0, 600.0, np.nan, 5400.0, 300.0, 9000.0][:n_hubs],
        "CompetitorOpenSinceMonth": [9.0, np.nan, np.nan, 3.0, 11.0, np.nan][:n_hubs],
        "CompetitorOpenSinceYear": [2008.0, np.nan, np.nan, 2010.0, 2012.0, np.nan][:n_hubs],
        "LoyaltyProgram": [0, 1, 1, 0, 1, 0][:n_hubs],
        "LoyaltyProgramSinceWeek": [np.nan, 13.0, 14.0, np.nan, 5.0, np.nan][:n_hubs],
        "LoyaltyProgramSinceYear": [np.nan, 2010.0, 2011.0, np.nan, 2013.0, np.nan][:n_hubs],
        "LoyaltyProgramInterval": [np.nan, "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", np.nan, "Mar,Jun,Sept,Dec", np.nan][:n_hubs],
    })
    return RawData(train=train, future=future, hub_meta=hub_meta)
