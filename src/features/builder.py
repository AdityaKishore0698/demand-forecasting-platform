"""Assemble the supervised (origin, horizon, hub) -> target-date dataset.

One row answers: "standing at the end of day ``Origin``, what will ``HubID`` sell
``h`` days later (on ``Date = Origin + h``)?"  The same function builds training,
back-test and live-inference rows, so there is no train/serve feature skew.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loaders import RawData
from src.data.schema import DAILY_ATTR_COLS, DATE_COL, ENTITY_COL, SCHEDULE_COLS, TARGET_COL
from src.features.history import compute_hub_weekday_origin_features, compute_origin_features
from src.features.metadata import add_hub_metadata_features
from src.features.panels import build_panel
from src.features.schedule import build_schedule_panels, compute_schedule_neighbor_features

FEATURE_PREFIX = "OV"  # "order volume" history features

CATEGORICAL_FEATURES = ["HubID", "Weekday", "HubFormat", "AssortmentTier", "LoyaltyProgram"]

# Columns present in the assembled frame that are NOT model inputs.
_NON_FEATURE_COLS = {
    "Origin", "Date", TARGET_COL,
    "CompetitorOpenSinceMonth", "CompetitorOpenSinceYear",
    "LoyaltyProgramSinceWeek", "LoyaltyProgramSinceYear", "LoyaltyProgramInterval",
}


def build_daily_attrs(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Per (hub, date) calendar/operational attributes (+ target where known).

    Deliberately excludes any column that exists only in the history file (e.g. a
    same-day activity counter): it would not be available for a future date.
    """
    hist = history[DAILY_ATTR_COLS + [TARGET_COL]].copy()
    fut = future[DAILY_ATTR_COLS].copy()
    fut[TARGET_COL] = np.nan
    return pd.concat([hist, fut], ignore_index=True)


_NEXT_COL = re.compile(r"_next(\d+)$")


def mask_unpublished_schedule(df: pd.DataFrame, schedule_horizon: int) -> pd.DataFrame:
    """Blank out "next-day" schedule features that would not be published at forecast time.

    The schedule is known only up to ``origin + schedule_horizon``. For a row with horizon step
    ``h`` the ``*_nextK`` feature describes day ``origin + h + K``, so it exists only when
    ``h + K <= schedule_horizon`` - i.e. it is missing for the last forecast day(s). This is applied
    identically in training, back-testing and serving so that all three see the same information.
    (``*_prevK`` features look backwards and are always known.)
    """
    df = df.copy()
    for col in df.columns:
        m = _NEXT_COL.search(col)
        if m:
            df.loc[df["h"] + int(m.group(1)) > schedule_horizon, col] = np.nan
    return df


def build_supervised_dataset(
    daily_attrs: pd.DataFrame,
    origin_feats: pd.DataFrame,
    hub_meta: pd.DataFrame,
    origins: Iterable,
    horizons: Iterable[int],
    require_label: bool,
    hubweekday_feats: Optional[pd.DataFrame] = None,
    schedule_neighbor_feats: Optional[pd.DataFrame] = None,
    schedule_horizon: Optional[int] = None,
) -> pd.DataFrame:
    hub_ids = daily_attrs[ENTITY_COL].unique()
    origin_arr = pd.to_datetime(pd.Index(list(origins)))

    grid = pd.MultiIndex.from_product(
        [origin_arr, list(horizons), hub_ids], names=["Origin", "h", ENTITY_COL]
    ).to_frame(index=False)
    grid[DATE_COL] = grid["Origin"] + pd.to_timedelta(grid["h"], unit="D")

    df = grid.merge(daily_attrs, on=[ENTITY_COL, DATE_COL], how="left")
    if require_label:
        # (hub, date) pairs inside a hub's history gap simply have no label.
        df = df[df[TARGET_COL].notna()].copy()

    df = df.merge(origin_feats.rename(columns={"Date": "Origin"}), on=["Origin", ENTITY_COL], how="left")
    if hubweekday_feats is not None:
        # joined on the TARGET's weekday (known for any future date), not the origin's
        df = df.merge(hubweekday_feats, on=["Origin", ENTITY_COL, "Weekday"], how="left")
    if schedule_neighbor_feats is not None:
        df = df.merge(schedule_neighbor_feats, on=[DATE_COL, ENTITY_COL], how="left")
        if schedule_horizon is not None:
            df = mask_unpublished_schedule(df, schedule_horizon)

    df = add_hub_metadata_features(df, hub_meta, date_col=DATE_COL)

    df["Month"] = df[DATE_COL].dt.month
    df["DayOfMonth"] = df[DATE_COL].dt.day
    df["WeekOfYear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["Weekday"].isin([6, 7]).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


@dataclass
class FeatureTables:
    """All precomputed lookup tables needed to assemble model rows for any origin."""
    daily_attrs: pd.DataFrame
    origin_feats: pd.DataFrame
    hubweekday_feats: pd.DataFrame
    schedule_neighbor_feats: pd.DataFrame
    schedule: pd.DataFrame            # long schedule table (history + future)
    hub_meta: pd.DataFrame
    target_panel: pd.DataFrame        # date x hub matrix of the target (history only)
    schedule_horizon: Optional[int] = None   # days of schedule published ahead of an origin

    def make_dataset(self, origins, horizons, require_label: bool) -> pd.DataFrame:
        return build_supervised_dataset(
            self.daily_attrs, self.origin_feats, self.hub_meta, origins, horizons,
            require_label=require_label,
            hubweekday_feats=self.hubweekday_feats,
            schedule_neighbor_feats=self.schedule_neighbor_feats,
            schedule_horizon=self.schedule_horizon,
        )


def build_feature_tables(raw: RawData, cfg: Config) -> FeatureTables:
    panel = build_panel(raw.train, TARGET_COL)
    origin_feats = compute_origin_features(
        panel, FEATURE_PREFIX, cfg.features.lag_days, cfg.features.rolling_windows
    )
    hubweekday = compute_hub_weekday_origin_features(panel, FEATURE_PREFIX)
    keys = [ENTITY_COL, DATE_COL] + SCHEDULE_COLS
    schedule = pd.concat([raw.train[keys], raw.future[keys]], ignore_index=True)
    neighbors = compute_schedule_neighbor_features(
        build_schedule_panels(schedule), cfg.features.schedule_shifts
    )
    return FeatureTables(
        daily_attrs=build_daily_attrs(raw.train, raw.future),
        origin_feats=origin_feats,
        hubweekday_feats=hubweekday,
        schedule_neighbor_feats=neighbors,
        schedule=schedule,
        hub_meta=raw.hub_meta,
        target_panel=panel,
        schedule_horizon=cfg.forecast.horizon_days,
    )
