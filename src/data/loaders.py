"""Raw data loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import (
    DATE_COL, ENTITY_COL, FUTURE_REQUIRED, HUB_META_REQUIRED, TARGET_COL, TRAIN_REQUIRED,
)


class DataValidationError(ValueError):
    """Raised when the raw data violates an invariant the pipeline depends on."""


@dataclass
class RawData:
    train: pd.DataFrame       # history WITH target
    future: pd.DataFrame      # the calendar/schedule to forecast (NO target)
    hub_meta: pd.DataFrame    # static per-hub attributes

    @property
    def train_max(self) -> pd.Timestamp:
        return self.train[DATE_COL].max()

    @property
    def train_min(self) -> pd.Timestamp:
        return self.train[DATE_COL].min()


def load_raw(cfg: Config) -> RawData:
    """Read the three source files from ``cfg.raw_dir`` and validate them."""
    raw_dir = cfg.raw_dir
    paths = {
        "train": raw_dir / cfg.data.train_file,
        "future": raw_dir / cfg.data.test_file,
        "hub_meta": raw_dir / cfg.data.hub_metadata_file,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw data files not found: " + ", ".join(missing) +
            ". Place the dataset in data/raw/ (see data/README.md)."
        )
    raw = RawData(
        train=pd.read_csv(paths["train"], parse_dates=[DATE_COL]),
        future=pd.read_csv(paths["future"], parse_dates=[DATE_COL]),
        hub_meta=pd.read_csv(paths["hub_meta"]),
    )
    validate_raw(raw, cfg)
    return raw


def validate_raw(raw: RawData, cfg: Config) -> None:
    """Fail fast if an invariant the leakage-safety argument relies on is broken."""
    for name, df, required in (
        ("train", raw.train, TRAIN_REQUIRED),
        ("future", raw.future, FUTURE_REQUIRED),
        ("hub_meta", raw.hub_meta, HUB_META_REQUIRED),
    ):
        absent = [c for c in required if c not in df.columns]
        if absent:
            raise DataValidationError(f"{name}: missing required columns {absent}")

    if raw.train.duplicated([ENTITY_COL, DATE_COL]).any():
        raise DataValidationError("train: duplicate (hub, date) rows")
    if raw.hub_meta[ENTITY_COL].duplicated().any():
        raise DataValidationError("hub_meta: duplicate hub ids")
    if (raw.train[TARGET_COL] < 0).any():
        raise DataValidationError("train: negative target values")

    # The forecast window must start strictly after the last known day, otherwise
    # "future" schedule rows could overlap history and features would not be causal.
    if raw.future[DATE_COL].min() <= raw.train_max:
        raise DataValidationError("future calendar overlaps the training period")

    horizon = (raw.future[DATE_COL].max() - raw.train_max).days
    if horizon < cfg.forecast.horizon_days:
        raise DataValidationError(
            f"future calendar covers {horizon} days but forecast.horizon_days is "
            f"{cfg.forecast.horizon_days}"
        )

    hubs_train = set(raw.train[ENTITY_COL].unique())
    hubs_future = set(raw.future[ENTITY_COL].unique())
    hubs_meta = set(raw.hub_meta[ENTITY_COL].unique())
    if not hubs_future <= hubs_train:
        raise DataValidationError(
            "future calendar contains hubs with no history (cold-start hubs are not supported)"
        )
    if not hubs_train <= hubs_meta:
        raise DataValidationError("some hubs in train have no metadata row")


def summarize_dataset(raw: RawData) -> Dict[str, Any]:
    """Descriptive facts about the dataset (used by the About page and the docs)."""
    train = raw.train
    open_days = train[train["IsOpen"] == 1]
    per_hub = train.groupby(ENTITY_COL).size()
    calendar_days = (raw.train_max - raw.train_min).days + 1
    weekday_open_rate = train.groupby("Weekday")["IsOpen"].mean()
    weekday_mean_demand = open_days.groupby("Weekday")[TARGET_COL].mean()
    promo_means = open_days.groupby("PromoActive")[TARGET_COL].mean()
    promo_uplift = (
        float(promo_means.get(1) / promo_means.get(0))
        if 0 in promo_means.index and 1 in promo_means.index else None
    )
    hub_means = open_days.groupby(ENTITY_COL)[TARGET_COL].mean()

    return {
        "train_rows": int(len(train)),
        "future_rows": int(len(raw.future)),
        "n_hubs": int(train[ENTITY_COL].nunique()),
        "train_start": raw.train_min.date().isoformat(),
        "train_end": raw.train_max.date().isoformat(),
        "forecast_start": raw.future[DATE_COL].min().date().isoformat(),
        "forecast_end": raw.future[DATE_COL].max().date().isoformat(),
        "calendar_days": int(calendar_days),
        "hubs_with_history_gaps": int((per_hub < calendar_days).sum()),
        "zero_demand_share": float((train[TARGET_COL] == 0).mean()),
        "closed_day_share": float((train["IsOpen"] == 0).mean()),
        "closed_days_always_zero": bool((train.loc[train["IsOpen"] == 0, TARGET_COL] == 0).all()),
        "open_days_with_zero_demand": int(((train["IsOpen"] == 1) & (train[TARGET_COL] == 0)).sum()),
        "mean_open_day_demand": float(open_days[TARGET_COL].mean()),
        "hub_demand_cv": float(hub_means.std() / hub_means.mean()),
        "promo_uplift_ratio": promo_uplift,
        "weekday_open_rate": {int(k): float(v) for k, v in weekday_open_rate.items()},
        "weekday_mean_demand": {int(k): float(v) for k, v in weekday_mean_demand.items()},
        "future_promo_rate": float(raw.future["PromoActive"].mean()),
        "future_holiday_rate": float((raw.future["RegionalHoliday"] > 0).mean()),
        "future_school_closure_rate": float(raw.future["SchoolClosureFlag"].mean()),
        "train_only_columns_excluded": sorted(set(train.columns) - set(raw.future.columns) - {TARGET_COL}),
    }
