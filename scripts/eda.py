"""Regenerate the exploratory figures and a small stats file used in docs/EDA.md.

    python scripts/eda.py            # or: make eda

Reads data/raw/*.csv (see data/README.md); writes docs/figures/*.png and eda_stats.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import load_raw
from src.data.schema import DATE_COL, ENTITY_COL, TARGET_COL
from src.utils.io import write_json

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE, ORANGE, GREY = "#3550d6", "#e8590c", "#8a90a0"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11.5,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    raw = load_raw(cfg)
    df = raw.train.copy()
    open_df = df[df["IsOpen"] == 1]

    # 1. Total daily demand
    daily = df.groupby(DATE_COL)[TARGET_COL].sum()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(daily.index, daily.values / 1e6, color=BLUE, lw=0.8)
    ax.set(title="Total daily orders across all stores", ylabel="Orders (millions)")
    save(fig, "01_total_daily_demand.png")

    # 2. Weekday pattern
    wk = open_df.groupby("Weekday")[TARGET_COL].mean()
    rate = df.groupby("Weekday")["IsOpen"].mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(names, wk.reindex(range(1, 8)).values, color=BLUE)
    axes[0].set(title="Average orders on open days, by weekday", ylabel="Orders per store-day")
    axes[1].bar(names, rate.reindex(range(1, 8)).values * 100, color=ORANGE)
    axes[1].set(title="Share of store-days that are open", ylabel="%")
    save(fig, "02_weekday_pattern.png")

    # 3. Target distribution
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(open_df[TARGET_COL].clip(upper=open_df[TARGET_COL].quantile(0.995)), bins=80, color=BLUE)
    ax.set(title="Distribution of daily orders on open days (clipped at the 99.5th percentile)", xlabel="Orders", ylabel="Store-days")
    save(fig, "03_target_distribution.png")

    # 4. Promotion effect
    promo = open_df.groupby("PromoActive")[TARGET_COL].mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(["No promotion", "Promotion"], promo.reindex([0, 1]).values, color=[GREY, ORANGE])
    ax.set(title="Average orders on open days", ylabel="Orders per store-day")
    save(fig, "04_promotion_effect.png")

    # 5. Store-level demand spread
    store_mean = open_df.groupby(ENTITY_COL)[TARGET_COL].mean()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(store_mean, bins=60, color=BLUE)
    ax.set(title="Average open-day demand per store", xlabel="Orders per open day", ylabel="Stores")
    save(fig, "05_store_demand_distribution.png")

    # 6. History coverage (stores with a gap)
    per_date = df.groupby(DATE_COL)[ENTITY_COL].nunique()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(per_date.index, per_date.values, color=ORANGE)
    ax.set(title="Number of stores with a record, by date", ylabel="Stores", ylim=(0, per_date.max() * 1.1))
    save(fig, "06_history_coverage.png")

    stats = {
        "rows": int(len(df)), "stores": int(df[ENTITY_COL].nunique()),
        "start": str(df[DATE_COL].min().date()), "end": str(df[DATE_COL].max().date()),
        "closed_share": float(1 - df["IsOpen"].mean()),
        "mean_open_day_orders": float(open_df[TARGET_COL].mean()),
        "median_open_day_orders": float(open_df[TARGET_COL].median()),
        "promo_uplift_ratio": float(promo[1] / promo[0]),
        "weekday_mean_open_day": {names[i - 1]: float(v) for i, v in wk.items()},
        "weekday_open_rate": {names[i - 1]: float(v) for i, v in rate.items()},
        "store_mean_p10_p50_p90": [float(store_mean.quantile(q)) for q in (0.1, 0.5, 0.9)],
        "min_stores_on_a_date": int(per_date.min()), "max_stores_on_a_date": int(per_date.max()),
        "skew_open_day_orders": float(open_df[TARGET_COL].skew()),
    }
    write_json(OUT / "eda_stats.json", stats)
    print(pd.Series(stats).to_string())


if __name__ == "__main__":
    main()
