"""Slice a back-test window's errors (by horizon, weekday, hub size, error size)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmsle, wape


def _group_report(df: pd.DataFrame, key: str, order: List[str]) -> List[Dict]:
    out = []
    for group in order:
        g = df[df[key] == group]
        if g.empty:
            continue
        o = g[g["IsOpen"] == 1]
        out.append({
            "group": group,
            "n_open_rows": int(len(o)),
            "rmsle": rmsle(g["actual"], g["predicted"]),
            "wape": wape(o["actual"], o["predicted"]) if len(o) else float("nan"),
            "mae": mae(o["actual"], o["predicted"]) if len(o) else float("nan"),
        })
    return out


def horizon_bucket_labels(h: pd.Series, horizon: int, size: int = 7) -> pd.Series:
    start = ((h - 1) // size) * size + 1
    end = np.minimum(start + size - 1, horizon)
    return start.astype(str) + "-" + end.astype(str)


def error_breakdowns(rows: pd.DataFrame, hub_avg_demand: pd.Series, horizon: int) -> Dict:
    df = rows.copy()
    df["horizon_bucket"] = horizon_bucket_labels(df["h"], horizon)
    weekday_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    df["weekday_name"] = df["Weekday"].map(weekday_names)

    # hub-size terciles from history only (no look-ahead)
    tiers = pd.qcut(hub_avg_demand.rank(method="first"), 3, labels=["Low volume", "Mid volume", "High volume"])
    df["hub_tier"] = df["HubID"].map(tiers.astype(str))

    buckets = sorted(df["horizon_bucket"].unique(), key=lambda s: int(s.split("-")[0]))
    return {
        "by_horizon": _group_report(df, "horizon_bucket", buckets),
        "by_weekday": _group_report(df, "weekday_name", list(weekday_names.values())),
        "by_hub_tier": _group_report(df, "hub_tier", ["Low volume", "Mid volume", "High volume"]),
        "by_promo": _group_report(
            df.assign(promo=np.where(df["PromoActive"] == 1, "Promotion day", "Regular day")),
            "promo", ["Regular day", "Promotion day"]),
    }


def error_distribution(rows: pd.DataFrame, limit: float = 0.5, step: float = 0.05) -> Dict:
    """Histogram of signed percentage error, (pred - actual) / actual, on open days."""
    o = rows[(rows["IsOpen"] == 1) & (rows["actual"] > 0)]
    pct = ((o["predicted"] - o["actual"]) / o["actual"]).to_numpy()
    edges = np.round(np.arange(-limit, limit + step / 2, step), 4)
    counts, _ = np.histogram(np.clip(pct, -limit + 1e-9, limit - 1e-9), bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "n": int(len(pct)),
        "clipped_low": int((pct < -limit).sum()),
        "clipped_high": int((pct > limit).sum()),
        "median_pct_error": float(np.median(pct)),
        "p10": float(np.quantile(pct, 0.10)),
        "p90": float(np.quantile(pct, 0.90)),
        "share_within_10pct": float((np.abs(pct) <= 0.10).mean()),
        "share_within_20pct": float((np.abs(pct) <= 0.20).mean()),
    }
