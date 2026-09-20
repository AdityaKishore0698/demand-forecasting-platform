"""Inference service: load the trained artifacts once, answer requests cheaply.

Nothing here trains. A forecast request:

1. takes the hub's *known future schedule* (open / promo / holiday / school flags),
2. rebuilds the model's feature row for each horizon step with the SAME code used in
   training (``build_supervised_dataset``) from a compact snapshot of history features
   frozen at the last known day,
3. runs the LightGBM model and applies the closed-hub rule.

It never invents inputs: horizons beyond the window whose schedule is known are refused.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.schema import DATE_COL, ENTITY_COL, SCHEDULE_COLS, TARGET_COL
from src.evaluation.metrics import apply_closed_hub_rule, regression_report, wape
from src.features.builder import build_supervised_dataset
from src.features.catalog import describe_feature
from src.features.schedule import build_schedule_panels, compute_schedule_neighbor_features
from src.forecasting.artifacts import ArtifactPaths
from src.models.lightgbm_model import TrainedModel, load_model, predict_contributions, predict_raw
from src.utils.io import read_json

WEEKDAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


class ArtifactsNotFound(FileNotFoundError):
    """The trained-model bundle is missing (run the training pipeline first)."""


class StoreNotFound(KeyError):
    pass


class ForecastRequestError(ValueError):
    pass


def _none_if_nan(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


class ForecastService:
    CACHE_SIZE = 256

    def __init__(self, paths: ArtifactPaths):
        missing = paths.missing()
        if missing:
            raise ArtifactsNotFound(
                "Model artifacts not found: " + ", ".join(str(p.relative_to(paths.root)) for p in missing)
                + ". Train first with `make train` (python -m src.forecasting.pipeline)."
            )
        self.paths = paths
        self.card: Dict[str, Any] = read_json(paths.model_card)
        self.metrics: Dict[str, Any] = read_json(paths.metrics)
        self.importance: Dict[str, Any] = read_json(paths.importance)
        self.dataset_summary: Dict[str, Any] = read_json(paths.dataset_summary)

        self.model: TrainedModel = load_model(
            paths.model_file, self.card["categories"], self.card["params"], self.card["n_estimators"]
        )
        self.origin = pd.Timestamp(self.card["origin_date"])
        self.horizon_days = int(self.card["horizon_days"])
        self.model_version = self.card["model_version"]
        self.data_label = self.dataset_summary.get("data_label") or self.card.get("data_label")

        self.hub_meta = pd.read_csv(paths.hub_metadata)
        self.origin_feats = pd.read_csv(paths.origin_features, parse_dates=["Date"])
        self.hubweekday_feats = pd.read_csv(paths.hubweekday_features, parse_dates=["Origin"])
        schedule = pd.read_csv(paths.schedule, parse_dates=[DATE_COL])
        self.neighbor_feats = compute_schedule_neighbor_features(
            build_schedule_panels(schedule), self.card["schedule_shifts"]
        )
        future = schedule[schedule[DATE_COL] > self.origin]
        self._future_by_hub = {h: g.sort_values(DATE_COL).reset_index(drop=True) for h, g in future.groupby(ENTITY_COL)}

        history = pd.read_csv(paths.history, parse_dates=[DATE_COL])
        self._hist_by_hub = {
            h: g.sort_values(DATE_COL).reset_index(drop=True) for h, g in history.groupby(ENTITY_COL)
        }
        self.backtest_df = pd.read_csv(paths.backtest_predictions, parse_dates=[DATE_COL])
        self._stores = self._build_store_table(history)

        self._cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def load(cls, artifact_dir: Path) -> "ForecastService":
        return cls(ArtifactPaths(Path(artifact_dir)))

    # ------------------------------------------------------------------ stores
    def _build_store_table(self, history: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        open_days = history[history["IsOpen"] == 1]
        avg = open_days.groupby(ENTITY_COL)[TARGET_COL].mean()
        n_days = history.groupby(ENTITY_COL).size()
        calendar_days = (history[DATE_COL].max() - history[DATE_COL].min()).days + 1
        rank = avg.rank(ascending=False, method="min").astype(int)
        pct = avg.rank(pct=True) * 100.0

        bt_open = self.backtest_df[self.backtest_df["IsOpen"] == 1]
        bt_wape = bt_open.groupby(ENTITY_COL).apply(
            lambda g: wape(g["actual"], g["predicted"]), include_groups=False
        ) if len(bt_open) else pd.Series(dtype=float)

        meta = self.hub_meta.set_index(ENTITY_COL)
        stores: Dict[int, Dict[str, Any]] = {}
        for hub in sorted(meta.index):
            if hub not in avg.index:
                continue
            m = meta.loc[hub]
            stores[int(hub)] = {
                "store_id": int(hub),
                "hub_format": _none_if_nan(m["HubFormat"]),
                "assortment_tier": _none_if_nan(m["AssortmentTier"]),
                "competitor_distance": _none_if_nan(m["CompetitorDistance"]),
                "loyalty_program": bool(m["LoyaltyProgram"] == 1),
                "avg_open_day_demand": round(float(avg[hub]), 1),
                "history_days": int(n_days[hub]),
                "has_history_gap": bool(n_days[hub] < calendar_days),
                "demand_rank": int(rank[hub]),
                "demand_percentile": round(float(pct[hub]), 1),
                "backtest_wape": _none_if_nan(float(bt_wape[hub])) if hub in bt_wape.index else None,
            }
        return stores

    def list_stores(self) -> List[Dict[str, Any]]:
        return list(self._stores.values())

    def get_store(self, store_id: int) -> Dict[str, Any]:
        try:
            return self._stores[int(store_id)]
        except KeyError:
            raise StoreNotFound(f"Unknown store_id {store_id}") from None

    # -------------------------------------------------------------- historical
    def historical(self, store_id: int, days: int = 180) -> Dict[str, Any]:
        self.get_store(store_id)
        if days < 1:
            raise ForecastRequestError("days must be >= 1")
        df = self._hist_by_hub[int(store_id)]
        first, last = df[DATE_COL].min(), self.origin
        start = max(first, last - pd.Timedelta(days=days - 1))
        cal = pd.date_range(start, last, freq="D")
        d = df.set_index(DATE_COL).reindex(cal)
        recorded = d[TARGET_COL].notna()
        open_mask = recorded & (d["IsOpen"] == 1)
        points = [
            {
                "date": ts.date(),
                "demand": None if pd.isna(r[TARGET_COL]) else int(r[TARGET_COL]),
                "is_open": None if pd.isna(r["IsOpen"]) else bool(r["IsOpen"]),
                "promo_active": None if pd.isna(r["PromoActive"]) else bool(r["PromoActive"]),
            }
            for ts, r in d.iterrows()
        ]
        return {
            "store_id": int(store_id),
            "start": cal[0].date(),
            "end": cal[-1].date(),
            "points": points,
            "summary": {
                "recorded_days": int(recorded.sum()),
                "missing_days": int((~recorded).sum()),
                "open_days": int(open_mask.sum()),
                "avg_open_day_demand": round(float(d.loc[open_mask, TARGET_COL].mean()), 1) if open_mask.any() else None,
                "first_history_date": first.date(),
                "last_history_date": last.date(),
            },
        }

    # ---------------------------------------------------------------- features
    def _features(self, store_id: int, horizons: List[int]) -> pd.DataFrame:
        daily = self._future_by_hub[int(store_id)].copy()
        daily[TARGET_COL] = np.nan
        rows = build_supervised_dataset(
            daily, self.origin_feats, self.hub_meta, [self.origin], horizons, require_label=False,
            hubweekday_feats=self.hubweekday_feats, schedule_neighbor_feats=self.neighbor_feats,
            schedule_horizon=self.horizon_days,
        )
        return rows.sort_values("h").reset_index(drop=True)

    # ---------------------------------------------------------------- forecast
    def forecast(self, store_id: int, horizon: int) -> Dict[str, Any]:
        self.get_store(store_id)
        if not 1 <= horizon <= self.horizon_days:
            raise ForecastRequestError(
                f"horizon must be between 1 and {self.horizon_days} days (the window whose schedule is known)"
            )
        key = (int(store_id), int(horizon))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        rows = self._features(int(store_id), list(range(1, horizon + 1)))
        pred = apply_closed_hub_rule(predict_raw(self.model, rows), rows["IsOpen"])

        forecast = []
        for (_, r), p in zip(rows.iterrows(), pred):
            forecast.append({
                "date": r[DATE_COL].date(),
                "weekday": WEEKDAY_NAMES[int(r["Weekday"])],
                "horizon_step": int(r["h"]),
                "predicted_demand": round(float(p), 1),
                "is_open": bool(r["IsOpen"] == 1),
                "promo_active": bool(r["PromoActive"] == 1),
                "holiday": bool(r["RegionalHoliday"] > 0),
                "school_closure": bool(r["SchoolClosureFlag"] == 1),
            })
        open_pts = [f for f in forecast if f["is_open"]]
        peak = max(open_pts, key=lambda f: f["predicted_demand"]) if open_pts else None
        low = min(open_pts, key=lambda f: f["predicted_demand"]) if open_pts else None
        result = {
            "store_id": int(store_id),
            "horizon": int(horizon),
            "origin_date": self.origin.date(),
            "model_version": self.model_version,
            "forecast": forecast,
            "summary": {
                "total_predicted": round(float(pred.sum()), 1),
                "mean_open_day": round(float(np.mean([f["predicted_demand"] for f in open_pts])), 1) if open_pts else None,
                "peak": {"date": peak["date"], "value": peak["predicted_demand"]} if peak else None,
                "lowest_open_day": {"date": low["date"], "value": low["predicted_demand"]} if low else None,
                "open_days": len(open_pts),
                "closed_days": len(forecast) - len(open_pts),
                "promo_days": int(sum(f["promo_active"] for f in forecast)),
            },
            "notes": [
                "Closed days are forecast as exactly 0 (a closed store has no orders).",
                "Point forecasts only - the model does not produce prediction intervals.",
                f"Uses history up to {self.origin.date().isoformat()} and the planned open/promo/holiday schedule.",
            ],
        }
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    # ----------------------------------------------------------------- explain
    def explain(self, store_id: int, date, top_k: int = 6) -> Dict[str, Any]:
        self.get_store(store_id)
        day = pd.Timestamp(date)
        h = (day - self.origin).days
        if not 1 <= h <= self.horizon_days:
            raise ForecastRequestError(
                f"date must be within the forecast window ({(self.origin + pd.Timedelta(days=1)).date()} "
                f"to {(self.origin + pd.Timedelta(days=self.horizon_days)).date()})"
            )
        rows = self._features(int(store_id), [h])
        contrib = predict_contributions(self.model, rows)[0]
        bias, parts = float(contrib[-1]), contrib[:-1]
        order = np.argsort(-np.abs(parts))
        names = self.model.feature_names
        row = rows.iloc[0]

        def item(i: int) -> Dict[str, Any]:
            meta = describe_feature(names[i])
            return {
                "feature": names[i], "label": meta["label"], "group": meta["group_label"],
                "value": _none_if_nan(row[names[i]]),
                "effect_pct": round((float(np.exp(parts[i])) - 1.0) * 100.0, 1),
            }

        top = [item(int(i)) for i in order[:top_k]]
        rest = order[top_k:]
        model_pred = float(np.exp(bias + parts.sum()))
        is_open = bool(row["IsOpen"] == 1)
        return {
            "store_id": int(store_id),
            "date": day.date(),
            "is_open": is_open,
            "baseline_demand": round(float(np.exp(bias)), 1),
            "model_prediction": round(model_pred, 1),
            "predicted_demand": round(model_pred, 1) if is_open else 0.0,
            "drivers": top,
            "other_features_effect_pct": round((float(np.exp(parts[rest].sum())) - 1.0) * 100.0, 1),
            "note": "Effects are multiplicative on expected demand (TreeSHAP, log link). "
                    "They show how each feature moves this forecast relative to the model's starting point (its typical output over all training days).",
        }

    # ------------------------------------------------------------- reporting
    def backtest(self, store_id: Optional[int] = None) -> Dict[str, Any]:
        bt = self.backtest_df
        if store_id is None:
            scope = "All stores (daily total)"
            grouped = bt.groupby(DATE_COL)[["actual", "predicted", "baseline"]].sum().reset_index()
        else:
            self.get_store(store_id)
            bt = bt[bt[ENTITY_COL] == int(store_id)]
            scope = f"Hub {int(store_id)}"
            grouped = bt[[DATE_COL, "actual", "predicted", "baseline"]].sort_values(DATE_COL)
        metrics = regression_report(bt["actual"], bt["predicted"], bt["IsOpen"])
        base = regression_report(bt["actual"], bt["baseline"], bt["IsOpen"])
        return {
            "scope": scope,
            "store_id": None if store_id is None else int(store_id),
            "window": self.metrics["primary_window"],
            "points": [
                {"date": r[DATE_COL].date(), "actual": float(r["actual"]),
                 "predicted": float(r["predicted"]), "baseline": float(r["baseline"])}
                for _, r in grouped.iterrows()
            ],
            "metrics": metrics,
            "baseline_metrics": base,
            "baseline_label": self.metrics["baselines"][self.metrics["best_baseline"]]["label"],
        }

    def model_info(self) -> Dict[str, Any]:
        card = self.card
        feats = [describe_feature(f) for f in card["feature_names"]]
        return {
            "model_version": card["model_version"],
            "algorithm": card["algorithm"],
            "created_at": card["created_at"],
            "seed": card["seed"],
            "n_estimators": card["n_estimators"],
            "params": card["params"],
            "n_features": len(feats),
            "features": feats,
            "horizon_days": card["horizon_days"],
            "origin_date": card["origin_date"],
            "forecast_window": card["forecast_window"],
            "training": card["training"],
            "validation_summary": card["validation_summary"],
            "serving_note": "Forecasts are generated on request by rebuilding features from a stored snapshot "
                            "and running the loaded model. No training happens at request time.",
        }
