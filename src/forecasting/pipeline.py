"""End-to-end training pipeline.

    raw data -> validation -> features -> chronological back-tests -> final model
             -> interpretation -> artifact bundle (model + serving snapshot + reports)

Run:  ``python -m src.forecasting.pipeline``   (see ``Makefile`` / README)

Training is deliberately a separate process from serving: the API only loads the
artifacts written here and never trains.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.loaders import RawData, load_raw, summarize_dataset, validate_raw
from src.data.schema import DAILY_ATTR_COLS, DATE_COL, ENTITY_COL, TARGET_COL
from src.evaluation.backtest import (
    make_backtest_windows, run_backtest_window, training_origins, training_performance,
)
from src.evaluation.breakdowns import error_breakdowns, error_distribution
from src.evaluation.interpretability import build_importance_report
from src.features.builder import build_feature_tables
from src.forecasting.artifacts import ArtifactPaths
from src.models.baselines import BASELINE_LABELS
from src.models.lightgbm_model import predict_raw, save_model, train_model
from src.utils.io import short_hash, write_json
from src.utils.seed import set_global_seed

logger = logging.getLogger("forecasting.pipeline")

# Constant gzip header (no timestamp) so regenerated bundles are byte-identical.
GZIP = {"method": "gzip", "mtime": 0}

METRIC_DEFINITIONS = {
    "rmsle": "Root mean squared log error over ALL hub-days (log-scale, relative error). The model-selection metric.",
    "mae": "Mean absolute error in orders per hub-day, OPEN days only.",
    "rmse": "Root mean squared error in orders per hub-day, OPEN days only.",
    "wape": "Weighted absolute percentage error = total absolute error / total actual orders, OPEN days only.",
    "rmspe": "Root mean squared percentage error, open days with at least 1 actual order.",
    "mape": "Mean absolute percentage error, open days with at least 1 actual order.",
    "bias": "Total predicted / total actual - 1, open days (positive = over-forecast).",
}


def _pct_reduction(baseline: float, model: float) -> Optional[float]:
    return float((baseline - model) / baseline * 100.0) if baseline and baseline > 0 else None


def run_pipeline(raw: RawData, cfg: Config, artifact_dir: Optional[Path] = None) -> Dict[str, Any]:
    t0 = time.time()
    paths = ArtifactPaths(Path(artifact_dir) if artifact_dir else cfg.artifact_dir)
    set_global_seed(cfg.seed)
    horizon = cfg.forecast.horizon_days

    validate_raw(raw, cfg)
    dataset_summary = summarize_dataset(raw)
    dataset_summary["data_label"] = cfg.data.label
    logger.info("Dataset: %d rows, %d hubs, %s -> %s",
                dataset_summary["train_rows"], dataset_summary["n_hubs"],
                dataset_summary["train_start"], dataset_summary["train_end"])

    tables = build_feature_tables(raw, cfg)
    windows = make_backtest_windows(raw.train_max, horizon, cfg.validation.n_windows)
    first_origin = tables.target_panel.index.min() + pd.Timedelta(days=cfg.forecast.min_history_days)
    for w in windows:
        if w.max_train_origin < first_origin:
            raise ValueError(
                f"Not enough history for {cfg.validation.n_windows} back-test windows of {horizon} days; "
                f"reduce validation.n_windows."
            )

    # ---- 1) chronological back-tests ------------------------------------------------
    primary = None
    train_perf = None
    window_summaries = []
    for i, w in enumerate(windows):
        t1 = time.time()
        res, sample = run_backtest_window(w, tables, cfg, keep_train_sample=200_000 if i == 0 else 0, seed=cfg.seed)
        best = min(res.baseline_metrics, key=lambda n: res.baseline_metrics[n]["rmsle"])
        window_summaries.append({
            **w.to_dict(),
            "n_train_rows": res.n_train_rows,
            "model": res.metrics,
            "best_baseline": best,
            "best_baseline_metrics": res.baseline_metrics[best],
        })
        logger.info("Window %-9s %s..%s  RMSLE=%.5f  WAPE=%.4f  (best baseline %s RMSLE=%.5f)  [%.0fs]",
                    w.name, w.start.date(), w.end.date(), res.metrics["rmsle"], res.metrics["wape"],
                    best, res.baseline_metrics[best]["rmsle"], time.time() - t1)
        if i == 0:
            primary = res
            train_perf = training_performance(res.model, sample)
            del sample
        else:
            res.model = None  # free memory

    # ---- 2) primary-window analysis ---------------------------------------------------
    hist = tables.target_panel.loc[: primary.window.cutoff]
    hub_avg = hist.where(hist > 0).mean().fillna(0.0)
    breakdowns = error_breakdowns(primary.rows, hub_avg, horizon)
    err_dist = error_distribution(primary.rows)

    best = min(primary.baseline_metrics, key=lambda n: primary.baseline_metrics[n]["rmsle"])
    improvement = {
        k: _pct_reduction(primary.baseline_metrics[best][k], primary.metrics[k])
        for k in ("rmsle", "wape", "mae", "rmse")
    }

    # ---- 3) final model on ALL history ---------------------------------------------
    t1 = time.time()
    final_origins = training_origins(tables, raw.train_max - pd.Timedelta(days=horizon), cfg)
    final_train = tables.make_dataset(final_origins, range(1, horizon + 1), require_label=True)
    if final_train[DATE_COL].max() > raw.train_max:
        raise AssertionError("final training targets extend past the last known day")
    final_model = train_model(final_train, cfg)
    n_final_rows, n_final_origins = len(final_train), len(final_origins)
    del final_train
    logger.info("Final model trained on %d rows (%d origins) [%.0fs]", n_final_rows, n_final_origins, time.time() - t1)

    forecast_rows = tables.make_dataset([raw.train_max], range(1, horizon + 1), require_label=False)
    preds = predict_raw(final_model, forecast_rows)
    logger.info("Sanity forecast: %d rows, mean %.0f", len(preds), float(np.mean(preds)))

    # ---- 4) interpretation ---------------------------------------------------------
    t1 = time.time()
    importance = build_importance_report(final_model, primary.model, primary.rows, cfg.interpretability, cfg.seed)
    logger.info("Importance computed [%.0fs]", time.time() - t1)

    # ---- 5) write artifacts ------------------------------------------------------------
    save_model(final_model, paths.model_file)

    paths.hub_metadata.parent.mkdir(parents=True, exist_ok=True)
    raw.hub_meta.to_csv(paths.hub_metadata, index=False)
    raw.train[[ENTITY_COL, DATE_COL, TARGET_COL, "IsOpen", "PromoActive"]].to_csv(paths.history, index=False, compression=GZIP)
    tables.origin_feats[tables.origin_feats["Date"] == raw.train_max].to_csv(paths.origin_features, index=False, compression=GZIP)
    tables.hubweekday_feats[tables.hubweekday_feats["Origin"] == raw.train_max].to_csv(
        paths.hubweekday_features, index=False, compression=GZIP)
    schedule = pd.concat([raw.train[DAILY_ATTR_COLS], raw.future[DAILY_ATTR_COLS]], ignore_index=True)
    schedule = schedule[(schedule[DATE_COL] >= raw.train_max)
                        & (schedule[DATE_COL] <= raw.train_max + pd.Timedelta(days=horizon))]
    schedule.to_csv(paths.schedule, index=False, compression=GZIP)

    bt = primary.rows[[ENTITY_COL, DATE_COL, "h", "IsOpen", "PromoActive", "actual", "predicted",
                       f"baseline_{best}"]].rename(columns={f"baseline_{best}": "baseline"})
    bt["actual"] = bt["actual"].round().astype(int)
    bt["predicted"] = bt["predicted"].round(1)
    bt["baseline"] = bt["baseline"].round(1)
    paths.backtest_predictions.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(paths.backtest_predictions, index=False, compression=GZIP)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "generated_at": now,
        "definitions": METRIC_DEFINITIONS,
        "primary_window": primary.window.to_dict(),
        "model": primary.metrics,
        "training_in_sample": train_perf,
        "baselines": {n: {"label": BASELINE_LABELS[n], **m} for n, m in primary.baseline_metrics.items()},
        "best_baseline": best,
        "improvement_vs_best_baseline_pct": improvement,
        "breakdowns": breakdowns,
        "error_distribution": err_dist,
        "windows": window_summaries,
        "validation_notes": [
            "Each window trains only on data whose TARGET dates precede the window, then forecasts the "
            "next %d days from the window's last known day." % horizon,
            "The primary window was also used to choose hyper-parameters and the boosting-round count, "
            "so it is a validation set, not a pristine test set. Earlier windows are robustness checks.",
            "The deployed model is retrained on all history; its future accuracy cannot be measured "
            "because actuals for the forecast window are not part of the dataset.",
        ],
    }
    write_json(paths.metrics, metrics)
    write_json(paths.importance, importance)
    write_json(paths.dataset_summary, dataset_summary)

    card = {
        "model_version": f"{raw.train_max:%Y%m%d}-{short_hash(paths.model_file)}",
        "algorithm": "LightGBM gradient-boosted trees, Tweedie objective (direct multi-horizon)",
        "data_label": cfg.data.label,
        "created_at": now,
        "seed": cfg.seed,
        "n_estimators": final_model.n_estimators,
        "params": final_model.params,
        "feature_names": final_model.feature_names,
        "categorical_features": final_model.categorical_features,
        "categories": final_model.categories,
        "schedule_shifts": cfg.features.schedule_shifts,
        "horizon_days": horizon,
        "origin_date": raw.train_max,
        "forecast_window": {
            "start": (raw.train_max + pd.Timedelta(days=1)).date().isoformat(),
            "end": (raw.train_max + pd.Timedelta(days=horizon)).date().isoformat(),
        },
        "training": {
            "n_rows": n_final_rows, "n_origins": n_final_origins,
            "origin_stride_days": cfg.forecast.origin_stride_days,
            "history_start": raw.train_min, "history_end": raw.train_max,
            "n_hubs": dataset_summary["n_hubs"],
        },
        "validation_summary": {"window": primary.window.to_dict(), **primary.metrics},
    }
    write_json(paths.model_card, card)

    logger.info("Artifacts written to %s (total %.0fs)", paths.root, time.time() - t0)
    return {"paths": paths, "metrics": metrics, "model_card": card, "importance": importance}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the demand-forecasting model and export artifacts.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--artifact-dir", default=None, help="Output directory (default: from config)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    raw = load_raw(cfg)
    result = run_pipeline(raw, cfg, args.artifact_dir)
    m = result["metrics"]["model"]
    print(f"\nPrimary hold-out: RMSLE={m['rmsle']:.4f}  WAPE={m['wape']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}")


if __name__ == "__main__":
    main()
