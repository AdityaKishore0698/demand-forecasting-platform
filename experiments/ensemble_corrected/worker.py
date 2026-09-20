"""Fit ONE model on ONE chronological window under the corrected methodology and save its validation predictions.

    python -m experiments.ensemble_corrected.worker --window primary --model xgboost --out <dir>

Each (window, model) runs in its own process so that wall-clock time and peak memory are attributable to that
model. The window / training-origin / feature logic is the production code (``src.evaluation.backtest``,
``src.features.builder``), i.e. the CURRENT corrected methodology: ``*_next1`` schedule flags are blanked on
horizon 42 in training rows and validation rows alike (``mask_unpublished_schedule``).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from experiments.ensemble_corrected import recipes as R
from src.config import load_config
from src.data.loaders import load_raw
from src.data.schema import TARGET_COL
from src.evaluation.backtest import assert_no_temporal_leakage, make_backtest_windows, training_origins
from src.features.builder import CATEGORICAL_FEATURES, build_feature_tables, feature_columns
from src.models.lightgbm_model import learn_categories, predict_raw, save_model, train_model

MODELS = ("lightgbm", "xgboost", "catboost")


def rss_mb() -> float:
    """Process high-water resident set size in MB (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def frame_hash(df: pd.DataFrame, cols) -> str:
    return str(int(pd.util.hash_pandas_object(df[list(cols)], index=False).sum() % (2 ** 63)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=["primary", "earlier_1", "earlier_2"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-models", action="store_true", help="persist the fitted model (primary window only)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    raw = load_raw(cfg)
    tables = build_feature_tables(raw, cfg)
    horizon = cfg.forecast.horizon_days
    windows = {w.name: w for w in make_backtest_windows(raw.train_max, horizon, cfg.validation.n_windows)}
    w = windows[a.window]

    origins = training_origins(tables, w.max_train_origin, cfg)
    train_df = tables.make_dataset(origins, range(1, horizon + 1), require_label=True)
    assert_no_temporal_leakage(train_df, w)                       # training targets all precede the window
    val = tables.make_dataset([w.cutoff], range(1, horizon + 1), require_label=True)
    assert val["Date"].min() == w.start and val["Date"].max() == w.end and (val["h"] <= horizon).all()
    assert tables.schedule_horizon == horizon                     # the corrected methodology is switched on

    feats = feature_columns(train_df)
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    levels = learn_categories(train_df, cats)
    nxt = [c for c in feats if c.endswith("_next1")]
    assert val.loc[val["h"] == horizon, nxt].isna().all().all() and train_df.loc[train_df["h"] == horizon, nxt].isna().all().all()

    meta = {
        "window": a.window, "model": a.model, "cutoff": str(w.cutoff.date()), "start": str(w.start.date()), "end": str(w.end.date()),
        "n_train_rows": int(len(train_df)), "n_val_rows": int(len(val)), "n_features": len(feats), "features": feats,
        "categorical": cats, "train_frame_hash": frame_hash(train_df, feats + [TARGET_COL]), "val_frame_hash": frame_hash(val, feats + [TARGET_COL]),
        "last_training_target_date": str(train_df["Date"].max().date()), "seed": cfg.seed,
        "versions": {}, "cpu_count": os.cpu_count(), "platform": platform.platform(),
    }
    keys = val[["HubID", "Date", "h", "IsOpen"]].copy()
    keys["actual"] = val[TARGET_COL].to_numpy(dtype=float)
    keys.to_csv(out / f"{a.window}_keys.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    if a.model == "lightgbm" and a.window == "primary":
        val.to_pickle(out / "primary_val_features.pkl")             # for the inference benchmark (local only)
    del tables, raw
    gc.collect()
    meta["rss_mb_before_fit"] = round(rss_mb(), 0)

    t0 = time.perf_counter()
    if a.model == "lightgbm":
        import lightgbm
        meta["versions"]["lightgbm"] = lightgbm.__version__
        model = train_model(train_df, cfg)                          # production recipe == archived recipe
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter(); pred = predict_raw(model, val); pred_s = time.perf_counter() - t1
        meta["params"] = {**model.params, "num_boost_round": model.n_estimators}
        if a.save_models:
            save_model(model, out / "lightgbm.txt.gz")
            model.booster.save_model(str(out / "lightgbm.txt"))
            meta["levels"] = model.categories
    elif a.model == "xgboost":
        import xgboost
        meta["versions"]["xgboost"] = xgboost.__version__
        model = R.fit_xgboost(train_df, feats, cats, levels, seed=cfg.seed)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter(); pred = R.predict_xgboost(model, val, feats, cats, levels); pred_s = time.perf_counter() - t1
        meta["params"] = {**R.XGB_PARAMS, "seed": cfg.seed, "num_boost_round": R.XGB_NUM_BOOST_ROUND}
        if a.save_models:
            model.save_model(str(out / "xgboost.json")); model.save_model(str(out / "xgboost.ubj"))
    else:
        import catboost
        meta["versions"]["catboost"] = catboost.__version__
        model = R.fit_catboost(train_df, feats, cats, seed=cfg.seed)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter(); pred = R.predict_catboost(model, val, feats, cats); pred_s = time.perf_counter() - t1
        meta["params"] = {**R.CATBOOST_PARAMS, "random_seed": cfg.seed}
        if a.save_models:
            model.save_model(str(out / "catboost.cbm"))

    meta.update({"fit_seconds": round(fit_s, 1), "predict_seconds_46830_rows": round(pred_s, 2), "rss_mb_peak_after_fit": round(rss_mb(), 0)})
    assert pred.shape == (len(val),) and np.isfinite(pred).all() and (pred >= 0).all()
    np.save(out / f"{a.window}_{a.model}.npy", pred)
    json.dump(meta, open(out / f"{a.window}_{a.model}.json", "w"), indent=1)
    print(f"[{a.window}/{a.model}] fit {fit_s:.0f}s  predict {pred_s:.1f}s  peak RSS {meta['rss_mb_peak_after_fit']:.0f} MB "
          f"(before fit {meta['rss_mb_before_fit']:.0f})  train_rows={len(train_df)}", flush=True)


if __name__ == "__main__":
    main()
