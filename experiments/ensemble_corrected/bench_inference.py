"""Load-time / memory / latency benchmark of the corrected primary-window models (run AFTER training).

    python -m experiments.ensemble_corrected.bench_inference --work <dir> --mode single|ensemble

Run as a fresh process per mode so the resident-memory high-water mark belongs to that mode alone.
Models were trained on the primary window's 5.09 M rows (same recipes as a final fit).
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from experiments.ensemble_corrected import recipes as R
from src.evaluation.metrics import apply_closed_hub_rule
from src.models.lightgbm_model import load_model, predict_raw


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def timeit(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"median_ms": round(st.median(ts), 2), "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 2), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--mode", choices=["single", "ensemble"], required=True)
    a = ap.parse_args(); W = Path(a.work)
    val = pd.read_pickle(W / "primary_val_features.pkl")
    lg = json.load(open(W / "primary_lightgbm.json")); xg = json.load(open(W / "primary_xgboost.json"))
    feats, cats = lg["features"], lg["categorical"]; levels = lg["levels"]
    store = val[val["HubID"] == 42].sort_values("h").reset_index(drop=True)
    out = {"mode": a.mode, "rss_mb_after_data_load": round(rss_mb(), 0)}

    t = time.perf_counter()
    lgb_model = load_model(W / "lightgbm.txt.gz", levels, {k: v for k, v in lg["params"].items() if k != "num_boost_round"}, lg["params"]["num_boost_round"])
    out["load_seconds"] = {"lightgbm": round(time.perf_counter() - t, 3)}
    predl = lambda df: predict_raw(lgb_model, df)
    if a.mode == "single":
        out["rss_mb_after_model_load"] = round(rss_mb(), 0)
        out["latency_42_rows_lightgbm"] = timeit(lambda: predl(store), 100)
        out["latency_46830_rows_lightgbm"] = timeit(lambda: predl(val), 3)
        out["rss_mb_after_inference"] = round(rss_mb(), 0)
        print(json.dumps(out, indent=1)); return

    import xgboost as xgb
    from catboost import CatBoostRegressor
    t = time.perf_counter(); cb = CatBoostRegressor(); cb.load_model(str(W / "catboost.cbm")); out["load_seconds"]["catboost"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter(); xg_json = xgb.Booster(); xg_json.load_model(str(W / "xgboost.json")); out["load_seconds"]["xgboost_json"] = round(time.perf_counter() - t, 3)
    del xg_json
    t = time.perf_counter(); xb = xgb.Booster(); xb.load_model(str(W / "xgboost.ubj")); out["load_seconds"]["xgboost_ubjson"] = round(time.perf_counter() - t, 3)
    out["rss_mb_after_model_load"] = round(rss_mb(), 0)

    predc = lambda df: R.predict_catboost(cb, df, feats, cats)
    predx = lambda df: R.predict_xgboost(xb, df, feats, cats, levels)
    ens = lambda df: apply_closed_hub_rule(R.blend(predl(df), predc(df), predx(df)), df["IsOpen"])
    for name, fn in (("lightgbm", predl), ("catboost", predc), ("xgboost", predx), ("ensemble", ens)):
        out[f"latency_42_rows_{name}"] = timeit(lambda fn=fn: fn(store), 100)
    out["latency_46830_rows_ensemble"] = timeit(lambda: ens(val), 3)
    out["rss_mb_after_inference"] = round(rss_mb(), 0)

    # serving-safety regression on the real corrected models: single-store == same rows inside the full frame
    full = {"lightgbm": pd.Series(predl(val), index=val.index), "catboost": pd.Series(predc(val), index=val.index), "xgboost": pd.Series(predx(val), index=val.index)}
    rng = np.random.default_rng(0); hubs = rng.choice(val["HubID"].unique(), 60, replace=False)
    worst = {k: 0.0 for k in full}
    for h in hubs:
        sub = val[val["HubID"] == h]
        for k, fn in (("lightgbm", predl), ("catboost", predc), ("xgboost", predx)):
            ref = full[k].loc[sub.index].to_numpy(); got = fn(sub)
            worst[k] = max(worst[k], float(np.max(np.abs(got - ref) / np.maximum(ref, 1.0))))
    out["single_store_vs_full_frame_max_rel_diff_60_stores"] = worst
    sub = val[val["HubID"] == 42]; naive = sub[feats].copy()
    for c in cats: naive[c] = naive[c].astype("category")
    bad = np.clip(xb.predict(xgb.DMatrix(naive, enable_categorical=True)), 0, None); ref = full["xgboost"].loc[sub.index].to_numpy()
    out["xgboost_UNPINNED_single_store_max_rel_diff_store_42"] = float(np.max(np.abs(bad - ref) / np.maximum(ref, 1.0)))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
