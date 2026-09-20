"""Combine the per-(window, model) worker outputs into the corrected-methodology comparison.

    python -m experiments.ensemble_corrected.combine --work <dir> --out <dir> [--windows primary earlier_1 earlier_2]

No tuning happens here: the blend weights are the archived 0.6 / 0.1 / 0.3 and every model is used exactly as the
worker produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from experiments.ensemble_corrected import recipes as R
from src.evaluation.metrics import apply_closed_hub_rule, regression_report

MEMBERS = ("lightgbm", "catboost", "xgboost")
ALL = MEMBERS + ("ensemble",)


def score(actual, pred, is_open) -> dict:
    r = regression_report(actual, pred, is_open)
    o = (np.asarray(is_open) == 1) & (np.asarray(actual) > 0)
    pct = np.abs(np.asarray(pred)[o] - np.asarray(actual)[o]) / np.asarray(actual)[o]
    return {"rmsle": r["rmsle"], "wape": r["wape"], "mae": r["mae"], "rmse": r["rmse"], "bias": r["bias"],
            "within_10pct": float((pct <= 0.10).mean()), "within_20pct": float((pct <= 0.20).mean()),
            "n_rows": r["n_rows"], "n_open_rows": r["n_open_rows"]}


def deltas(ens: dict, base: dict) -> dict:
    """Ensemble minus LightGBM. Negative error deltas mean the ensemble is better."""
    return {"rmsle_abs": ens["rmsle"] - base["rmsle"], "rmsle_rel_pct": (ens["rmsle"] / base["rmsle"] - 1) * 100,
            "wape_pp": (ens["wape"] - base["wape"]) * 100, "mae": ens["mae"] - base["mae"], "rmse": ens["rmse"] - base["rmse"],
            "within_10pct_pp": (ens["within_10pct"] - base["within_10pct"]) * 100}


def store_bootstrap(keys: pd.DataFrame, preds: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over STORES for (ensemble - LightGBM). Captures which stores are in the window only; it does not
    capture training randomness, window-to-window variation, or the fact that tree counts/weights were tuned on the
    primary window."""
    y = keys["actual"].to_numpy(float); op = keys["IsOpen"].to_numpy() == 1
    codes, uniq = pd.factorize(keys["HubID"]); n = len(uniq)
    def agg(p):
        p = apply_closed_hub_rule(p, keys["IsOpen"])
        sle = (np.log1p(p) - np.log1p(y)) ** 2
        f = lambda v, m=None: np.bincount(codes, weights=(v if m is None else v * m), minlength=n)
        return dict(sle=f(sle), n=np.bincount(codes, minlength=n).astype(float), ab=f(np.abs(p - y), op), act=f(y, op),
                    sq=f((p - y) ** 2, op), no=f(np.ones(len(y)), op))
    A, B = agg(preds["ensemble"]), agg(preds["lightgbm"])
    rng = np.random.default_rng(seed); out = {"rmsle": [], "wape": [], "mae": [], "rmse": []}
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        def met(G):
            return {"rmsle": np.sqrt(G["sle"][s].sum() / G["n"][s].sum()), "wape": G["ab"][s].sum() / G["act"][s].sum(),
                    "mae": G["ab"][s].sum() / G["no"][s].sum(), "rmse": np.sqrt(G["sq"][s].sum() / G["no"][s].sum())}
        a, b = met(A), met(B)
        for k in out: out[k].append(a[k] - b[k])
    res = {}
    for k, v in out.items():
        v = np.array(v)
        res[k] = {"mean_diff": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "share_resamples_ensemble_better": float((v < 0).mean())}
    res["n_boot"] = n_boot; res["n_stores"] = int(n)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--windows", nargs="+", default=["primary", "earlier_1", "earlier_2"])
    a = ap.parse_args(); W, O = Path(a.work), Path(a.out); O.mkdir(parents=True, exist_ok=True)

    res = {"windows": {}, "checks": {}, "resources": {"per_fit": {}}}
    pooled_keys, pooled_pred = [], {m: [] for m in ALL}
    pooled_ab = []
    BW = R.BLEND_WEIGHTS
    prod = json.load(open(REPO / "artifacts/reports/metrics.json"))
    prod_w = {w["name"]: w["model"] for w in prod["windows"]}
    for wn in a.windows:
        meta = {m: json.load(open(W / f"{wn}_{m}.json")) for m in MEMBERS}
        keys = pd.read_csv(W / f"{wn}_keys.csv.gz", parse_dates=["Date"])
        # ---- apples-to-apples checks -------------------------------------------------------------------------
        chk = {"identical_training_frame_hash": len({meta[m]["train_frame_hash"] for m in MEMBERS}) == 1,
               "identical_validation_frame_hash": len({meta[m]["val_frame_hash"] for m in MEMBERS}) == 1,
               "identical_feature_names_and_order": all(meta[m]["features"] == meta["lightgbm"]["features"] for m in MEMBERS),
               "identical_training_row_count": len({meta[m]["n_train_rows"] for m in MEMBERS}) == 1,
               "training_targets_precede_window": meta["lightgbm"]["last_training_target_date"] < meta["lightgbm"]["start"],
               "n_features": meta["lightgbm"]["n_features"], "n_train_rows": meta["lightgbm"]["n_train_rows"],
               "last_training_target_date": meta["lightgbm"]["last_training_target_date"], "window": [meta["lightgbm"]["start"], meta["lightgbm"]["end"]]}
        assert all(v for k, v in chk.items() if isinstance(v, bool)), (wn, chk)
        raw = {m: np.load(W / f"{wn}_{m}.npy") for m in MEMBERS}
        raw["ensemble"] = R.blend(raw["lightgbm"], raw["catboost"], raw["xgboost"])
        pred = {m: apply_closed_hub_rule(raw[m], keys["IsOpen"]) for m in ALL}
        assert all(np.isfinite(pred[m]).all() and pred[m].shape == (len(keys),) for m in ALL)
        sc = {m: score(keys["actual"], pred[m], keys["IsOpen"]) for m in ALL}
        # DIAGNOSTIC ONLY (not a candidate, not tuned): the archived weights with CatBoost removed and the remaining two
        # renormalised to sum to 1 (0.6/0.9, 0.3/0.9). Declared before looking at any result; shows CatBoost's marginal effect.
        ab_raw = (BW["lightgbm"] * raw["lightgbm"] + BW["xgboost"] * raw["xgboost"]) / (BW["lightgbm"] + BW["xgboost"])
        ab = apply_closed_hub_rule(ab_raw, keys["IsOpen"]); pooled_ab.append(ab)
        sc_ab = score(keys["actual"], ab, keys["IsOpen"])
        lg_prod = prod_w[wn]
        chk["lightgbm_equals_production_pipeline_metrics"] = bool(abs(sc["lightgbm"]["rmsle"] - lg_prod["rmsle"]) < 1e-9 and abs(sc["lightgbm"]["wape"] - lg_prod["wape"]) < 1e-9)
        res["windows"][wn] = {"start": meta["lightgbm"]["start"], "end": meta["lightgbm"]["end"], "cutoff": meta["lightgbm"]["cutoff"], "metrics": sc,
                              "ensemble_minus_lightgbm": deltas(sc["ensemble"], sc["lightgbm"]), "diagnostic_drop_catboost_renormalized": sc_ab, "store_bootstrap_ensemble_minus_lightgbm": store_bootstrap(keys, raw),
                              "checks": chk}
        pd.DataFrame({"origin_date": meta["lightgbm"]["cutoff"], "store_id": keys["HubID"], "date": keys["Date"].dt.date.astype(str), "horizon": keys["h"],
                      "is_open": keys["IsOpen"], "actual": keys["actual"].astype(int), **{f"{m}_prediction": pred[m].round(2) for m in ALL}}
                     ).to_csv(O / f"predictions_{wn}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pooled_keys.append(keys); [pooled_pred[m].append(pred[m]) for m in ALL]
        for m in MEMBERS:
            res["resources"]["per_fit"][f"{wn}/{m}"] = {k: meta[m][k] for k in ("fit_seconds", "predict_seconds_46830_rows", "rss_mb_before_fit", "rss_mb_peak_after_fit", "n_train_rows")}
    # ---- pooled and mean-of-windows --------------------------------------------------------------------------------
    K = pd.concat(pooled_keys, ignore_index=True); P = {m: np.concatenate(pooled_pred[m]) for m in ALL}
    pooled = {m: score(K["actual"], P[m], K["IsOpen"]) for m in ALL}
    mean_w = {m: {k: float(np.mean([res["windows"][w]["metrics"][m][k] for w in a.windows])) for k in ("rmsle", "wape", "mae", "rmse", "within_10pct")} for m in ALL}
    res["pooled_over_windows"] = {"n_rows": len(K), "diagnostic_drop_catboost_renormalized": score(K["actual"], np.concatenate(pooled_ab), K["IsOpen"]), "metrics": pooled, "ensemble_minus_lightgbm": deltas(pooled["ensemble"], pooled["lightgbm"])}
    res["mean_of_window_metrics"] = {"metrics": mean_w, "ensemble_minus_lightgbm": deltas({**mean_w["ensemble"]}, {**mean_w["lightgbm"]})}
    res["ensemble_better_than_lightgbm_in_windows"] = {k: {w: bool(res["windows"][w]["metrics"]["ensemble"][k] < res["windows"][w]["metrics"]["lightgbm"][k]) for w in a.windows} for k in ("rmsle", "wape", "mae", "rmse")}
    # ---- references (old numbers) ------------------------------------------------------------------------------------
    old = json.load(open(REPO / "artifacts/experiments/ensemble/metrics.json"))
    res["references"] = {"pre_correction_reproduced_primary": {m: {k: old["reproduced_validation_metrics"][m][k] for k in ("rmsle", "wape", "mae", "rmse")} for m in ALL},
                         "note": "Pre-correction numbers used the un-masked next-day schedule flags on horizon 42; they are NOT comparable and are shown only to quantify the shift."}
    if "primary" in a.windows:
        c, o = res["windows"]["primary"]["metrics"], old["reproduced_validation_metrics"]
        res["references"]["shift_corrected_minus_precorrection_primary"] = {m: {k: c[m][k] - o[m][k] for k in ("rmsle", "wape", "mae", "rmse")} for m in ALL}
    # ---- sizes and benchmark -------------------------------------------------------------------------------------------
    sizes = {}
    for f in ("lightgbm.txt", "lightgbm.txt.gz", "catboost.cbm", "xgboost.json", "xgboost.ubj"):
        if (W / f).exists(): sizes[f] = round((W / f).stat().st_size / 1048576, 2)
    res["resources"]["model_files_mb_primary_window_models"] = sizes
    for mode in ("single", "ensemble"):
        p = W / f"bench_{mode}.json"
        if p.exists(): res["resources"][f"inference_benchmark_{mode}"] = json.load(open(p))
    res["resources"]["total_fit_seconds"] = {w: round(sum(res["resources"]["per_fit"][f"{w}/{m}"]["fit_seconds"] for m in MEMBERS), 1) for w in a.windows}
    res["methodology"] = {"label": "CORRECTED: *_next1 schedule flags masked on horizon 42 in training, validation and serving (mask_unpublished_schedule)",
        "windows": "make_backtest_windows(train_max, 42, 3): primary / earlier_1 / earlier_2; training origins <= cutoff-42 days, weekly stride; assert_no_temporal_leakage per window",
        "blend": R.BLEND_WEIGHTS, "blend_space": "raw demand; members' outputs converted to demand first (CatBoost: expm1)", "post_rule": "clip >= 0; IsOpen == 0 -> 0",
        "no_tuning": "weights, tree counts and hyper-parameters are the archived final recipe; nothing was tuned on these results",
        "tuning_caveat": "the archived tree counts (1160 / 1320 / 590) and weights were chosen on the PRIMARY window under the pre-correction methodology, so the primary window is not an untouched test set",
        "xgboost_categoricals": "pinned levels learned from each window's training frame", "catboost_categoricals": "strings (frame-independent)"}
    json.dump(res, open(O / "metrics.json", "w"), indent=1)
    # ---- console tables ----------------------------------------------------------------------------------------------------
    def table(title, M):
        print(f"\n{title}\nModel        RMSLE      WAPE      MAE      RMSE   within10%")
        for m in ALL: print(f"{m.capitalize():11s} {M[m]['rmsle']:.5f}  {M[m]['wape']*100:6.3f}%  {M[m]['mae']:7.2f}  {M[m]['rmse']:7.2f}   {M[m]['within_10pct']*100:5.1f}%")
    for w in a.windows:
        table(f"{w}  ({res['windows'][w]['start']} -> {res['windows'][w]['end']})", res["windows"][w]["metrics"]); print("  ens - lgbm:", {k: round(v, 5) for k, v in res["windows"][w]["ensemble_minus_lightgbm"].items()})
    table("POOLED over windows", pooled); print("  ens - lgbm:", {k: round(v, 5) for k, v in res["pooled_over_windows"]["ensemble_minus_lightgbm"].items()})


if __name__ == "__main__":
    main()
