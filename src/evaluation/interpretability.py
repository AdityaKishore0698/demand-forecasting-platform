"""Model interpretation: gain, permutation and SHAP importance.

* gain        - how much each feature reduced the loss inside the trees (fast, structural). For the ensemble this
                is reported PER MODEL (units differ across libraries); the headline ``gain`` list is the LightGBM
                component only and is labelled as such.
* permutation - how much held-out RMSLE gets worse when a feature is shuffled. Model-agnostic: for the ensemble
                it is computed on the ENSEMBLE prediction, so it is exact for the deployed model.
* SHAP        - mean |contribution| to individual predictions (exact TreeSHAP per model; contributions are in
                log / log1p space). The ensemble-level list is the weight-averaged importance of the three models
                and is labelled APPROXIMATE (a weighted average of margin-space attributions is not an exact
                attribution of the blended output).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.schema import TARGET_COL
from src.evaluation.metrics import apply_closed_hub_rule, rmsle
from src.features.catalog import describe_feature
from src.models.dispatch import AnyModel, feature_names, is_ensemble, predict as predict_demand
from src.models.ensemble import HUMAN_NAME, MEMBERS, EnsembleModel, predict_contributions_members
from src.models.lightgbm_model import TrainedModel, gain_importance, predict_contributions


def permutation_importance(model: AnyModel, rows: pd.DataFrame, n_rows: int, n_repeats: int,
                           seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed).reset_index(drop=True)
    y = sample[TARGET_COL].to_numpy(dtype=float)
    is_open = sample["IsOpen"].to_numpy()
    base = rmsle(y, apply_closed_hub_rule(predict_demand(model, sample), is_open))

    out: Dict[str, Dict[str, float]] = {}
    for feat in feature_names(model):
        deltas = []
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[feat] = shuffled[feat].to_numpy()[rng.permutation(len(shuffled))]
            score = rmsle(y, apply_closed_hub_rule(predict_demand(model, shuffled), is_open))
            deltas.append(score - base)
        out[feat] = {"mean": float(np.mean(deltas)), "std": float(np.std(deltas)), "baseline_rmsle": float(base)}
    return out


def shap_importance(model: TrainedModel, rows: pd.DataFrame, n_rows: int, seed: int) -> Dict[str, float]:
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed)
    contrib = predict_contributions(model, sample)[:, :-1]  # drop bias column
    return dict(zip(model.feature_names, np.abs(contrib).mean(axis=0).astype(float)))


def shap_importance_by_member(model: EnsembleModel, rows: pd.DataFrame, n_rows: int, seed: int) -> Dict[str, Dict[str, float]]:
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed)
    return {m: dict(zip(model.feature_names, np.abs(c[:, :-1]).mean(axis=0).astype(float)))
            for m, c in predict_contributions_members(model, sample).items()}


def gain_importance_by_member(model: EnsembleModel) -> Dict[str, Dict[str, float]]:
    """Structural importance per model, each in its own units (not comparable across libraries)."""
    names = model.feature_names
    lg = gain_importance(model.lightgbm)
    xs = model.xgboost.get_score(importance_type="total_gain")
    cb = dict(zip(model.catboost.feature_names_, map(float, model.catboost.get_feature_importance())))
    return {"lightgbm": lg, "catboost": {n: cb.get(n, 0.0) for n in names}, "xgboost": {n: float(xs.get(n, 0.0)) for n in names}}


def _ranked(values: Dict[str, float], extra: Dict[str, Dict[str, float]] = None) -> List[Dict]:
    total = sum(max(v, 0.0) for v in values.values()) or 1.0
    items = []
    for feat, val in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        item = describe_feature(feat)
        item["value"] = float(val)
        item["share"] = float(max(val, 0.0) / total)
        if extra and feat in extra:
            item["std"] = extra[feat]["std"]
        items.append(item)
    return items


def build_importance_report(final_model: AnyModel, backtest_model: AnyModel,
                            holdout_rows: pd.DataFrame, interp_cfg, seed: int) -> Dict:
    perm = permutation_importance(
        backtest_model, holdout_rows, interp_cfg.permutation_rows, interp_cfg.permutation_repeats, seed
    )
    base = {"permutation_baseline_rmsle": next(iter(perm.values()))["baseline_rmsle"]}
    if not is_ensemble(final_model):
        shap = shap_importance(backtest_model, holdout_rows, interp_cfg.shap_rows, seed)
        return {
            "gain": _ranked(gain_importance(final_model)),
            "permutation": _ranked({k: v["mean"] for k, v in perm.items()}, perm),
            "shap": _ranked(shap),
            "meta": {
                "gain": "Loss reduction attributed to each feature inside the final model's trees.",
                "permutation": "Increase in hold-out RMSLE when the feature is shuffled (backtest model, hold-out window).",
                "shap": "Mean absolute TreeSHAP contribution, log-scale (backtest model, hold-out window).",
                **base,
            },
        }

    w = final_model.weights
    per_member = shap_importance_by_member(backtest_model, holdout_rows, interp_cfg.shap_rows, seed)
    weighted = {f: sum(w[m] * per_member[m][f] for m in MEMBERS) for f in final_model.feature_names}
    gains = gain_importance_by_member(final_model)
    return {
        "gain": _ranked(gains["lightgbm"]),
        "permutation": _ranked({k: v["mean"] for k, v in perm.items()}, perm),
        "shap": _ranked(weighted),
        "shap_by_model": {m: _ranked(v) for m, v in per_member.items()},
        "gain_by_model": {m: _ranked(v) for m, v in gains.items()},
        "meta": {
            "model_type": "LightGBM + CatBoost + XGBoost ensemble",
            "gain": "Loss reduction inside the LightGBM component's trees ONLY (0.6 of the blend); not a measure of the ensemble. Per-model gain: gain_by_model.",
            "permutation": "Increase in hold-out RMSLE when the feature is shuffled, measured on the ENSEMBLE prediction (backtest ensemble, hold-out window).",
            "shap": "APPROXIMATE ensemble view: mean absolute TreeSHAP contribution of each model (log / log1p scale), averaged with the blend weights "
                    f"({w['lightgbm']} / {w['catboost']} / {w['xgboost']}). Each model's own list is exact: shap_by_model.",
            **base,
        },
    }
