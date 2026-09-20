"""Model interpretation: gain, permutation and SHAP importance.

* gain        - how much each feature reduced the loss inside the trees (fast, structural).
* permutation - how much held-out RMSLE gets WORSE when a feature is shuffled
                (model-agnostic, measured on data the model did not train on).
* SHAP        - mean |contribution| to individual predictions (TreeSHAP built into
                LightGBM; contributions are in log-space because of the Tweedie link).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.schema import TARGET_COL
from src.evaluation.metrics import apply_closed_hub_rule, rmsle
from src.features.catalog import describe_feature
from src.models.lightgbm_model import (
    TrainedModel, gain_importance, predict_contributions, predict_raw,
)


def permutation_importance(model: TrainedModel, rows: pd.DataFrame, n_rows: int, n_repeats: int,
                           seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed).reset_index(drop=True)
    y = sample[TARGET_COL].to_numpy(dtype=float)
    is_open = sample["IsOpen"].to_numpy()
    base = rmsle(y, apply_closed_hub_rule(predict_raw(model, sample), is_open))

    out: Dict[str, Dict[str, float]] = {}
    for feat in model.feature_names:
        deltas = []
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[feat] = shuffled[feat].to_numpy()[rng.permutation(len(shuffled))]
            score = rmsle(y, apply_closed_hub_rule(predict_raw(model, shuffled), is_open))
            deltas.append(score - base)
        out[feat] = {"mean": float(np.mean(deltas)), "std": float(np.std(deltas)), "baseline_rmsle": float(base)}
    return out


def shap_importance(model: TrainedModel, rows: pd.DataFrame, n_rows: int, seed: int) -> Dict[str, float]:
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed)
    contrib = predict_contributions(model, sample)[:, :-1]  # drop bias column
    return dict(zip(model.feature_names, np.abs(contrib).mean(axis=0).astype(float)))


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


def build_importance_report(final_model: TrainedModel, backtest_model: TrainedModel,
                            holdout_rows: pd.DataFrame, interp_cfg, seed: int) -> Dict:
    perm = permutation_importance(
        backtest_model, holdout_rows, interp_cfg.permutation_rows, interp_cfg.permutation_repeats, seed
    )
    shap = shap_importance(backtest_model, holdout_rows, interp_cfg.shap_rows, seed)
    return {
        "gain": _ranked(gain_importance(final_model)),
        "permutation": _ranked({k: v["mean"] for k, v in perm.items()}, perm),
        "shap": _ranked(shap),
        "meta": {
            "gain": "Loss reduction attributed to each feature inside the final model's trees.",
            "permutation": "Increase in hold-out RMSLE when the feature is shuffled (backtest model, hold-out window).",
            "shap": "Mean absolute TreeSHAP contribution, log-scale (backtest model, hold-out window).",
            "permutation_baseline_rmsle": next(iter(perm.values()))["baseline_rmsle"],
        },
    }
