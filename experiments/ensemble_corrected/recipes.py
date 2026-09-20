"""Model recipes for the corrected-methodology ensemble evaluation.

SOURCE OF TRUTH: the archived final recipes from the original experiment scripts (fit / predict functions, the
LightGBM / XGBoost / CatBoost parameter dictionaries, the round counts and the blend weights). Those original scripts
are not part of this repository; nothing here was tuned.

LightGBM is NOT re-implemented: the production ``src.models.lightgbm_model`` already reproduces the
archived recipe bit-for-bit (verified: identical predictions to the legacy code). This module only adds
CatBoost and XGBoost, plus the explicit categorical encodings.

Changes relative to the archived code (all documented in docs/ENSEMBLE_CORRECTED_EVALUATION.md):
  1. XGBoost categoricals are encoded with PINNED levels (see ``encode_xgboost``). The archived code used
     ``astype("category")`` per frame, which is only correct when every frame contains every level.
  2. CatBoost is created with ``allow_writing_files=False`` (no ``catboost_info/`` scratch directory);
     this does not affect the fitted model.
  Everything else - parameters, round counts, target transforms, seeds, thread settings - is verbatim.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data.schema import TARGET_COL

# ---- blend (archived final recipe) -------------------------------------------------------------------
BLEND_WEIGHTS = {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3}

# ---- CatBoost (archived final recipe) ---------------------------------------------------------------
CATBOOST_PARAMS = dict(
    loss_function="RMSE", eval_metric="RMSE",
    iterations=1320, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
    boosting_type="Plain", bootstrap_type="Bernoulli", subsample=0.8,
    thread_count=-1, verbose=0,
)

# ---- XGBoost (archived final recipe) ----------------------------------------------------------------
XGB_PARAMS = {
    "objective": "reg:tweedie", "tweedie_variance_power": 1.1,
    "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50, "reg_lambda": 1.0,
    "subsample": 0.85, "colsample_bytree": 0.85, "tree_method": "hist",
}
XGB_NUM_BOOST_ROUND = 590


def blend(lightgbm: np.ndarray, catboost: np.ndarray, xgboost: np.ndarray) -> np.ndarray:
    """0.6 / 0.1 / 0.3 weighted sum of the three members' outputs in raw demand space."""
    w = BLEND_WEIGHTS
    return w["lightgbm"] * lightgbm + w["catboost"] * catboost + w["xgboost"] * xgboost


# ------------------------------------------------------------------------------------------------------------
# Categorical encodings
# ------------------------------------------------------------------------------------------------------------
def encode_xgboost(df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str],
                   levels: Dict[str, list]) -> pd.DataFrame:
    """Encode a frame for XGBoost with category levels PINNED to those seen in training.

    Why this matters (measured, see docs): XGBoost 2.1 stores no category levels in the model; it consumes the
    integer *codes* of the pandas Categorical. ``astype("category")`` derives codes from the values present in
    THAT frame, so a single-store frame gives store 42 the code 0 instead of its training code (41), and the
    model then behaves as if it were a different store. Pinning ``categories=levels`` makes codes identical
    for every frame; an unseen level becomes code -1, which XGBoost treats as missing.
    """
    X = df[list(feature_names)].copy()
    for c in categorical:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def encode_catboost(df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    """Archived CatBoost encoding: categoricals as strings (frame-independent by construction)."""
    X = df[list(feature_names)].copy()
    for c in categorical:
        X[c] = X[c].fillna(-1).astype(int).astype(str)
    return X


# ------------------------------------------------------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------------------------------------------------------
def fit_catboost(train_df: pd.DataFrame, feature_names: List[str], categorical: List[str], seed: int,
                 iterations: int | None = None, **overrides):
    from catboost import CatBoostRegressor, Pool

    params = dict(CATBOOST_PARAMS, **overrides)
    if iterations is not None:                     # only the unit tests shrink this
        params["iterations"] = iterations
    X = encode_catboost(train_df, feature_names, categorical)
    y = np.log1p(train_df[TARGET_COL].to_numpy(dtype=float))
    cat_idx = [list(feature_names).index(c) for c in categorical]
    model = CatBoostRegressor(random_seed=seed, allow_writing_files=False, **params)
    model.fit(Pool(X, y, cat_features=cat_idx))
    return model


def predict_catboost(model, df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str]) -> np.ndarray:
    """expm1 of the log1p-target prediction (archived behaviour), clipped at 0."""
    return np.clip(np.expm1(model.predict(encode_catboost(df, feature_names, categorical))), 0.0, None)


# ------------------------------------------------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------------------------------------------------
def fit_xgboost(train_df: pd.DataFrame, feature_names: List[str], categorical: List[str],
                levels: Dict[str, list], seed: int, num_boost_round: int | None = None, **overrides):
    import xgboost as xgb

    params = dict(XGB_PARAMS, seed=seed, **overrides)
    X = encode_xgboost(train_df, feature_names, categorical, levels)
    dtrain = xgb.DMatrix(X, label=train_df[TARGET_COL].to_numpy(dtype=float), enable_categorical=True)
    return xgb.train(params, dtrain, num_boost_round=int(num_boost_round or XGB_NUM_BOOST_ROUND), verbose_eval=False)


def predict_xgboost(model, df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str],
                    levels: Dict[str, list]) -> np.ndarray:
    import xgboost as xgb

    X = encode_xgboost(df, feature_names, categorical, levels)
    return np.clip(model.predict(xgb.DMatrix(X, enable_categorical=True)), 0.0, None)
