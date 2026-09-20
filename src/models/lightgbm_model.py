"""LightGBM (Tweedie) demand model: train / predict / persist.

Why Tweedie on the *raw* target: daily order volume has a point mass at zero
(closed hubs) plus a right-skewed positive tail. The Tweedie family is built for
exactly that shape; with the log link the model predicts a *multiplicative*
demand level, which is also what a relative-error metric (RMSLE) rewards.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import TARGET_COL
from src.features.builder import CATEGORICAL_FEATURES, feature_columns


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_names: List[str]
    categorical_features: List[str]
    categories: Dict[str, list]     # fixed category levels per categorical feature
    params: Dict[str, Any]
    n_estimators: int


def learn_categories(df: pd.DataFrame, categorical_features: List[str]) -> Dict[str, list]:
    return {c: sorted(pd.unique(df[c].dropna()).tolist()) for c in categorical_features}


def prepare_matrix(df: pd.DataFrame, feature_names: List[str], categorical_features: List[str],
                   categories: Dict[str, list]) -> pd.DataFrame:
    """Select model inputs and pin categorical dtypes to the levels seen in training.

    Pinning the levels means a request for a single hub is encoded exactly like the
    training frame; unseen levels become missing rather than silently re-coded.
    """
    X = df[feature_names].copy()
    for c in categorical_features:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def train_model(train_df: pd.DataFrame, cfg: Config, n_estimators: Optional[int] = None) -> TrainedModel:
    feat_cols = feature_columns(train_df)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_cols]
    categories = learn_categories(train_df, cat_cols)
    X = prepare_matrix(train_df, feat_cols, cat_cols, categories)
    y = train_df[TARGET_COL].to_numpy(dtype=float)

    params = dict(cfg.model.params)
    params["seed"] = cfg.seed
    rounds = int(n_estimators or cfg.model.n_estimators)

    dataset = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    return TrainedModel(booster, feat_cols, cat_cols, categories, params, rounds)


def predict_raw(model: TrainedModel, df: pd.DataFrame, num_iteration: Optional[int] = None) -> np.ndarray:
    """Expected demand (Tweedie mean, i.e. exp of the raw score)."""
    X = prepare_matrix(df, model.feature_names, model.categorical_features, model.categories)
    return np.clip(model.booster.predict(X, num_iteration=num_iteration), 0.0, None)


def predict_contributions(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Per-feature TreeSHAP contributions in log-space; last column is the bias.

    ``exp(row.sum())`` equals the prediction, so each contribution ``c`` is a
    multiplicative effect ``exp(c)`` on expected demand.
    """
    X = prepare_matrix(df, model.feature_names, model.categorical_features, model.categories)
    return model.booster.predict(X, pred_contrib=True)


def gain_importance(model: TrainedModel) -> Dict[str, float]:
    gains = model.booster.feature_importance(importance_type="gain")
    return dict(zip(model.feature_names, map(float, gains)))


# --- persistence -------------------------------------------------------------
def save_model(model: TrainedModel, path: Path) -> None:
    """Persist as gzip'd LightGBM text (a few MB instead of ~20 MB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip header constant, so identical models give byte-identical files
    # (and therefore a stable model_version hash).
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(model.booster.model_to_string().encode("utf-8"))


def load_model(path: Path, categories: Dict[str, list], params: Dict[str, Any], n_estimators: int) -> TrainedModel:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        booster = lgb.Booster(model_str=fh.read())
    feat_names = list(booster.feature_name())
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_names]
    return TrainedModel(booster, feat_names, cat_cols, categories, params, n_estimators)
