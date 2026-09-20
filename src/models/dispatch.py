"""One place that decides between the final ensemble and the single-LightGBM rollback path.

Everything downstream (back-tests, pipeline, interpretability, the service) calls these helpers instead of
importing a specific model family, so ``model.type: lightgbm_single`` keeps working unchanged.
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from src.config import Config
from src.models.ensemble import MODEL_TYPE_ENSEMBLE, MODEL_TYPE_SINGLE, EnsembleModel, fit_ensemble, predict_ensemble
from src.models.lightgbm_model import TrainedModel, predict_raw, train_model

AnyModel = Union[TrainedModel, EnsembleModel]


def is_ensemble(model: AnyModel) -> bool:
    return isinstance(model, EnsembleModel)


def model_type_id(model: AnyModel) -> str:
    return MODEL_TYPE_ENSEMBLE if is_ensemble(model) else MODEL_TYPE_SINGLE


def fit(train_df: pd.DataFrame, cfg: Config) -> AnyModel:
    return fit_ensemble(train_df, cfg) if cfg.model.type == MODEL_TYPE_ENSEMBLE else train_model(train_df, cfg)


def predict(model: AnyModel, df: pd.DataFrame) -> np.ndarray:
    """Demand prediction (before the closed-day rule) for the rows of ``df``."""
    return predict_ensemble(model, df) if is_ensemble(model) else predict_raw(model, df)


def feature_names(model: AnyModel) -> List[str]:
    return list(model.feature_names)
