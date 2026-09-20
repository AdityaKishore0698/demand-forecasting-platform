"""Production 3-model boosting ensemble: LightGBM + CatBoost + XGBoost, weights 0.6 / 0.1 / 0.3.

Ported from the validated experiment ``experiments/ensemble_corrected/`` (recipes are the archived final
recipes; nothing here is tuned). The blend is a weighted sum of the three members' outputs in raw demand space:

    final = 0.6 * LightGBM + 0.1 * CatBoost + 0.3 * XGBoost      (then the closed-day rule, applied by callers)

Serving rules enforced here:
* every member scores the SAME rows in the SAME order; lengths are checked, NaN / inf are rejected (never
  silently repaired) and demand must be non-negative;
* XGBoost categoricals are encoded with category levels PINNED to the training levels. XGBoost 2.1 stores no
  category levels in the model - it consumes the integer codes of the pandas Categorical - so an unpinned
  ``astype("category")`` on a single-store request re-codes the levels and changes predictions (measured up to
  21.6 %). CatBoost receives categoricals as strings (frame-independent); LightGBM stores its own mapping.
* models are loaded once (``load_ensemble``) and verified against the SHA-256 hashes in the model card.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import TARGET_COL
from src.features.builder import CATEGORICAL_FEATURES, feature_columns
from src.models.lightgbm_model import (
    TrainedModel, learn_categories, load_model, predict_contributions, prepare_matrix, save_model, train_model,
)

MODEL_TYPE_ENSEMBLE = "boosting_ensemble"
MODEL_TYPE_SINGLE = "lightgbm_single"
MEMBERS = ("lightgbm", "catboost", "xgboost")
FILES = {"lightgbm": "lightgbm.txt.gz", "catboost": "catboost.cbm", "xgboost": "xgboost.ubj"}
HUMAN_NAME = {"lightgbm": "LightGBM", "catboost": "CatBoost", "xgboost": "XGBoost"}
TARGET_TRANSFORM = {"lightgbm": "raw demand (Tweedie)", "catboost": "log1p(demand); prediction = expm1",
                    "xgboost": "raw demand (Tweedie)"}


class EnsembleError(ValueError):
    """An ensemble invariant was violated (bad weights, feature mismatch, corrupt/mismatched artifact, bad output)."""


# ---------------------------------------------------------------------------------------------------------
# Encodings (verbatim from experiments/ensemble_corrected/recipes.py)
# ---------------------------------------------------------------------------------------------------------
def encode_xgboost(df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str],
                   levels: Dict[str, list]) -> pd.DataFrame:
    """Categoricals as pandas Categoricals with PINNED training levels (unseen level -> code -1 = missing)."""
    X = df[list(feature_names)].copy()
    for c in categorical:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def encode_catboost(df: pd.DataFrame, feature_names: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    """Categoricals as strings (frame-independent), exactly as the archived recipe."""
    X = df[list(feature_names)].copy()
    for c in categorical:
        X[c] = X[c].fillna(-1).astype(int).astype(str)
    return X


# ---------------------------------------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------------------------------------
@dataclass
class EnsembleModel:
    lightgbm: TrainedModel
    catboost: Any                      # catboost.CatBoostRegressor
    xgboost: Any                       # xgboost.Booster
    weights: Dict[str, float]
    feature_names: List[str]
    categorical_features: List[str]
    categories: Dict[str, list]        # pinned training levels (persisted in the model card)

    def __post_init__(self) -> None:
        if set(self.weights) != set(MEMBERS):
            raise EnsembleError(f"ensemble weights must be given for exactly {MEMBERS}, got {sorted(self.weights)}")
        if any(w < 0 for w in self.weights.values()) or abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise EnsembleError(f"ensemble weights must be non-negative and sum to 1, got {self.weights}")
        names = {"lightgbm": list(self.lightgbm.feature_names), "catboost": list(self.catboost.feature_names_),
                 "xgboost": list(self.xgboost.feature_names or [])}
        for m, n in names.items():
            if n != list(self.feature_names):
                raise EnsembleError(f"{m} feature names/order differ from the ensemble feature list")

    @property
    def n_estimators(self) -> Dict[str, int]:
        return {"lightgbm": int(self.lightgbm.n_estimators), "catboost": int(self.catboost.tree_count_),
                "xgboost": int(self.xgboost.num_boosted_rounds())}


def _check_finite(name: str, raw: np.ndarray, n: int) -> None:
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (n,):
        raise EnsembleError(f"{name} returned shape {raw.shape}, expected ({n},)")
    if not np.isfinite(raw).all():
        raise EnsembleError(f"{name} produced NaN/inf predictions (refusing to repair silently)")


def _check(name: str, pred: np.ndarray, n: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (n,):
        raise EnsembleError(f"{name} returned shape {pred.shape}, expected ({n},)")
    if not np.isfinite(pred).all():
        raise EnsembleError(f"{name} produced NaN/inf predictions (refusing to repair silently)")
    if (pred < 0).any():
        raise EnsembleError(f"{name} produced negative demand after clipping")
    return pred


# ---------------------------------------------------------------------------------------------------------
# Training (final-fit recipe; same code path for back-test windows and the final model)
# ---------------------------------------------------------------------------------------------------------
def fit_ensemble(train_df: pd.DataFrame, cfg: Config) -> EnsembleModel:
    import xgboost as xgb
    from catboost import CatBoostRegressor, Pool

    e = cfg.model.ensemble
    feats = feature_columns(train_df)
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    levels = learn_categories(train_df, cats)
    y = train_df[TARGET_COL].to_numpy(dtype=float)

    lgbm = train_model(train_df, cfg)                                   # production LightGBM recipe
    assert lgbm.feature_names == feats and lgbm.categories == levels

    cat = CatBoostRegressor(random_seed=cfg.seed, allow_writing_files=False, **e.catboost_params)
    cat.fit(Pool(encode_catboost(train_df, feats, cats), np.log1p(y), cat_features=[feats.index(c) for c in cats]))

    dtrain = xgb.DMatrix(encode_xgboost(train_df, feats, cats, levels), label=y, enable_categorical=True)
    booster = xgb.train(dict(e.xgboost_params, seed=cfg.seed), dtrain, num_boost_round=int(e.xgboost_rounds),
                        verbose_eval=False)
    return EnsembleModel(lgbm, cat, booster, dict(e.weights), feats, cats, levels)


# ---------------------------------------------------------------------------------------------------------
# Inference (no training, no file access)
# ---------------------------------------------------------------------------------------------------------
def predict_members(model: EnsembleModel, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Each member's demand prediction for the rows of ``df`` (same rows, same order)."""
    import xgboost as xgb

    n = len(df)
    # Raw model outputs are validated BEFORE any transform / clip, so a NaN or +-inf can never be turned into a
    # plausible-looking number (expm1(-inf) = -1 would otherwise be clipped to 0 and silently "repaired").
    lg_raw = model.lightgbm.booster.predict(prepare_matrix(df, model.lightgbm.feature_names, model.lightgbm.categorical_features,
                                                           model.lightgbm.categories))
    cb_raw = model.catboost.predict(encode_catboost(df, model.feature_names, model.categorical_features))
    xg_raw = model.xgboost.predict(xgb.DMatrix(
        encode_xgboost(df, model.feature_names, model.categorical_features, model.categories), enable_categorical=True))
    for name, raw in (("LightGBM", lg_raw), ("CatBoost", cb_raw), ("XGBoost", xg_raw)):
        _check_finite(name, raw, n)
    lg = np.clip(lg_raw, 0.0, None)
    cb = np.clip(np.expm1(cb_raw), 0.0, None)          # archived behaviour: log1p target, expm1, clip at 0
    xg = np.clip(xg_raw, 0.0, None)
    return {"lightgbm": _check("LightGBM", lg, n), "catboost": _check("CatBoost", cb, n), "xgboost": _check("XGBoost", xg, n)}


def blend(members: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted sum of the members' outputs (raw demand space)."""
    lens = {k: len(v) for k, v in members.items()}
    if len(set(lens.values())) != 1:
        raise EnsembleError(f"members returned different lengths: {lens}")
    return sum(weights[k] * members[k] for k in MEMBERS)


def predict_ensemble(model: EnsembleModel, df: pd.DataFrame) -> np.ndarray:
    members = predict_members(model, df)
    return _check("ensemble", blend(members, model.weights), len(df))


def predict_contributions_members(model: EnsembleModel, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Exact TreeSHAP contributions per member, shape (n, n_features + 1) with the bias last.

    LightGBM / XGBoost: margin (log-link) space, ``exp(sum) == prediction``.
    CatBoost:           log1p-target space,      ``expm1(sum) == prediction``.
    """
    import xgboost as xgb
    from catboost import Pool

    idx = [model.feature_names.index(c) for c in model.categorical_features]
    return {
        "lightgbm": predict_contributions(model.lightgbm, df),
        "catboost": model.catboost.get_feature_importance(
            Pool(encode_catboost(df, model.feature_names, model.categorical_features), cat_features=idx), type="ShapValues"),
        "xgboost": model.xgboost.predict(xgb.DMatrix(
            encode_xgboost(df, model.feature_names, model.categorical_features, model.categories), enable_categorical=True),
            pred_contribs=True),
    }


# ---------------------------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_ensemble(model: EnsembleModel, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Write the three model files; return {member: {file, sha256, size_bytes}} for the model card."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model.lightgbm, out_dir / FILES["lightgbm"])
    model.catboost.save_model(str(out_dir / FILES["catboost"]))
    model.xgboost.save_model(str(out_dir / FILES["xgboost"]))
    return {m: {"file": FILES[m], "sha256": _sha256(out_dir / FILES[m]), "size_bytes": (out_dir / FILES[m]).stat().st_size}
            for m in MEMBERS}


def load_ensemble(model_dir: Path, card: Dict[str, Any]) -> EnsembleModel:
    """Load all three members ONCE (start-up), verifying every file against the hashes recorded in the card."""
    import xgboost as xgb
    from catboost import CatBoostRegressor

    model_dir = Path(model_dir)
    comps = {c["name"]: c for c in card.get("components", [])}
    if set(comps) != set(MEMBERS):
        raise EnsembleError(f"model card lists components {sorted(comps)}, expected {sorted(MEMBERS)}")
    for m in MEMBERS:
        path = model_dir / comps[m]["file"]
        if not path.exists():
            raise EnsembleError(f"missing model file for {HUMAN_NAME[m]}: {path.name}")
        if _sha256(path) != comps[m]["sha256"]:
            raise EnsembleError(f"{HUMAN_NAME[m]} file {path.name} does not match the hash in the model card")
    lgbm = load_model(model_dir / comps["lightgbm"]["file"], card["categories"], card["params"], card["n_estimators"])
    cat = CatBoostRegressor(); cat.load_model(str(model_dir / comps["catboost"]["file"]))
    booster = xgb.Booster(); booster.load_model(str(model_dir / comps["xgboost"]["file"]))
    return EnsembleModel(lgbm, cat, booster, {m: float(comps[m]["weight"]) for m in MEMBERS},
                         list(card["feature_names"]), list(card["categorical_features"]), card["categories"])
