"""Categorical-encoding regression tests for the ensemble members (experiments/ensemble_corrected).

XGBoost 2.1 stores no category levels in the model: it consumes the integer codes of the pandas Categorical.
With per-frame ``astype("category")`` a single-store frame therefore gets different codes than the training
frame and predictions change (up to 21.6 % on the real data). These tests pin the required behaviour: with
levels pinned to the training levels, scoring one store alone equals scoring it inside the full frame.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("catboost")

from experiments.ensemble_corrected import recipes as R                      # noqa: E402
from src.features.builder import CATEGORICAL_FEATURES, feature_columns       # noqa: E402
from src.models.lightgbm_model import learn_categories                       # noqa: E402


@pytest.fixture(scope="module")
def frames(tables):
    origins = pd.date_range("2014-02-01", "2014-08-15", freq="7D")
    train = tables.make_dataset(origins, range(1, 43), require_label=True)
    score = tables.make_dataset([pd.Timestamp("2014-08-20")], range(1, 43), require_label=False)
    feats = feature_columns(train)
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    return train, score, feats, cats, learn_categories(train, cats)


@pytest.fixture(scope="module")
def xgb_model(frames, cfg):
    train, _, feats, cats, levels = frames
    # small model for speed; recipe otherwise identical (depth 8, mcw, subsample ... come from XGB_PARAMS)
    return R.fit_xgboost(train, feats, cats, levels, seed=cfg.seed, num_boost_round=60, min_child_weight=5)


def test_blend_weights_sum_to_one():
    assert sum(R.BLEND_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)
    assert R.BLEND_WEIGHTS == {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3}


def test_blend_is_the_weighted_sum():
    a, b, c = np.array([100.0, 0.0]), np.array([110.0, 0.0]), np.array([90.0, 0.0])
    np.testing.assert_allclose(R.blend(a, b, c), [0.6 * 100 + 0.1 * 110 + 0.3 * 90, 0.0])


def test_xgboost_single_store_equals_full_frame_with_pinned_levels(frames, xgb_model):
    _, score, feats, cats, levels = frames
    full = pd.Series(R.predict_xgboost(xgb_model, score, feats, cats, levels), index=score.index)
    for hub in score["HubID"].unique():
        sub = score[score["HubID"] == hub]
        np.testing.assert_allclose(R.predict_xgboost(xgb_model, sub, feats, cats, levels),
                                   full.loc[sub.index].to_numpy(), rtol=1e-9,
                                   err_msg=f"single-store scoring differs from full-frame for store {hub}")


def test_naive_per_frame_categories_reproduce_the_failure(frames, xgb_model):
    """Documents the trap: encoding each frame with its own levels changes the codes, hence the predictions."""
    import xgboost as xgb

    _, score, feats, cats, levels = frames
    full = pd.Series(R.predict_xgboost(xgb_model, score, feats, cats, levels), index=score.index)
    hub = sorted(score["HubID"].unique())[3]                 # a store whose training code is not 0
    sub = score[score["HubID"] == hub]
    naive = sub[feats].copy()
    for c in cats:
        naive[c] = naive[c].astype("category")               # per-frame levels (the archived behaviour)
    assert list(naive["HubID"].cat.codes.unique()) == [0]    # code collapses to 0 ...
    assert levels["HubID"].index(hub) != 0                   # ... but the training code was not 0
    bad = np.clip(xgb_model.predict(xgb.DMatrix(naive, enable_categorical=True)), 0, None)
    assert not np.allclose(bad, full.loc[sub.index].to_numpy(), rtol=1e-6)


def test_xgboost_full_frame_is_unaffected_by_pinning(frames, xgb_model):
    """When a frame contains every level, per-frame and pinned encodings coincide (why the old runs were valid)."""
    import xgboost as xgb

    train, score, feats, cats, levels = frames
    naive = score[feats].copy()
    for c in cats:
        naive[c] = naive[c].astype("category")
    assert all(list(naive[c].cat.categories) == levels[c] for c in cats)
    np.testing.assert_allclose(np.clip(xgb_model.predict(xgb.DMatrix(naive, enable_categorical=True)), 0, None),
                               R.predict_xgboost(xgb_model, score, feats, cats, levels), rtol=1e-12)


def test_xgboost_unseen_level_becomes_missing_not_a_crash(frames, xgb_model):
    _, score, feats, cats, levels = frames
    sub = score[score["HubID"] == score["HubID"].iloc[0]].copy()
    sub["HubID"] = 987654
    X = R.encode_xgboost(sub, feats, cats, levels)
    assert set(X["HubID"].cat.codes) == {-1}
    assert np.isfinite(R.predict_xgboost(xgb_model, sub, feats, cats, levels)).all()


def test_catboost_single_store_equals_full_frame(frames, cfg):
    train, score, feats, cats, _ = frames
    model = R.fit_catboost(train, feats, cats, seed=cfg.seed, iterations=40, depth=4)
    full = pd.Series(R.predict_catboost(model, score, feats, cats), index=score.index)
    for hub in score["HubID"].unique():
        sub = score[score["HubID"] == hub]
        np.testing.assert_allclose(R.predict_catboost(model, sub, feats, cats), full.loc[sub.index].to_numpy(), rtol=1e-9)
