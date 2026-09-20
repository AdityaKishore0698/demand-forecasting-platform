import copy

import numpy as np
import pandas as pd

from src.models import (
    load_model, predict_contributions, predict_raw, save_model, train_model,
)


def _train_frame(tables):
    origins = pd.date_range("2014-02-01", "2014-08-15", freq="7D")
    return tables.make_dataset(origins, range(1, 43), require_label=True)


def test_train_and_predict_shapes_and_bounds(tables, cfg):
    df = _train_frame(tables)
    model = train_model(df, cfg)
    pred = predict_raw(model, df)
    assert pred.shape == (len(df),)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    assert len(model.feature_names) == 44


def test_save_load_roundtrip_gives_identical_predictions(tables, cfg, tmp_path):
    df = _train_frame(tables)
    model = train_model(df, cfg)
    save_model(model, tmp_path / "m.txt.gz")
    reloaded = load_model(tmp_path / "m.txt.gz", model.categories, model.params, model.n_estimators)
    np.testing.assert_allclose(predict_raw(model, df), predict_raw(reloaded, df), rtol=1e-12)


def test_single_hub_prediction_equals_slice_of_full_prediction(tables, cfg):
    """Categorical levels are pinned, so scoring one hub alone must equal scoring it
    inside the full frame (guards against a serving-time re-coding bug)."""
    df = _train_frame(tables)
    model = train_model(df, cfg)
    full = pd.Series(predict_raw(model, df), index=df.index)
    subset = df[df["HubID"] == 4]
    np.testing.assert_allclose(predict_raw(model, subset), full.loc[subset.index].to_numpy(), rtol=1e-12)


def test_contributions_reconstruct_the_prediction(tables, cfg):
    df = _train_frame(tables).sample(200, random_state=0)
    model = train_model(df, cfg)
    contrib = predict_contributions(model, df)
    np.testing.assert_allclose(np.exp(contrib.sum(axis=1)), predict_raw(model, df), rtol=1e-6)


def test_closed_days_are_learned_as_near_zero(tables, cfg):
    """The Tweedie/log-link model can represent the zero-inflated structure: with enough
    learning it predicts almost nothing on closed days. (A faster learning rate is used
    because the shared test config trains only 40 trees.)"""
    fast = copy.deepcopy(cfg)
    fast.model.params["learning_rate"] = 0.3
    df = _train_frame(tables)
    model = train_model(df, fast)
    closed = df[df["IsOpen"] == 0]
    assert predict_raw(model, closed).mean() < 0.05 * predict_raw(model, df[df["IsOpen"] == 1]).mean()
