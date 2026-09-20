"""Production 3-model ensemble (src/models/ensemble.py): loading, blend, validation, category safety, persistence."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.forecasting.service import ArtifactsNotFound, ForecastService, ModelLoadError
from src.models import ensemble as E
from src.models.dispatch import predict as predict_any

pytest.importorskip("xgboost")
pytest.importorskip("catboost")


@pytest.fixture(scope="module")
def model(ens_service):
    return ens_service.model


@pytest.fixture(scope="module")
def score_rows(ens_tables, ens_service):
    return ens_tables.make_dataset([ens_service.origin], range(1, 43), require_label=False)


# ---- recipes: production config == validated experiment recipes --------------------------------------------------
def test_config_recipes_match_the_validated_experiment():
    from experiments.ensemble_corrected import recipes as R
    from src.config import load_config

    e = load_config().model.ensemble
    assert e.weights == R.BLEND_WEIGHTS == {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3}
    assert e.catboost_params == R.CATBOOST_PARAMS
    assert e.xgboost_params == R.XGB_PARAMS and e.xgboost_rounds == R.XGB_NUM_BOOST_ROUND
    assert load_config().model.n_estimators == 1160


def test_config_rejects_bad_weights():
    from src.config import load_config

    with pytest.raises(ValueError):
        load_config(overrides={"model": {"ensemble": {"weights": {"lightgbm": 0.5, "catboost": 0.1, "xgboost": 0.3}}}})


# ---- 1. all three models load; 2. weights ------------------------------------------------------------------------
def test_all_three_models_are_loaded_at_startup(ens_service, ens_artifact_dir):
    m = ens_service.model
    assert isinstance(m, E.EnsembleModel)
    assert all(n > 0 for n in m.n_estimators.values()) and set(m.n_estimators) == set(E.MEMBERS)
    for f in E.FILES.values():
        assert (ens_artifact_dir / "model" / f).exists()
    assert ens_service.model_type_id == "boosting_ensemble"


def test_ensemble_weights_sum_to_one_and_are_persisted(model, ens_service):
    assert sum(model.weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert model.weights == {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3} == ens_service.card["weights"]
    with pytest.raises(E.EnsembleError):
        E.EnsembleModel(model.lightgbm, model.catboost, model.xgboost, {"lightgbm": 0.5, "catboost": 0.1, "xgboost": 0.3},
                        model.feature_names, model.categorical_features, model.categories)


# ---- 3./4. shape and ordering; 5. blend correctness; 6. NaN / inf ---------------------------------------------------
def test_members_have_matching_shape_and_row_order(model, score_rows):
    members = E.predict_members(model, score_rows)
    assert set(members) == set(E.MEMBERS)
    assert all(v.shape == (len(score_rows),) for v in members.values())
    rev = E.predict_members(model, score_rows.iloc[::-1])                       # nothing is silently reordered
    for k in members:
        np.testing.assert_allclose(rev[k], members[k][::-1], rtol=1e-12)


def test_weighted_blend_is_exactly_the_weighted_sum(model, score_rows):
    m = E.predict_members(model, score_rows)
    np.testing.assert_allclose(E.predict_ensemble(model, score_rows), 0.6 * m["lightgbm"] + 0.1 * m["catboost"] + 0.3 * m["xgboost"], rtol=1e-12)
    np.testing.assert_allclose(predict_any(model, score_rows), E.predict_ensemble(model, score_rows), rtol=0, atol=0)
    a, b, c = np.array([100.0, 0.0]), np.array([110.0, 0.0]), np.array([90.0, 0.0])
    np.testing.assert_allclose(E.blend({"lightgbm": a, "catboost": b, "xgboost": c}, model.weights), [98.0, 0.0])


def test_outputs_are_finite_and_non_negative(model, score_rows):
    for p in list(E.predict_members(model, score_rows).values()) + [E.predict_ensemble(model, score_rows)]:
        assert np.isfinite(p).all() and (p >= 0).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_from_a_member_is_rejected_not_repaired(model, score_rows, monkeypatch, bad):
    monkeypatch.setattr(model.catboost, "predict", lambda X: np.full(len(X), bad))
    with pytest.raises(E.EnsembleError, match="CatBoost"):
        E.predict_ensemble(model, score_rows)


def test_wrong_shape_and_length_mismatch_are_rejected(model, score_rows, monkeypatch):
    monkeypatch.setattr(model.catboost, "predict", lambda X: np.zeros(len(X) - 1))
    with pytest.raises(E.EnsembleError, match="shape"):
        E.predict_members(model, score_rows)
    with pytest.raises(E.EnsembleError, match="lengths"):
        E.blend({"lightgbm": np.zeros(3), "catboost": np.zeros(3), "xgboost": np.zeros(2)}, model.weights)


# ---- 7./8. pinned XGBoost categories; single-store == full-frame ----------------------------------------------------
def test_xgboost_uses_the_persisted_training_levels(model, ens_service, score_rows):
    assert model.categories == ens_service.card["categories"]                      # levels persisted in the model card
    X = E.encode_xgboost(score_rows[score_rows["HubID"] == 4], model.feature_names, model.categorical_features, model.categories)
    for c in model.categorical_features:
        assert list(X[c].cat.categories) == model.categories[c]                    # pinned, not re-derived from the frame
    assert set(X["HubID"].cat.codes) == {model.categories["HubID"].index(4)}       # training code, never 0 by accident


def test_single_store_prediction_equals_full_frame_for_every_member_and_the_blend(model, score_rows):
    full_m = E.predict_members(model, score_rows)
    full_e = pd.Series(E.predict_ensemble(model, score_rows), index=score_rows.index)
    for hub in score_rows["HubID"].unique():
        sub = score_rows[score_rows["HubID"] == hub]
        for k, v in E.predict_members(model, sub).items():
            np.testing.assert_allclose(v, pd.Series(full_m[k], index=score_rows.index).loc[sub.index].to_numpy(), rtol=1e-9)
        np.testing.assert_allclose(E.predict_ensemble(model, sub), full_e.loc[sub.index].to_numpy(), rtol=1e-9)


def test_naive_recoding_would_change_predictions_but_the_implementation_never_does_it(model, score_rows):
    import xgboost as xgb

    hub = sorted(score_rows["HubID"].unique())[3]
    sub = score_rows[score_rows["HubID"] == hub]
    naive = sub[model.feature_names].copy()
    for c in model.categorical_features:
        naive[c] = naive[c].astype("category")                                  # the trap: per-frame levels
    assert set(naive["HubID"].cat.codes) == {0} and model.categories["HubID"].index(hub) != 0
    bad = np.clip(model.xgboost.predict(xgb.DMatrix(naive, enable_categorical=True)), 0, None)
    good = E.predict_members(model, sub)["xgboost"]
    assert not np.allclose(bad, good, rtol=1e-6)                                # the failure is real for this model ...
    full = pd.Series(E.predict_members(model, score_rows)["xgboost"], index=score_rows.index)
    np.testing.assert_allclose(good, full.loc[sub.index].to_numpy(), rtol=1e-9)  # ... and the implementation avoids it


def test_unseen_categories_are_handled_safely_by_every_member(model, score_rows):
    sub = score_rows[score_rows["HubID"] == score_rows["HubID"].iloc[0]].copy()
    sub["HubID"] = 987654
    X = E.encode_xgboost(sub, model.feature_names, model.categorical_features, model.categories)
    assert set(X["HubID"].cat.codes) == {-1}                                    # missing, never a recycled code
    members = E.predict_members(model, sub)                                     # no crash, finite, non-negative
    assert all(np.isfinite(v).all() and (v >= 0).all() for v in members.values())


# ---- persistence ---------------------------------------------------------------------------------------------------------
def test_save_load_roundtrip_gives_identical_predictions(model, ens_service, score_rows, tmp_path):
    files = E.save_ensemble(model, tmp_path)
    card = dict(ens_service.card)
    card["components"] = [dict(c, sha256=files[c["name"]]["sha256"]) for c in card["components"]]
    reloaded = E.load_ensemble(tmp_path, card)
    np.testing.assert_allclose(E.predict_ensemble(reloaded, score_rows), E.predict_ensemble(model, score_rows), rtol=1e-12)


def test_tampered_or_missing_model_files_fail_clearly(ens_artifact_dir, tmp_path):
    bad = tmp_path / "tampered"; shutil.copytree(ens_artifact_dir, bad)
    (bad / "model" / "xgboost.ubj").write_bytes(b"not a model")
    with pytest.raises(ModelLoadError, match="hash"):
        ForecastService.load(bad)
    gone = tmp_path / "missing"; shutil.copytree(ens_artifact_dir, gone)
    (gone / "model" / "catboost.cbm").unlink()
    with pytest.raises(ArtifactsNotFound, match="catboost.cbm"):
        ForecastService.load(gone)


def test_contributions_are_exact_per_model(model, score_rows):
    sample = score_rows.iloc[:20]
    m = E.predict_members(model, sample)
    c = E.predict_contributions_members(model, sample)
    np.testing.assert_allclose(np.exp(c["lightgbm"].sum(1)), m["lightgbm"], rtol=1e-6)
    np.testing.assert_allclose(np.exp(c["xgboost"].sum(1)), m["xgboost"], rtol=1e-4)
    np.testing.assert_allclose(np.expm1(c["catboost"].sum(1)), m["catboost"], rtol=1e-4)
