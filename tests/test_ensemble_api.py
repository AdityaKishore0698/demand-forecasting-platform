"""Service + HTTP behaviour of the final ensemble bundle (and the single-LightGBM rollback bundle)."""
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.evaluation.metrics import apply_closed_hub_rule
from src.forecasting import service as svc_module
from src.models import ensemble as E

pytest.importorskip("xgboost")
pytest.importorskip("catboost")


@pytest.fixture(scope="module")
def client(ens_artifact_dir):
    with TestClient(create_app(ens_artifact_dir)) as c:
        yield c


@pytest.fixture(scope="module")
def single_client(artifact_dir):
    with TestClient(create_app(artifact_dir)) as c:
        yield c


# ---- 9. health ---------------------------------------------------------------------------------------------------
def test_health_reports_the_ensemble_loaded(client):
    b = client.get("/health").json()
    assert b["status"] == "ok" and b["model_loaded"] is True
    assert b["model_type"] == "LightGBM + CatBoost + XGBoost ensemble"
    assert b["model_components"] == ["LightGBM", "CatBoost", "XGBoost"]
    assert b["data_label"] == "user-supplied"


def test_health_is_degraded_with_a_clear_reason_when_a_model_file_is_bad(ens_artifact_dir, tmp_path):
    bad = tmp_path / "bad"; shutil.copytree(ens_artifact_dir, bad)
    (bad / "model" / "lightgbm.txt.gz").write_bytes(b"corrupt")
    with TestClient(create_app(bad)) as c:
        b = c.get("/health").json()
        assert b["status"] == "degraded" and b["model_loaded"] is False and "hash" in b["detail"]
        assert c.post("/forecast", json={"store_id": 1, "horizon": 7}).status_code == 503


# ---- 10. model-info -------------------------------------------------------------------------------------------------
def test_model_info_describes_the_ensemble_honestly(client):
    b = client.get("/model-info").json()
    assert b["model_type"] == "LightGBM + CatBoost + XGBoost ensemble" and b["model_type_id"] == "boosting_ensemble"
    assert b["weights"] == {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3}
    assert [c["name"] for c in b["components"]] == ["lightgbm", "catboost", "xgboost"]
    assert [c["weight"] for c in b["components"]] == [0.6, 0.1, 0.3]
    assert all(c["library_version"] and c["sha256"] and c["size_bytes"] > 0 for c in b["components"])
    assert b["n_features"] == 44 and "ensemble" in b["algorithm"].lower()
    assert set(b["component_params"]) == {"lightgbm", "catboost", "xgboost"}


# ---- 11. forecast -------------------------------------------------------------------------------------------------
def test_forecast_endpoint_contract_and_values(client, ens_service):
    r = client.post("/forecast", json={"store_id": 3, "horizon": 14})
    assert r.status_code == 200
    b = r.json()
    assert b["model_type"] == "LightGBM + CatBoost + XGBoost ensemble" and len(b["forecast"]) == 14
    assert [p["horizon_step"] for p in b["forecast"]] == list(range(1, 15))
    assert all(p["predicted_demand"] >= 0 and np.isfinite(p["predicted_demand"]) for p in b["forecast"])
    assert all(p["predicted_demand"] == 0 for p in b["forecast"] if not p["is_open"])
    assert {"total_predicted", "peak", "open_days"} <= set(b["summary"])
    # the returned numbers are exactly the 0.6/0.1/0.3 blend of the three members, with the closed-day rule
    rows = ens_service._features(3, list(range(1, 15)))
    m = E.predict_members(ens_service.model, rows)
    expect = apply_closed_hub_rule(0.6 * m["lightgbm"] + 0.1 * m["catboost"] + 0.3 * m["xgboost"], rows["IsOpen"])
    np.testing.assert_allclose([p["predicted_demand"] for p in b["forecast"]], np.round(expect, 1), atol=0.051)


def test_validation_errors_are_unchanged(client):
    assert client.post("/forecast", json={"store_id": 3, "horizon": 99}).status_code == 422
    assert client.post("/forecast", json={"store_id": 999999, "horizon": 7}).status_code == 404
    assert client.post("/forecast", json={"horizon": 7}).status_code == 422


def test_shorter_horizon_is_a_prefix_of_a_longer_one(ens_service):
    a = [p["predicted_demand"] for p in ens_service.forecast(2, 7)["forecast"]]
    b = [p["predicted_demand"] for p in ens_service.forecast(2, 42)["forecast"]]
    assert a == b[:7]


def test_requests_never_load_or_train_models(ens_artifact_dir, monkeypatch):
    service = svc_module.ForecastService.load(ens_artifact_dir)                        # start-up: models loaded here
    def boom(*a, **k): raise AssertionError("model loading / training attempted inside a request")
    for name in ("load_ensemble", "load_model"):
        monkeypatch.setattr(svc_module, name, boom)
    monkeypatch.setattr(E, "fit_ensemble", boom); monkeypatch.setattr(E, "load_ensemble", boom)
    d = service.forecast(4, 10)["forecast"][0]["date"]
    assert service.forecast(5, 42) and service.explain(4, d)


# ---- integrity checks: nothing returned unless it matches the request ------------------------------------------------------------
def test_reordered_or_wrong_feature_rows_are_rejected(ens_service, monkeypatch):
    real = ens_service._features
    monkeypatch.setattr(ens_service, "_features", lambda s, h: real(s, h).iloc[::-1].reset_index(drop=True))
    with pytest.raises(svc_module.PredictionIntegrityError, match="horizons"):
        ens_service.forecast(1, 5)
    monkeypatch.setattr(ens_service, "_features", lambda s, h: real(s + 1, h))
    with pytest.raises(svc_module.PredictionIntegrityError, match="store"):
        ens_service.forecast(1, 6)


def test_a_bad_member_output_is_a_clear_500_not_a_number(ens_artifact_dir, monkeypatch):
    with TestClient(create_app(ens_artifact_dir), raise_server_exceptions=False) as c:
        monkeypatch.setattr(c.app.state.service.model.xgboost, "predict", lambda *a, **k: np.full(7, np.nan))
        r = c.post("/forecast", json={"store_id": 6, "horizon": 7})
        assert r.status_code == 500 and "XGBoost" in r.json()["detail"]


# ---- 12. explanations do not overclaim ------------------------------------------------------------------------------------
def test_explain_labels_the_ensemble_view_approximate_and_keeps_exact_per_model_views(client):
    d = client.post("/forecast", json={"store_id": 3, "horizon": 14}).json()["forecast"]
    day = next(p["date"] for p in d if p["is_open"])
    b = client.get("/explain", params={"store_id": 3, "date": day, "top_k": 5}).json()
    assert b["explanation_scope"] == "ensemble_approximate" and b["explanation_exactness"] == "approximate"
    assert "APPROXIMATE" in b["note"] and b["approximation"]["weights"] == {"lightgbm": 0.6, "catboost": 0.1, "xgboost": 0.3}
    assert abs(b["approximation"]["reconstruction_error_pct"]) < 5           # reported, and small for this bundle
    comps = {c["model"]: c for c in b["components"]}
    assert set(comps) == {"lightgbm", "catboost", "xgboost"}
    assert all(c["exactness"].startswith("exact TreeSHAP") and abs(c["reconstruction_error_pct"]) < 0.01 for c in comps.values())
    assert [c["weight"] for c in b["components"]] == [0.6, 0.1, 0.3]
    assert len(b["drivers"]) == 5 and all(len(c["drivers"]) == 5 for c in b["components"])
    # the blended forecast itself is the exact blend of the member predictions
    blend = sum(c["weight"] * c["prediction"] for c in b["components"])
    assert b["model_prediction"] == pytest.approx(blend, abs=0.2)


def test_explain_never_returns_an_unlabelled_single_model_explanation_for_the_ensemble(ens_service):
    day = ens_service.forecast(2, 7)["forecast"][0]["date"]
    out = ens_service.explain(2, day)
    assert out["explanation_scope"] != "lightgbm_exact" and out["components"]


def test_feature_importance_and_metrics_describe_the_ensemble(client):
    fi = client.get("/feature-importance").json()
    assert "ensemble" in fi["meta"]["model_type"].lower() and "APPROXIMATE" in fi["meta"]["shap"]
    assert "ONLY" in fi["meta"]["gain"] and "ENSEMBLE prediction" in fi["meta"]["permutation"]
    assert set(fi["shap_by_model"]) == {"lightgbm", "catboost", "xgboost"} and set(fi["gain_by_model"]) == {"lightgbm", "catboost", "xgboost"}
    m = client.get("/metrics").json()
    assert m["model_type"] == "boosting_ensemble" and m["model_label"].startswith("Ensemble")
    assert set(m["members"]) == {"lightgbm", "catboost", "xgboost"}
    assert all(set(w["members"]) == {"lightgbm", "catboost", "xgboost"} for w in m["windows"])
    assert any("blend weights" in n for n in m["validation_notes"])


# ---- rollback path stays a working, honestly-labelled single model ---------------------------------------------------------------
def test_single_lightgbm_bundle_still_loads_and_is_labelled_as_such(single_client):
    h = single_client.get("/health").json()
    assert h["model_type"] == "LightGBM (single model)" and h["model_components"] == ["LightGBM"]
    info = single_client.get("/model-info").json()
    assert info["model_type_id"] == "lightgbm_single" and "weights" not in info
    day = single_client.post("/forecast", json={"store_id": 3, "horizon": 7}).json()["forecast"][0]["date"]
    e = single_client.get("/explain", params={"store_id": 3, "date": day}).json()
    assert e["explanation_scope"] == "lightgbm_exact" and e["components"] is None
