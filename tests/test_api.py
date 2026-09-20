"""HTTP contract: status codes, payload shapes, validation, error handling."""
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def client(artifact_dir):
    with TestClient(create_app(artifact_dir)) as c:      # context manager runs the lifespan (loads model once)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["forecast_horizon_days"] == 42


def test_stores(client):
    r = client.get("/stores")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 6
    assert {"store_id", "avg_open_day_demand", "demand_percentile", "backtest_wape"} <= set(body["stores"][0])


def test_forecast_success(client):
    r = client.post("/forecast", json={"store_id": 2, "horizon": 14})
    assert r.status_code == 200
    body = r.json()
    assert body["store_id"] == 2 and len(body["forecast"]) == 14
    assert all(p["predicted_demand"] >= 0 for p in body["forecast"])
    assert {"total_predicted", "peak", "lowest_open_day", "mean_open_day"} <= set(body["summary"])
    assert any("prediction intervals" in n for n in body["notes"])       # honest about no intervals


def test_forecast_accepts_numeric_string_store_id(client):
    assert client.post("/forecast", json={"store_id": "2", "horizon": 7}).status_code == 200


@pytest.mark.parametrize("payload", [
    {"store_id": 2, "horizon": 0},
    {"store_id": 2, "horizon": -3},
    {"store_id": 2, "horizon": "abc"},
    {"store_id": 0, "horizon": 7},
    {"horizon": 7},
    {"store_id": "x", "horizon": 7},
])
def test_forecast_rejects_invalid_input_with_422(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422


def test_forecast_horizon_beyond_known_schedule_is_refused(client):
    r = client.post("/forecast", json={"store_id": 2, "horizon": 43})
    assert r.status_code == 422 and "between 1 and 42" in r.json()["detail"]


def test_unknown_store_is_404(client):
    r = client.post("/forecast", json={"store_id": 424242, "horizon": 7})
    assert r.status_code == 404 and "Unknown store_id" in r.json()["detail"]
    assert client.get("/historical-data", params={"store_id": 424242}).status_code == 404


def test_historical_data(client):
    r = client.get("/historical-data", params={"store_id": 3, "days": 300})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["missing_days"] > 0 and body["points"][0]["date"] == body["start"]
    assert client.get("/historical-data", params={"store_id": 3, "days": 0}).status_code == 422


def test_metrics_model_info_and_feature_importance(client):
    m = client.get("/metrics").json()
    assert m["model"]["rmsle"] > 0 and "baselines" in m and len(m["windows"]) == 2
    info = client.get("/model-info").json()
    assert info["n_features"] == 44 and info["horizon_days"] == 42 and info["seed"] == 42
    imp = client.get("/feature-importance").json()
    assert imp["gain"][0]["share"] >= imp["gain"][-1]["share"]


def test_backtest_and_explain(client):
    bt = client.get("/backtest", params={"store_id": 1}).json()
    assert len(bt["points"]) == 42 and bt["scope"] == "Hub 1"
    assert client.get("/backtest").json()["scope"].startswith("All stores")
    first_day = client.post("/forecast", json={"store_id": 1, "horizon": 3}).json()["forecast"][0]["date"]
    ex = client.get("/explain", params={"store_id": 1, "date": first_day, "top_k": 4})
    assert ex.status_code == 200 and len(ex.json()["drivers"]) == 4
    assert client.get("/explain", params={"store_id": 1, "date": "1999-01-01"}).status_code == 422


def test_service_unavailable_when_artifacts_missing(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        r = c.get("/stores")
        assert r.status_code == 503 and "make train" in r.json()["detail"]


def test_health_reports_data_label(client):
    body = client.get("/health").json()
    assert "data_label" in body      # provenance tag written by the pipeline (None for older bundles)
