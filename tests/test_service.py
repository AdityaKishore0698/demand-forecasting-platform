"""Inference service: on-demand features must equal the batch features used offline."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import apply_closed_hub_rule
from src.forecasting.service import ArtifactsNotFound, ForecastRequestError, ForecastService, StoreNotFound
from src.models import predict_raw


def test_serving_matches_offline_batch_prediction(service, tables, raw):
    """Train/serve skew guard: rebuild the same rows offline in one batch, score them
    with the model loaded from disk, and compare with the service response."""
    hub = 2
    batch = tables.make_dataset([raw.train_max], range(1, 43), require_label=False)
    batch = batch[batch["HubID"] == hub].sort_values("h")
    expected = apply_closed_hub_rule(predict_raw(service.model, batch), batch["IsOpen"])

    got = [p["predicted_demand"] for p in service.forecast(hub, 42)["forecast"]]
    np.testing.assert_allclose(got, expected, atol=0.06)        # response is rounded to 0.1


def test_forecast_shape_dates_and_business_rules(service, raw):
    out = service.forecast(1, 14)
    pts = out["forecast"]
    assert len(pts) == 14 and out["horizon"] == 14
    dates = [p["date"] for p in pts]
    assert dates[0] == (raw.train_max + pd.Timedelta(days=1)).date()
    assert all((b - a).days == 1 for a, b in zip(dates, dates[1:]))
    assert all(p["predicted_demand"] >= 0 for p in pts)
    assert all(p["predicted_demand"] == 0 for p in pts if not p["is_open"])          # closed => 0
    assert out["summary"]["open_days"] + out["summary"]["closed_days"] == 14


def test_shorter_horizon_is_a_prefix_of_longer_horizon(service):
    a = [p["predicted_demand"] for p in service.forecast(5, 7)["forecast"]]
    b = [p["predicted_demand"] for p in service.forecast(5, 21)["forecast"]][:7]
    assert a == b


def test_horizon_and_store_validation(service):
    with pytest.raises(ForecastRequestError):
        service.forecast(1, 0)
    with pytest.raises(ForecastRequestError):
        service.forecast(1, service.horizon_days + 1)     # never invents inputs beyond the known schedule
    with pytest.raises(StoreNotFound):
        service.forecast(9999, 7)


def test_explain_reconstructs_the_forecast_and_orders_drivers(service, raw):
    day = (raw.train_max + pd.Timedelta(days=3)).date()
    ex = service.explain(1, day, top_k=5)
    fc = {p["date"]: p for p in service.forecast(1, 7)["forecast"]}[day]
    assert ex["is_open"] == fc["is_open"]
    if fc["is_open"]:
        assert abs(ex["model_prediction"] - fc["predicted_demand"]) < 0.11
    effects = [abs(np.log1p(d["effect_pct"] / 100.0)) for d in ex["drivers"]]
    assert effects == sorted(effects, reverse=True) and len(ex["drivers"]) == 5
    with pytest.raises(ForecastRequestError):
        service.explain(1, raw.train_max.date())                   # not in the forecast window


def test_historical_marks_missing_dates_as_null(service):
    hist = service.historical(3, days=400)
    assert hist["summary"]["missing_days"] > 0                    # the 40-day hole in hub 3
    assert any(p["demand"] is None for p in hist["points"])
    assert hist["summary"]["last_history_date"] == service.origin.date()


def test_store_table_and_backtest(service):
    stores = service.list_stores()
    assert len(stores) == 6 and {s["store_id"] for s in stores} == set(range(1, 7))
    assert next(s for s in stores if s["store_id"] == 3)["has_history_gap"] is True
    assert all(0 < s["demand_percentile"] <= 100 for s in stores)
    net, hub = service.backtest(None), service.backtest(2)
    assert len(net["points"]) == 42 and len(hub["points"]) == 42
    assert hub["metrics"]["n_rows"] == 42


def test_missing_artifacts_raise_a_helpful_error(tmp_path):
    with pytest.raises(ArtifactsNotFound, match="make train"):
        ForecastService.load(tmp_path)


def test_last_forecast_day_uses_the_same_masked_schedule_as_training(service):
    """The deployed forecast for day 42 must be built exactly like a training/back-test row for h=42."""
    rows = service._features(1, list(range(1, service.horizon_days + 1)))
    nxt = [c for c in rows.columns if c.endswith("_next1")]
    assert rows.loc[rows["h"] == service.horizon_days, nxt].isna().all().all()
    assert rows.loc[rows["h"] < service.horizon_days, nxt].notna().all().all()
