import numpy as np

from src.evaluation.metrics import (
    apply_closed_hub_rule, mae, mape, regression_report, rmse, rmsle, rmspe, wape,
)


def test_rmsle_known_values():
    assert rmsle([10, 20], [10, 20]) == 0.0
    expected = np.sqrt(np.mean((np.log1p([12, 18]) - np.log1p([10, 20])) ** 2))
    assert np.isclose(rmsle([10, 20], [12, 18]), expected)
    assert np.isclose(rmsle([10], [-5]), rmsle([10], [0]))          # negatives are clipped to 0


def test_error_metrics_known_values():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert np.isclose(mae(y, p), 15.0)
    assert np.isclose(rmse(y, p), np.sqrt((100 + 400) / 2))
    assert np.isclose(wape(y, p), 30.0 / 300.0)
    assert np.isclose(mape(y, p), np.mean([0.10, 0.10]))
    assert np.isclose(rmspe(y, p), np.sqrt(np.mean([0.10 ** 2, 0.10 ** 2])))


def test_percentage_metrics_skip_zero_actuals():
    y, p = np.array([0.0, 100.0]), np.array([5.0, 110.0])
    assert np.isclose(mape(y, p), 0.10)
    assert np.isclose(rmspe(y, p), 0.10)


def test_closed_hub_rule():
    out = apply_closed_hub_rule([5.0, -3.0, 7.0], [1, 1, 0])
    assert out.tolist() == [5.0, 0.0, 0.0]


def test_report_only_scores_percentage_metrics_on_open_days():
    rep = regression_report([0, 100, 100], [0, 110, 90], [0, 1, 1])
    assert rep["n_rows"] == 3 and rep["n_open_rows"] == 2
    assert np.isclose(rep["wape"], 20 / 200)
    assert np.isclose(rep["bias"], 0.0)
