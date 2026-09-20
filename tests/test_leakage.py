"""No future information may reach a feature, a training row, or a prediction."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    BacktestWindow, LeakageError, assert_no_temporal_leakage, make_backtest_windows,
    run_backtest_window, training_origins,
)
from src.features.builder import build_feature_tables


def _scramble_after(raw, cutoff):
    """Copy of the data where every target AFTER ``cutoff`` is wildly different."""
    other = copy.deepcopy(raw)
    mask = other.train["Date"] > cutoff
    other.train.loc[mask, "OrderVolume"] = other.train.loc[mask, "OrderVolume"] * 37 + 999
    return other


def test_origin_features_are_causal(raw, cfg, tables):
    cutoff = pd.Timestamp("2014-08-01")
    tables2 = build_feature_tables(_scramble_after(raw, cutoff), cfg)

    a = tables.origin_feats[tables.origin_feats["Date"] <= cutoff].reset_index(drop=True)
    b = tables2.origin_feats[tables2.origin_feats["Date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)                       # lags / rolling / expanding unchanged

    a = tables.hubweekday_feats[tables.hubweekday_feats["Origin"] <= cutoff].reset_index(drop=True)
    b = tables2.hubweekday_feats[tables2.hubweekday_feats["Origin"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)                       # hub-weekday baseline unchanged


def test_training_targets_must_precede_validation_window(tables, raw, cfg):
    window = make_backtest_windows(raw.train_max, cfg.forecast.horizon_days, 1)[0]
    ok_origins = training_origins(tables, window.max_train_origin, cfg)
    ok = tables.make_dataset(ok_origins, range(1, 43), require_label=True)
    assert_no_temporal_leakage(ok, window)                    # passes

    too_late = training_origins(tables, window.cutoff, cfg)   # origins allowed to run up to the cutoff
    bad = tables.make_dataset(too_late, range(1, 43), require_label=True)
    with pytest.raises(LeakageError):
        assert_no_temporal_leakage(bad, window)


def test_backtest_predictions_do_not_depend_on_window_actuals(tables, raw, cfg):
    """End-to-end: rewrite all targets inside the validation window, retrain, and the
    predictions for that window must be identical (the model never saw them)."""
    window = make_backtest_windows(raw.train_max, cfg.forecast.horizon_days, 1)[0]
    det_cfg = copy.deepcopy(cfg)
    det_cfg.model.params.update({"deterministic": True, "num_threads": 1, "force_row_wise": True})

    res_a, _ = run_backtest_window(window, tables, det_cfg)
    tables_b = build_feature_tables(_scramble_after(raw, window.cutoff), det_cfg)
    res_b, _ = run_backtest_window(window, tables_b, det_cfg)

    a = res_a.rows.sort_values(["HubID", "h"])["predicted"].to_numpy()
    b = res_b.rows.sort_values(["HubID", "h"])["predicted"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-9)
    # ...while the *actuals* really did change, so the test is meaningful
    assert not np.allclose(res_a.rows["actual"].to_numpy().sum(), res_b.rows["actual"].to_numpy().sum())


def test_windows_are_chronological_and_non_overlapping(raw, cfg):
    windows = make_backtest_windows(raw.train_max, 42, 3)
    assert windows[0].end == raw.train_max
    for newer, older in zip(windows, windows[1:]):
        assert older.end < newer.start
