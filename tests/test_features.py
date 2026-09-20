"""Feature semantics: what each feature means, and that history gaps are respected."""
import numpy as np
import pandas as pd

from src.features import build_supervised_dataset, feature_columns


def _origin_row(tables, hub, origin):
    o = tables.origin_feats
    return o[(o["HubID"] == hub) & (o["Date"] == pd.Timestamp(origin))].iloc[0]


def test_lag_semantics_are_calendar_days(tables):
    panel = tables.target_panel
    origin = pd.Timestamp("2014-08-20")
    row = _origin_row(tables, 1, origin)
    assert row["OV_lag1"] == panel.loc[origin, 1]                                   # value AT the origin
    assert row["OV_lag7"] == panel.loc[origin - pd.Timedelta(days=6), 1]            # 6 days earlier
    assert row["OV_lag14"] == panel.loc[origin - pd.Timedelta(days=13), 1]


def test_rolling_mean_and_expanding_mean_match_manual(tables):
    panel = tables.target_panel
    origin = pd.Timestamp("2014-08-20")
    window = panel.loc[origin - pd.Timedelta(days=6): origin, 2]
    row = _origin_row(tables, 2, origin)
    assert np.isclose(row["OV_rollmean7"], window.mean())
    assert np.isclose(row["OV_rollstd7"], window.std())
    assert np.isclose(row["OV_expanding_mean"], panel.loc[:origin, 2].mean())


def test_history_gap_yields_nan_not_a_shifted_row(tables, raw):
    """Hub 3 has no rows for a 40-day block. A row-offset shift would silently pull
    older values; calendar-aware lags must return NaN instead."""
    gap_dates = pd.date_range("2014-01-01", periods=140)[100:140]
    inside = gap_dates[5]
    assert pd.isna(_origin_row(tables, 3, inside)["OV_lag1"])
    later = gap_dates[-1] + pd.Timedelta(days=3)      # lag7 = 6 days back -> still inside the hole
    assert pd.isna(_origin_row(tables, 3, later)["OV_lag7"])
    assert not raw.train[(raw.train["HubID"] == 3) & raw.train["Date"].isin(gap_dates)].shape[0]


def test_hub_weekday_mean_uses_only_that_weekday_up_to_origin(tables):
    panel = tables.target_panel
    origin = pd.Timestamp("2014-08-20")
    for weekday in (1, 5, 6):
        hw = tables.hubweekday_feats
        got = hw[(hw["HubID"] == 1) & (hw["Origin"] == origin) & (hw["Weekday"] == weekday)].iloc[0]["OV_hubweekday_mean"]
        series = panel.loc[:origin, 1]
        expected = series[(series.index.dayofweek + 1) == weekday].mean()
        assert np.isclose(got, expected)


def test_schedule_neighbor_features(tables, raw):
    sched = raw.train.set_index(["HubID", "Date"])["PromoActive"]
    day = pd.Timestamp("2014-06-10")
    nb = tables.schedule_neighbor_feats
    row = nb[(nb["HubID"] == 1) & (nb["Date"] == day)].iloc[0]
    assert row["PromoActive_prev1"] == sched.loc[(1, day - pd.Timedelta(days=1))]
    assert row["PromoActive_next1"] == sched.loc[(1, day + pd.Timedelta(days=1))]


def test_supervised_rows_structure(tables, cfg, raw):
    origins = pd.date_range("2014-03-01", "2014-03-15", freq="7D")
    df = tables.make_dataset(origins, range(1, 43), require_label=True)
    assert (df["Date"] == df["Origin"] + pd.to_timedelta(df["h"], unit="D")).all()
    assert df["h"].between(1, 42).all()
    assert df["OrderVolume"].notna().all()
    # a label-free frame keeps every (hub, horizon) even without a target
    fut = tables.make_dataset([raw.train_max], range(1, 43), require_label=False)
    assert len(fut) == raw.train["HubID"].nunique() * 42
    assert fut["OrderVolume"].isna().all()


def test_feature_contract_excludes_target_and_train_only_columns(tables, raw):
    df = tables.make_dataset(pd.date_range("2014-03-01", periods=2, freq="7D"), range(1, 8), require_label=True)
    feats = feature_columns(df)
    assert "OrderVolume" not in feats and "Origin" not in feats and "Date" not in feats
    assert "AppSessions" not in df.columns                 # train-only column never enters the frame
    assert len(feats) == 44                                # pins the feature contract


def test_next_day_schedule_is_blank_when_it_would_not_be_published(tables, cfg):
    """The schedule is published `horizon` days ahead, so the flag for the day AFTER the last forecast
    day does not exist at inference. It must be missing in training/back-test rows too (h == horizon),
    and present for every earlier step."""
    horizon = cfg.forecast.horizon_days
    assert tables.schedule_horizon == horizon
    ds = tables.make_dataset([pd.Timestamp("2014-08-20")], range(1, horizon + 1), require_label=True)
    nxt = [c for c in ds.columns if c.endswith("_next1")]
    assert nxt
    last, earlier = ds[ds["h"] == horizon], ds[ds["h"] < horizon]
    assert last[nxt].isna().all().all()
    assert earlier[nxt].notna().all().all()
    prev = [c for c in ds.columns if c.endswith("_prev1")]
    assert ds[prev].notna().all().all()                      # backward-looking flags are always known


def test_masking_does_not_touch_features_when_disabled(tables, cfg):
    origins = [pd.Timestamp("2014-08-20")]
    masked = tables.make_dataset(origins, [cfg.forecast.horizon_days], require_label=True)
    raw_rows = build_supervised_dataset(
        tables.daily_attrs, tables.origin_feats, tables.hub_meta, origins, [cfg.forecast.horizon_days],
        require_label=True, hubweekday_feats=tables.hubweekday_feats,
        schedule_neighbor_feats=tables.schedule_neighbor_feats,              # no schedule_horizon
    )
    assert raw_rows["IsOpen_next1"].notna().any() and masked["IsOpen_next1"].isna().all()
