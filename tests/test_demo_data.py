"""The synthetic demo dataset must be loadable by the real pipeline and must be labelled as synthetic."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import RawData, validate_raw
from src.data.synthetic import MARKER, make_synthetic_frames, write_synthetic_csvs
from src.forecasting.artifacts import default_artifact_dir


def test_synthetic_frames_pass_pipeline_validation():
    train, future, meta = make_synthetic_frames(n_stores=12, n_train_days=200)
    cfg = load_config()
    validate_raw(RawData(train=train, future=future, hub_meta=meta), cfg)
    assert (train.loc[train["IsOpen"] == 0, "OrderVolume"] == 0).all()
    assert "OrderVolume" not in future.columns and "AppSessions" not in future.columns


def test_synthetic_generation_is_deterministic():
    a = make_synthetic_frames(n_stores=6, n_train_days=120, seed=3)[0]
    b = make_synthetic_frames(n_stores=6, n_train_days=120, seed=3)[0]
    c = make_synthetic_frames(n_stores=6, n_train_days=120, seed=4)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not np.array_equal(a["OrderVolume"].to_numpy(), c["OrderVolume"].to_numpy())


def test_written_csvs_carry_a_synthetic_marker(tmp_path):
    out = write_synthetic_csvs(tmp_path, n_stores=6, n_train_days=100)
    assert "SYNTHETIC" in (out / MARKER).read_text()
    assert {"orders_train.csv", "orders_test.csv", "hub_metadata.csv"} <= {p.name for p in out.iterdir()}


def test_default_artifact_dir_is_a_bundle_location():
    d = default_artifact_dir()
    assert d.name in {"artifacts", "demo"}
