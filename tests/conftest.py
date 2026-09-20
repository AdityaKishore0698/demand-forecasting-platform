import pytest

from src.config import load_config
from src.features.builder import build_feature_tables
from src.forecasting.pipeline import run_pipeline
from src.forecasting.service import ForecastService
from tests.synthetic import make_synthetic_raw


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={
        "model": {"n_estimators": 40, "params": {"num_leaves": 15, "min_data_in_leaf": 20}},
        "validation": {"n_windows": 2},
        "interpretability": {"permutation_rows": 400, "permutation_repeats": 1, "shap_rows": 300},
    })


@pytest.fixture(scope="session")
def raw():
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def tables(raw, cfg):
    return build_feature_tables(raw, cfg)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, raw, cfg):
    out = tmp_path_factory.mktemp("artifacts")
    run_pipeline(raw, cfg, out)
    return out


@pytest.fixture(scope="session")
def service(artifact_dir):
    return ForecastService.load(artifact_dir)
