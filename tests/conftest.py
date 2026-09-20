import pytest

from src.config import load_config
from src.features.builder import build_feature_tables
from src.forecasting.pipeline import run_pipeline
from src.forecasting.service import ForecastService
from tests.synthetic import make_synthetic_raw


@pytest.fixture(scope="session")
def cfg():
    """Tiny SINGLE-LightGBM config: exercises the rollback path (model.type = lightgbm_single)."""
    return load_config(overrides={
        "model": {"type": "lightgbm_single", "n_estimators": 40, "params": {"num_leaves": 15, "min_data_in_leaf": 20}},
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


# ---- final ensemble (LightGBM + CatBoost + XGBoost), same recipe with small round counts for speed ----------------------
@pytest.fixture(scope="session")
def ens_cfg():
    return load_config(overrides={
        "model": {"type": "boosting_ensemble", "n_estimators": 40, "params": {"num_leaves": 15, "min_data_in_leaf": 20},
                  "ensemble": {"catboost_params": {"iterations": 30, "depth": 4},
                               "xgboost_params": {"max_depth": 4, "min_child_weight": 5}, "xgboost_rounds": 30}},
        "validation": {"n_windows": 2},
        "interpretability": {"permutation_rows": 300, "permutation_repeats": 1, "shap_rows": 200},
    })


@pytest.fixture(scope="session")
def ens_tables(raw, ens_cfg):
    return build_feature_tables(raw, ens_cfg)


@pytest.fixture(scope="session")
def ens_artifact_dir(tmp_path_factory, raw, ens_cfg):
    out = tmp_path_factory.mktemp("ens_artifacts")
    run_pipeline(raw, ens_cfg, out)
    return out


@pytest.fixture(scope="session")
def ens_service(ens_artifact_dir):
    return ForecastService.load(ens_artifact_dir)
