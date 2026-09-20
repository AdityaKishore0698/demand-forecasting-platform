import json

import numpy as np

from src.config import load_config
from src.forecasting.artifacts import ArtifactPaths


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg.seed == 42 and cfg.forecast.horizon_days == 42
    assert cfg.model.params["objective"] == "tweedie"


def test_pipeline_writes_the_full_artifact_bundle(artifact_dir):
    assert ArtifactPaths(artifact_dir).missing() == []


def test_metrics_report_is_complete_and_model_beats_baselines(artifact_dir):
    m = json.loads((artifact_dir / "reports" / "metrics.json").read_text())
    for key in ("rmsle", "mae", "rmse", "wape", "rmspe", "mape"):
        assert np.isfinite(m["model"][key])
    assert len(m["windows"]) == 2 and m["windows"][0]["name"] == "primary"
    assert m["model"]["rmsle"] < m["baselines"][m["best_baseline"]]["rmsle"]
    assert m["improvement_vs_best_baseline_pct"]["rmsle"] > 0
    # training and validation are reported separately
    assert m["training_in_sample"]["rmsle"] is not None
    assert {"by_horizon", "by_weekday", "by_hub_tier"} <= set(m["breakdowns"])


def test_model_card_pins_the_serving_contract(artifact_dir):
    card = json.loads((artifact_dir / "model" / "model_card.json").read_text())
    assert card["seed"] == 42 and card["horizon_days"] == 42
    assert len(card["feature_names"]) == 44
    assert set(card["categories"]) == set(card["categorical_features"])


def test_importance_report_has_three_views_with_labels(artifact_dir):
    imp = json.loads((artifact_dir / "reports" / "feature_importance.json").read_text())
    for view in ("gain", "permutation", "shap"):
        assert len(imp[view]) == 44
        assert {"feature", "label", "group_label", "value", "share"} <= set(imp[view][0])
