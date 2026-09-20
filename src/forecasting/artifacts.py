"""On-disk layout of the trained-model bundle (written by the pipeline, read by the API)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_artifact_dir() -> Path:
    """Where the API looks for a model bundle when ``ARTIFACT_DIR`` is not set.

    ``artifacts/`` (trained locally on your own data; never committed) wins if it holds a
    model, otherwise the committed synthetic demo bundle in ``artifacts/demo/`` is used.
    """
    local = REPO_ROOT / "artifacts"
    return local if (local / "model" / "model_card.json").exists() else local / "demo"


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    # model/
    @property
    def model_file(self) -> Path: return self.root / "model" / "lgbm_tweedie.txt.gz"
    @property
    def model_card(self) -> Path: return self.root / "model" / "model_card.json"
    @property
    def model_dir(self) -> Path: return self.root / "model"
    # ensemble members (model_type == "boosting_ensemble")
    @property
    def lightgbm_file(self) -> Path: return self.root / "model" / "lightgbm.txt.gz"
    @property
    def catboost_file(self) -> Path: return self.root / "model" / "catboost.cbm"
    @property
    def xgboost_file(self) -> Path: return self.root / "model" / "xgboost.ubj"

    # serving/  (compact snapshot needed to build features for a forecast on request)
    @property
    def hub_metadata(self) -> Path: return self.root / "serving" / "hub_metadata.csv"
    @property
    def history(self) -> Path: return self.root / "serving" / "history.csv.gz"
    @property
    def origin_features(self) -> Path: return self.root / "serving" / "origin_features.csv.gz"
    @property
    def hubweekday_features(self) -> Path: return self.root / "serving" / "hubweekday_features.csv.gz"
    @property
    def schedule(self) -> Path: return self.root / "serving" / "schedule.csv.gz"

    # reports/
    @property
    def metrics(self) -> Path: return self.root / "reports" / "metrics.json"
    @property
    def backtest_predictions(self) -> Path: return self.root / "reports" / "backtest_predictions.csv.gz"
    @property
    def importance(self) -> Path: return self.root / "reports" / "feature_importance.json"
    @property
    def dataset_summary(self) -> Path: return self.root / "reports" / "dataset_summary.json"

    def required(self, model_type: str = "lightgbm_single") -> List[Path]:
        model_files = ([self.lightgbm_file, self.catboost_file, self.xgboost_file]
                       if model_type == "boosting_ensemble" else [self.model_file])
        return model_files + [
            self.model_card, self.hub_metadata, self.history,
            self.origin_features, self.hubweekday_features, self.schedule, self.metrics,
            self.backtest_predictions, self.importance, self.dataset_summary,
        ]

    def missing(self, model_type: str = "lightgbm_single") -> List[Path]:
        return [p for p in self.required(model_type) if not p.exists()]
