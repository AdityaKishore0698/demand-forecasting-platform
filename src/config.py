"""Centralised, typed configuration.

All tunable behaviour (seed, horizon, feature definitions, model parameters, paths)
is read from ``configs/default.yaml`` into the dataclasses below, so a training run
is fully described by one file plus the raw data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    artifact_dir: str = "artifacts"


@dataclass
class DataConfig:
    train_file: str = "orders_train.csv"
    test_file: str = "orders_test.csv"
    hub_metadata_file: str = "hub_metadata.csv"
    label: str = "user-supplied"     # "synthetic-demo" marks generated data (shown in the UI and the API)


@dataclass
class ForecastConfig:
    horizon_days: int = 42
    origin_stride_days: int = 7
    min_history_days: int = 28


@dataclass
class FeatureConfig:
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 21, 28, 35, 42])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 42])
    schedule_shifts: List[int] = field(default_factory=lambda: [-1, 1])


@dataclass
class ValidationConfig:
    n_windows: int = 3


@dataclass
class ModelConfig:
    n_estimators: int = 1160
    params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "tweedie",
        "tweedie_variance_power": 1.05,
        "metric": "tweedie",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "lambda_l2": 1.0,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
    })


@dataclass
class InterpretabilityConfig:
    permutation_rows: int = 15000
    permutation_repeats: int = 3
    shap_rows: int = 10000


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)

    # -- resolved locations -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return _resolve(self.paths.raw_dir)

    @property
    def artifact_dir(self) -> Path:
        return _resolve(self.paths.artifact_dir)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _merge(dc: Any, values: Dict[str, Any]) -> Any:
    """Recursively overlay ``values`` onto dataclass instance ``dc``."""
    for f in fields(dc):
        if f.name not in values:
            continue
        current = getattr(dc, f.name)
        new = values[f.name]
        if is_dataclass(current) and isinstance(new, dict):
            _merge(current, new)
        elif isinstance(current, dict) and isinstance(new, dict):
            merged = dict(current)
            merged.update(new)
            setattr(dc, f.name, merged)
        else:
            setattr(dc, f.name, new)
    unknown = set(values) - {f.name for f in fields(dc)}
    if unknown:
        raise KeyError(f"Unknown config keys for {type(dc).__name__}: {sorted(unknown)}")
    return dc


def load_config(path: Optional[os.PathLike] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration from YAML (defaults to ``configs/default.yaml`` or ``$DF_CONFIG``)."""
    cfg_path = Path(path or os.environ.get("DF_CONFIG") or DEFAULT_CONFIG_PATH)
    cfg = Config()
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge(cfg, overrides)
    if cfg.forecast.horizon_days < 1:
        raise ValueError("forecast.horizon_days must be >= 1")
    if cfg.validation.n_windows < 1:
        raise ValueError("validation.n_windows must be >= 1")
    return cfg
