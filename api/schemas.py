"""Request / response models (Pydantic v2). These are the API contract."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_version: Optional[str] = None
    data_through: Optional[date] = Field(None, description="Last day of known history")
    forecast_horizon_days: Optional[int] = None
    data_label: Optional[str] = Field(None, description="Provenance tag, e.g. 'synthetic-demo'")
    detail: Optional[str] = None


class StoreInfo(BaseModel):
    store_id: int
    hub_format: Optional[int] = None
    assortment_tier: Optional[int] = None
    competitor_distance: Optional[float] = None
    loyalty_program: bool
    avg_open_day_demand: float = Field(..., description="Mean orders on days the hub was open")
    history_days: int
    has_history_gap: bool = Field(..., description="True if the hub has missing dates in its history")
    demand_rank: int = Field(..., description="1 = highest average demand among all stores")
    demand_percentile: float
    backtest_wape: Optional[float] = Field(None, description="Hold-out WAPE for this hub (open days)")


class StoresResponse(BaseModel):
    count: int
    stores: List[StoreInfo]


class HistoricalPoint(BaseModel):
    date: date
    demand: Optional[int] = Field(None, description="null when no row was recorded for that date")
    is_open: Optional[bool] = None
    promo_active: Optional[bool] = None


class HistorySummary(BaseModel):
    recorded_days: int
    missing_days: int
    open_days: int
    avg_open_day_demand: Optional[float] = None
    first_history_date: date
    last_history_date: date


class HistoricalResponse(BaseModel):
    store_id: int
    start: date
    end: date
    points: List[HistoricalPoint]
    summary: HistorySummary


class ForecastRequest(BaseModel):
    store_id: int = Field(..., ge=1, description="Hub / store identifier", examples=[42])
    horizon: int = Field(14, ge=1, le=366, description="Days ahead to forecast (max = model horizon, 42)", examples=[14])


class ForecastPoint(BaseModel):
    date: date
    weekday: str
    horizon_step: int
    predicted_demand: float
    is_open: bool
    promo_active: bool
    holiday: bool
    school_closure: bool


class PeakPoint(BaseModel):
    date: date
    value: float


class ForecastSummary(BaseModel):
    total_predicted: float
    mean_open_day: Optional[float] = None
    peak: Optional[PeakPoint] = None
    lowest_open_day: Optional[PeakPoint] = None
    open_days: int
    closed_days: int
    promo_days: int


class ForecastResponse(BaseModel):
    store_id: int
    horizon: int
    origin_date: date = Field(..., description="Last known day the forecast is anchored to")
    model_version: str
    forecast: List[ForecastPoint]
    summary: ForecastSummary
    notes: List[str]


class Driver(BaseModel):
    feature: str
    label: str
    group: str
    value: Optional[float] = None
    effect_pct: float = Field(..., description="Multiplicative effect on expected demand, in %")


class ExplainResponse(BaseModel):
    store_id: int
    date: date
    is_open: bool
    baseline_demand: float
    model_prediction: float
    predicted_demand: float
    drivers: List[Driver]
    other_features_effect_pct: float
    note: str


class BacktestPoint(BaseModel):
    date: date
    actual: float
    predicted: float
    baseline: float


class BacktestResponse(BaseModel):
    scope: str
    store_id: Optional[int] = None
    window: Dict[str, Any]
    points: List[BacktestPoint]
    metrics: Dict[str, Optional[float]]
    baseline_metrics: Dict[str, Optional[float]]
    baseline_label: str
