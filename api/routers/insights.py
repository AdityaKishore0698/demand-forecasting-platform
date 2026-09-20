from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_service
from api.schemas import BacktestResponse
from src.forecasting.service import ForecastService
from src.utils.io import to_jsonable

router = APIRouter(tags=["model"])


@router.get("/metrics", summary="Hold-out back-test metrics, baselines, error breakdowns, robustness windows")
def metrics(service: ForecastService = Depends(get_service)):
    return service.metrics


@router.get("/backtest", response_model=BacktestResponse,
            summary="Actual vs predicted over the hold-out window (one store, or all stores summed)")
def backtest(
    store_id: Optional[int] = Query(None, ge=1),
    service: ForecastService = Depends(get_service),
) -> BacktestResponse:
    return BacktestResponse(**to_jsonable_metrics(service.backtest(store_id)))


@router.get("/feature-importance", summary="Gain, permutation and SHAP importance with plain-English labels")
def feature_importance(service: ForecastService = Depends(get_service)):
    return service.importance


def to_jsonable_metrics(payload: dict) -> dict:
    payload = dict(payload)
    payload["metrics"] = to_jsonable(payload["metrics"])
    payload["baseline_metrics"] = to_jsonable(payload["baseline_metrics"])
    return payload
