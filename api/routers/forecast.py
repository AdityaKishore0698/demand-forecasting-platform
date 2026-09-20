from datetime import date as date_type

from fastapi import APIRouter, Depends, Query

from api.deps import get_service
from api.schemas import ExplainResponse, ForecastRequest, ForecastResponse
from src.forecasting.service import ForecastService

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse,
             summary="Forecast daily demand for a store over the next `horizon` days")
def forecast(body: ForecastRequest, service: ForecastService = Depends(get_service)) -> ForecastResponse:
    return ForecastResponse(**service.forecast(body.store_id, body.horizon))


@router.get("/explain", response_model=ExplainResponse,
            summary="Why the model forecast what it did for one store-day (exact TreeSHAP per model; ensemble view labelled approximate)")
def explain(
    store_id: int = Query(..., ge=1),
    date: date_type = Query(..., description="A date inside the forecast window"),
    top_k: int = Query(6, ge=1, le=15),
    service: ForecastService = Depends(get_service),
) -> ExplainResponse:
    return ExplainResponse(**service.explain(store_id, date, top_k))
