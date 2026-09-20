from fastapi import APIRouter, Depends, Query

from api.deps import get_service
from api.schemas import HistoricalResponse, StoresResponse
from src.forecasting.service import ForecastService

router = APIRouter(tags=["stores"])


@router.get("/stores", response_model=StoresResponse, summary="All stores/hubs with summary statistics")
def list_stores(service: ForecastService = Depends(get_service)) -> StoresResponse:
    stores = service.list_stores()
    return StoresResponse(count=len(stores), stores=stores)


@router.get("/historical-data", response_model=HistoricalResponse,
            summary="Daily demand history for one store (missing dates are returned as nulls)")
def historical_data(
    store_id: int = Query(..., ge=1),
    days: int = Query(180, ge=1, le=2000, description="Trailing days to return"),
    service: ForecastService = Depends(get_service),
) -> HistoricalResponse:
    return HistoricalResponse(**service.historical(store_id, days))
