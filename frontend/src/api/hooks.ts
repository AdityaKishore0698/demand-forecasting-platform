import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { get, post } from "./client";
import type {
  BacktestResponse, DatasetInfo, ExplainResponse, ForecastResponse, Health, HistoricalResponse,
  ImportanceResponse, MetricsResponse, ModelInfo, StoresResponse,
} from "./types";

const STATIC = { staleTime: Infinity, gcTime: Infinity, retry: 1 } as const;

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: ({ signal }) => get<Health>("/health", signal), refetchInterval: 60_000, retry: 0 });

export const useStores = () =>
  useQuery({ queryKey: ["stores"], queryFn: ({ signal }) => get<StoresResponse>("/stores", signal), ...STATIC });

export const useHistory = (storeId: number | null, days = 2000) =>
  useQuery({
    queryKey: ["history", storeId, days],
    queryFn: ({ signal }) => get<HistoricalResponse>(`/historical-data?store_id=${storeId}&days=${days}`, signal),
    enabled: storeId != null, staleTime: 10 * 60_000,
  });

export const useForecast = (storeId: number | null, horizon: number) =>
  useQuery({
    queryKey: ["forecast", storeId, horizon],
    queryFn: ({ signal }) => post<ForecastResponse>("/forecast", { store_id: storeId, horizon }, signal),
    enabled: storeId != null, staleTime: 10 * 60_000, placeholderData: keepPreviousData,
  });

export const useExplain = (storeId: number | null, date: string | null) =>
  useQuery({
    queryKey: ["explain", storeId, date],
    queryFn: ({ signal }) => get<ExplainResponse>(`/explain?store_id=${storeId}&date=${date}&top_k=7`, signal),
    enabled: storeId != null && !!date, staleTime: 10 * 60_000,
  });

export const useMetrics = () =>
  useQuery({ queryKey: ["metrics"], queryFn: ({ signal }) => get<MetricsResponse>("/metrics", signal), ...STATIC });

export const useBacktest = (storeId: number | null) =>
  useQuery({
    queryKey: ["backtest", storeId ?? "all"],
    queryFn: ({ signal }) => get<BacktestResponse>(storeId == null ? "/backtest" : `/backtest?store_id=${storeId}`, signal),
    staleTime: Infinity, placeholderData: keepPreviousData,
  });

export const useImportance = () =>
  useQuery({ queryKey: ["importance"], queryFn: ({ signal }) => get<ImportanceResponse>("/feature-importance", signal), ...STATIC });

export const useModelInfo = () =>
  useQuery({ queryKey: ["model-info"], queryFn: ({ signal }) => get<ModelInfo>("/model-info", signal), ...STATIC });

export const useDatasetInfo = () =>
  useQuery({ queryKey: ["dataset-info"], queryFn: ({ signal }) => get<DatasetInfo>("/dataset-info", signal), ...STATIC });
