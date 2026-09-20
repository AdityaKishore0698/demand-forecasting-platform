// Mirrors api/schemas.py (the API contract).

export interface Health {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  data_through: string | null;
  forecast_horizon_days: number | null;
  data_label?: string | null;
  model_type?: string | null; model_components?: string[] | null;
  detail: string | null;
}

export interface StoreInfo {
  store_id: number;
  hub_format: number | null;
  assortment_tier: number | null;
  competitor_distance: number | null;
  loyalty_program: boolean;
  avg_open_day_demand: number;
  history_days: number;
  has_history_gap: boolean;
  demand_rank: number;
  demand_percentile: number;
  backtest_wape: number | null;
}
export interface StoresResponse { count: number; stores: StoreInfo[] }

export interface HistoricalPoint { date: string; demand: number | null; is_open: boolean | null; promo_active: boolean | null }
export interface HistorySummary {
  recorded_days: number; missing_days: number; open_days: number; avg_open_day_demand: number | null;
  first_history_date: string; last_history_date: string;
}
export interface HistoricalResponse { store_id: number; start: string; end: string; points: HistoricalPoint[]; summary: HistorySummary }

export interface ForecastPoint {
  date: string; weekday: string; horizon_step: number; predicted_demand: number;
  is_open: boolean; promo_active: boolean; holiday: boolean; school_closure: boolean;
}
export interface PeakPoint { date: string; value: number }
export interface ForecastSummary {
  total_predicted: number; mean_open_day: number | null; peak: PeakPoint | null; lowest_open_day: PeakPoint | null;
  open_days: number; closed_days: number; promo_days: number;
}
export interface ForecastResponse {
  store_id: number; horizon: number; origin_date: string; model_version: string;
  forecast: ForecastPoint[]; summary: ForecastSummary; notes: string[];
}

export interface Driver { feature: string; label: string; group: string; value: number | null; effect_pct: number }
export interface ComponentExplanation {
  model: string; label: string; weight: number; exactness: string; prediction: number; baseline_demand: number;
  reconstruction_error_pct: number; drivers: Driver[]; other_features_effect_pct: number;
}
export interface ExplainResponse {
  store_id: number; date: string; is_open: boolean; baseline_demand: number; model_prediction: number;
  predicted_demand: number; drivers: Driver[]; other_features_effect_pct: number; note: string;
  /** "ensemble_approximate" (blend-weighted view) or "lightgbm_exact" (single-model bundle). */
  explanation_scope?: string; explanation_exactness?: string;
  approximation?: { method: string; weights: Record<string, number>; reconstruction_error_pct: number; note: string } | null;
  components?: ComponentExplanation[] | null;
}

export interface BacktestPoint { date: string; actual: number; predicted: number; baseline: number }
export type MetricMap = Record<string, number | null>;
export interface BacktestResponse {
  scope: string; store_id: number | null; window: WindowInfo; points: BacktestPoint[];
  metrics: MetricMap; baseline_metrics: MetricMap; baseline_label: string;
}

export interface WindowInfo { name: string; cutoff: string; start: string; end: string; horizon: number }
export interface GroupError { group: string; n_open_rows: number; rmsle: number | null; wape: number | null; mae: number | null }
export interface MetricsResponse {
  generated_at: string;
  model_label?: string; model_type?: string;
  members?: Record<string, MetricMap>;
  definitions: Record<string, string>;
  primary_window: WindowInfo;
  model: MetricMap;
  training_in_sample: MetricMap;
  baselines: Record<string, { label: string } & MetricMap>;
  best_baseline: string;
  improvement_vs_best_baseline_pct: Record<string, number | null>;
  breakdowns: { by_horizon: GroupError[]; by_weekday: GroupError[]; by_hub_tier: GroupError[]; by_promo: GroupError[] };
  error_distribution: {
    bin_edges: number[]; counts: number[]; n: number; clipped_low: number; clipped_high: number;
    median_pct_error: number; p10: number; p90: number; share_within_10pct: number; share_within_20pct: number;
  };
  windows: Array<WindowInfo & {
    n_train_rows: number; model: MetricMap; best_baseline: string; best_baseline_metrics: MetricMap;
  }>;
  validation_notes: string[];
}

export interface ImportanceItem {
  feature: string; label: string; group: string; group_label: string; description: string;
  value: number; share: number; std?: number;
}
export interface ImportanceResponse {
  gain: ImportanceItem[]; permutation: ImportanceItem[]; shap: ImportanceItem[];
  shap_by_model?: Record<string, ImportanceItem[]>; gain_by_model?: Record<string, ImportanceItem[]>;
  meta: Record<string, string | number>;
}
export type ImportanceView = "permutation" | "shap" | "gain";

export interface ModelInfo {
  model_version: string; algorithm: string; created_at: string; seed: number; n_estimators: number;
  model_type?: string; model_type_id?: string; data_label?: string | null;
  weights?: Record<string, number>;
  components?: Array<{ name: string; family: string; weight: number; n_trees?: number; library_version?: string; size_bytes?: number }>;
  params: Record<string, number | string>; n_features: number;
  features: Array<{ feature: string; label: string; group: string; group_label: string; description: string }>;
  horizon_days: number; origin_date: string; forecast_window: { start: string; end: string };
  training: { n_rows: number; n_origins: number; origin_stride_days: number; history_start: string; history_end: string; n_hubs: number };
  validation_summary: MetricMap & { window: WindowInfo };
  serving_note: string;
}

export interface DatasetInfo {
  train_rows: number; future_rows: number; n_hubs: number; train_start: string; train_end: string;
  forecast_start: string; forecast_end: string; calendar_days: number; hubs_with_history_gaps: number;
  zero_demand_share: number; closed_day_share: number; closed_days_always_zero: boolean;
  open_days_with_zero_demand: number; mean_open_day_demand: number; hub_demand_cv: number;
  promo_uplift_ratio: number | null; weekday_open_rate: Record<string, number>; weekday_mean_demand: Record<string, number>;
  future_promo_rate: number; future_holiday_rate: number; future_school_closure_rate: number;
  train_only_columns_excluded: string[];
}
