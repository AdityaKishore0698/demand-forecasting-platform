# Architecture and design decisions

This document explains *why* the project is built the way it is. Facts and numbers come from the code, `configs/default.yaml` and `artifacts/reports/`.

## 1. Problem framing

**Task.** For each of 1,115 stores, predict daily order volume for every day in a 42-day window that starts the day after the last recorded day (19 Jun 2015).

**What is known about the future.** Only the *schedule*: whether the store is open, whether a promotion runs, regional holidays and school closures (`orders_test.csv` carries these flags but no demand). Everything else about the future is unknown, so the model may not use it. One column (`AppSessions`) exists only in the training file; it is unknown for future days, so it is excluded.

**Target properties** (see [EDA.md](EDA.md)): ~17% of store-days are closed and always zero; open-day demand is right-skewed; strong weekday rhythm; promotions lift demand ~39%.

## 2. Direct multi-horizon forecasting with origin-anchored features

Two standard options for multi-step forecasting:

| Approach | Idea | Downside here |
|---|---|---|
| Recursive | Predict day +1, feed it back as a lag, predict +2… | Errors compound; needs simulated inputs for the whole path |
| **Direct (chosen)** | One model that predicts `y[origin + h]` for any `h` in 1..42 | Slightly larger training set; the step `h` must be a feature |

For a *forecast origin* (the last known day) every history feature is computed **as of the origin** and frozen: lags (1, 7, 14, 21, 28, 35, 42 days back from the origin), rolling mean/std over 7/14/28/42 days, an expanding mean, and a store×weekday historical average. The target day contributes only what is genuinely known about it: its weekday, month, day-of-month, open/promotion/holiday/school flags (plus the previous and next day's flags), store attributes and the step `h`.

Training examples are (origin, store, h) triples. Origins are spaced 7 days apart (`forecast.origin_stride_days`) across the history — 5,374,050 rows in the final model. A stride of 3 days gave no better validation score in an earlier experiment (RMSLE 0.12299 vs 0.12281) at ~2× the cost.

### Calendar-aware panels

Each store's history is reindexed to the full daily calendar so `shift(k)` and rolling windows mean *calendar days*, not "k rows". 181 stores have a gap in their history (for almost all of them a single 184-day stretch); features that would need missing days are `NaN`, which LightGBM handles natively. Rolling windows use `min_periods = max(3, w // 2)`.

## 3. Feature set (44 features)

| Group | Features | Notes |
|---|---|---|
| Forecast horizon | `h` | days ahead of the origin |
| Operating schedule | `IsOpen`, `IsOpen_prev1`, `IsOpen_next1` | known in advance; strongest signal |
| Promotions & holidays | `PromoActive`, `RegionalHoliday`, `SchoolClosureFlag` + prev/next-day versions | known in advance |
| Calendar | `Weekday`, `Month`, `DayOfMonth`, `WeekOfYear`, `IsWeekend` | derived from the target date |
| Recent demand | `OV_lag{1,7,14,21,28,35,42}`, `OV_rollmean{7,14,28,42}`, `OV_rollstd{7,14,28,42}` | anchored at the origin |
| Store baseline | `OV_expanding_mean`, `OV_hubweekday_mean` | long-run level and per-weekday level |
| Store attributes | `HubID`, `HubFormat`, `AssortmentTier`, `CompetitorDistance`, `MonthsSinceCompetitorOpen`, `HasKnownCompetitorOpenDate`, `LoyaltyProgram`, `IsLoyaltyActive`, `IsLoyaltyPromoMonth` | static metadata |

`OV_*` = order volume. Human-readable names and descriptions for all features live in `src/features/catalog.py` and are shown in the dashboard.

## 4. Model

**A 3-model boosting ensemble: `forecast = 0.6·LightGBM + 0.1·CatBoost + 0.3·XGBoost`** (raw demand space; closed days then forced to 0). Every parameter lives in `configs/default.yaml` (`model.ensemble`), asserted equal to the evaluated recipes by a test; nothing was re-tuned for deployment.

| Member | Objective / target | Rounds | Key parameters |
|---|---|---|---|
| LightGBM | Tweedie p = 1.05, raw demand | 1,160 | lr 0.05, 127 leaves, `min_data_in_leaf` 200, L2 1.0, feature/bagging fraction 0.85 |
| CatBoost | RMSE on `log1p(demand)`, prediction `expm1` | 1,320 | lr 0.08, depth 6, l2 3.0, Plain boosting, Bernoulli subsample 0.8 |
| XGBoost | `reg:tweedie` p = 1.1, raw demand | 590 | lr 0.05, depth 8, `min_child_weight` 50, subsample/colsample 0.85, `hist` |

All three share the same 44 features in the same order. The single-LightGBM model remains supported (`model.type: lightgbm_single`) as the rollback path. Evidence for the choice: [ENSEMBLE_CORRECTED_EVALUATION.md](ENSEMBLE_CORRECTED_EVALUATION.md).

*Why trees?* Mixed numeric/categorical inputs, strong interactions (weekday × promotion × store), missing values, no scaling, fast on 5M rows, and exact TreeSHAP explanations. *Why not a linear model?* It cannot capture the store × weekday × promotion interactions or zero-inflation without extensive manual feature crossing. *Why not deep learning?* No evidence it would help on tabular data of this size, and it would cost interpretability and training time; the project deliberately avoids it.

*Why Tweedie?* The target is non-negative, right-skewed and contains many exact zeros. A Tweedie loss with a log link models this on the raw scale (predictions can never be negative), and it beat a log1p-transformed RMSE model in validation (RMSLE 0.09474 vs 0.09695). Power 1.05 was chosen from a small search over 1.01–1.5 on the primary window.

*Closed-day rule.* `IsOpen` is known in advance and demand is always 0 when closed (verified in the data), so the forecast is forced to 0 on closed days. It is applied consistently in evaluation, the API and the dashboard.

*Round counts.* Fixed in the config: each is the best early-stopping iteration on the primary window (LightGBM 1,059, CatBoost 1,199, XGBoost 532) plus ~10% margin because the final model sees more data. The training functions never use a validation set, so they cannot silently peek — but the primary window therefore informed these choices (and the blend weights), which is why it is a validation result, not a test result.

## 5. Validation design

`src/evaluation/backtest.py` builds **rolling-origin chronological windows** of 42 days ending at the last known day and stepping back (primary, earlier_1, earlier_2). For each window:

1. Cut-off = window start − 1 day.
2. Training origins are chosen so every training *target date* is ≤ cut-off. `assert_no_temporal_leakage` verifies this and raises otherwise.
3. Train, forecast all 42 days from the cut-off, score against actuals, and score five baselines (global mean, store mean, last value, same weekday last week, 7-day rolling mean) with the same closed-day rule.

Metrics (`src/evaluation/metrics.py`): **RMSLE** on all rows (model-selection metric, relative error, stable near zero); **MAE, RMSE, WAPE, RMSPE, MAPE** on open days only. WAPE is the headline business metric (total error ÷ total demand); RMSPE is reported for comparison but is dominated by low-volume days and undefined at zero. Each figure is labelled as *training in-sample* (a 200k-row sample, WAPE 5.4%), *primary validation* (6.9%; RMSLE 0.0947) or *earlier windows* (7.6%, 8.8%).

**Honest caveat.** The primary window drove hyper-parameter and tree-count choices, so it is optimistic relative to a fresh test. The earlier windows are less tuned. There is no measurable score for the true future window.

## 6. Interpretability

- **Permutation importance** — increase in hold-out RMSLE when a feature is shuffled, measured on the **ensemble's blended output** (backtest ensemble, 15k rows × 3 repeats): exact for the deployed model.
- **SHAP** — exact TreeSHAP per model (LightGBM `pred_contrib`, XGBoost `pred_contribs`, CatBoost `ShapValues`); contributions are additive in each model's own space (log link for LightGBM/XGBoost, log1p target for CatBoost), shown as multiplicative effects (`exp(c) − 1`). The ensemble-level list is the blend-weighted average of the three and is labelled **approximate**; each model's own list is exact.
- **Gain** — split gain; units differ across libraries, so the headline list is the LightGBM component only (labelled as such) and per-model lists are provided separately.
- **Per-forecast drivers** (`/explain`) — exact per-model drivers plus a blend-weighted ensemble view marked `ensemble_approximate`, with the reconstruction error of that approximation returned per request. The blended forecast itself is exact; only its attribution is approximate.

Top features by permutation importance on the earlier single-LightGBM model: store open that day (62%), store's typical demand on that weekday (29%), regional holiday (3.9%), promotion running (2.3%) (the ensemble's own ranking is on the Feature Insights page). Recent-demand lags rank lower, plausibly because the store×weekday baseline already carries much of the same information (not tested in isolation). Importance ≠ causation, and correlated features share credit.

## 7. Serving design

```
artifacts/<bundle>/        # <bundle> = demo (committed, synthetic) or the repo-level artifacts/ (yours, git-ignored)
  model/     lightgbm.txt.gz, catboost.cbm, xgboost.ubj, model_card.json     (single-LightGBM bundles: lgbm_tweedie.txt.gz)
  serving/   history.csv.gz, origin_features.csv.gz, hubweekday_features.csv.gz, schedule.csv.gz, hub_metadata.csv
  reports/   metrics.json, feature_importance.json, backtest_predictions.csv.gz, dataset_summary.json
```

The API loads `ARTIFACT_DIR` if set, else `artifacts/` when it holds a trained model, else the committed synthetic demo bundle `artifacts/demo/`. The pipeline records a provenance tag (`data.label` in the config) in `dataset_summary.json` and the model card; for the demo bundle it is `synthetic-demo`, which `/health` returns and the dashboard shows as a banner. `serving/history.csv.gz` is a copy of the training records, and `reports/backtest_predictions.csv.gz` holds actuals, so a bundle is only as publishable as the data it was trained on.

- The API never trains and never loads a model inside a request: the three model files are loaded once at start-up and verified against the SHA-256 hashes and blend weights in `model_card.json` (a mismatch or missing file gives a degraded `/health`, not a crash).
- The API never trains. `ForecastService.load()` reads the artifacts once at start-up (FastAPI lifespan).
- The forecast origin is fixed at the last known day, so origin features are **pre-computed and stored**; per request the service only rebuilds the small per-store feature rows with the *same* `build_supervised_dataset` used in training — this is what prevents train/serve skew (`tests/test_service.py` asserts equality with batch features).
- Categorical levels are pinned from training (`prepare_matrix`), so scoring one store alone equals scoring it inside the full frame (tested).
- Results are LRU-cached per `(store, horizon)`. Horizon is validated (1–42); unknown stores return 404; missing artifacts return 503 with instructions.
- Requests never supply future inputs — the schedule comes from the stored snapshot — so the API cannot be used to invent scenarios by accident.

## 8. Frontend

React 18 + TypeScript (strict) on Vite. Data layer: a thin typed `fetch` client + TanStack Query hooks (`src/api/`), one hook per endpoint, with caching, `keepPreviousData` on selector changes, and explicit loading/error/empty states. Design tokens are CSS variables with separate light and dark palettes; charts read the active palette at render time (`useChartColors`) so Recharts (SVG) follows the theme. The theme is applied before first paint by an inline script to avoid a flash. Animations use Framer Motion and respect `prefers-reduced-motion`.

## 9. Alternatives considered

| Idea | Outcome |
|---|---|
| LightGBM + CatBoost + XGBoost blend (0.6/0.1/0.3) | **Adopted.** Corrected evaluation: lower error than LightGBM alone in all three windows (RMSLE −0.7 % to −1.9 %, WAPE −0.04 to −0.15 pp), at higher training, memory and dependency cost — see ENSEMBLE_CORRECTED_EVALUATION.md |
| log1p target + RMSE | Worse than Tweedie (0.09695 vs 0.09474) |
| Extra store × promo / store × school baselines | Worse (0.09864–0.10006 vs 0.09695) — overfit |
| Momentum ratio feature (rollmean7 / rollmean28) | Worse (0.115 vs 0.097), three variants tried |
| Year / linear-trend feature | Worse (0.0960–0.1005 vs 0.0947) |
| Cross-store network-wide demand features | Worse (0.09686 vs 0.09474) on the primary window |
| 7-day schedule-density window | Worse (0.09705 vs 0.09474) |

Details in [EXPERIMENTS.md](EXPERIMENTS.md) and [ENSEMBLE_CORRECTED_EVALUATION.md](ENSEMBLE_CORRECTED_EVALUATION.md). The rejected-idea rows are single-window, pre-correction validation numbers.

## 10. Schedule-horizon boundary (resolved) and production hardening not yet done

**Resolved.** Each row carries the previous and next day's schedule flags (`*_prev1`, `*_next1`). The schedule is published only `horizon` (42) days ahead, so for the *last* forecast day the "next day" flag does not exist at inference. An earlier version let training and back-testing see that flag while the API could not, which flattered the last-day validation. `mask_unpublished_schedule` (in `src/features/builder.py`) now blanks any `*_nextK` feature whose day would fall beyond the published schedule (`h + K > 42`), and it is applied by the one shared dataset builder — so training, back-testing and serving are identical by construction (tests: `test_next_day_schedule_is_blank_when_it_would_not_be_published`, `test_last_forecast_day_uses_the_same_masked_schedule_as_training`). The corrected results and the pre-correction numbers are in [ML_AUDIT.md](ML_AUDIT.md) §3.

**Not done (out of scope for this portfolio project):**
- **Monitoring**: log request volume/latency; track input drift (PSI on key features) and, once actuals arrive, rolling WAPE by horizon.
- **Retraining**: scheduled `make train` on new data with a promotion gate (new model must beat the current one on the latest back-test window).
- **Intervals**: quantile regression (LightGBM `objective=quantile`) or conformal prediction on back-test residuals.
- **Scale to thousands of stores**: the model is global, so cost grows with rows, not stores — pre-compute origin features in batch, shard by store, and cache forecasts; the API only needs the per-store feature rows.
- **Auth / rate limiting** — none, and the API is read-only.
