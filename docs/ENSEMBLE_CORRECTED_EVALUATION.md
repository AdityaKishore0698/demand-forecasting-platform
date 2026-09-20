# Corrected-methodology ensemble evaluation

**Status:** the evaluation below was run first, with no production code changed; on its basis the 3-model ensemble was then adopted as the deployed model (see §11). The numbers are unchanged from the accepted evaluation.

**Question.** When all three models are evaluated under the *current, corrected* forecasting methodology, does the archived 0.6 / 0.1 / 0.3 LightGBM + CatBoost + XGBoost ensemble still have lower error than the corrected LightGBM?

**Short answer.** Under the corrected methodology the ensemble has lower error than the corrected single LightGBM in **all three validation windows, on all four metrics**, by small margins (RMSLE −0.7 % to −1.9 %; WAPE −0.04 to −0.15 percentage points). It is the lowest-error of the four models in the primary window, in `earlier_2` and pooled — **but not in `earlier_1`, where XGBoost alone is marginally lower.** CatBoost is the highest-error model in every window.

---

## 1. Why this experiment was necessary

The ensemble numbers used so far (RMSLE ≈ 0.09414, WAPE 6.81 %) were produced **before** the horizon-42 schedule-availability correction. The single-LightGBM figures being compared with them (RMSLE 0.09472, WAPE 6.86 %) were produced **after** it. That mixes two methodologies. The ensemble had never been evaluated with the correction, and the ensemble members' XGBoost/CatBoost code does not exist in the production pipeline. This experiment evaluates all four candidates on identical footing.

## 2. What changed from the old ensemble evaluation

| | Old evaluation (reproduced locally before this one) | This evaluation |
|---|---|---|
| Horizon-42 `*_next1` schedule flags | Present in training and validation rows (not available to the API) | **Blanked on horizon 42** in training and validation rows for every model (`mask_unpublished_schedule`) |
| Feature code | The earlier experiment's feature code (same 44 features, same order) | Current `src/features/` (same 44 features, same order; masking is the only behavioural difference) |
| Windows | Primary only | **Primary, `earlier_1`, `earlier_2`** |
| XGBoost categorical encoding | `astype("category")` per frame (correct only for full frames) | **Pinned levels** learned from each window's training frame (§8) |
| CatBoost scratch files | `catboost_info/` written | `allow_writing_files=False` (no effect on the fitted model) |
| Metrics logged | RMSLE only (WAPE/MAE/RMSE computed later) | RMSLE, WAPE, MAE, RMSE, bias, ±10 %/±20 % coverage for every model |
| Machine-readable output | metrics + primary predictions | metrics + predictions for all three windows |

Nothing else changed: hyper-parameters, round counts, target transforms, seeds, thread settings and the blend are the archived recipe. **No weights or tree counts were tuned on any of these results.**

## 3. Methodology

Code: `experiments/ensemble_corrected/` (`recipes.py`, `worker.py`, `combine.py`, `bench_inference.py`); tests: `tests/test_ensemble_encoding.py`. The window, origin, feature and leakage logic is the **production code**, imported unchanged (`src.evaluation.backtest`, `src.features.builder`).

For each window (`make_backtest_windows(train_max, 42, 3)`) and each model, in its own process:

1. Training origins weekly up to `cut-off − 42 days`; training rows = (origin, store, h = 1…42) with labels. **`assert_no_temporal_leakage`**: every training target date precedes the window. The last training target date was 2015-05-05 / 2015-03-24 / 2015-02-10 for primary / earlier_1 / earlier_2.
2. Validation rows = the 1,115 stores × 42 horizons forecast from the cut-off origin (46,830 rows per window).
3. The worker asserts that the corrected methodology is switched on (`schedule_horizon == 42`, `*_next1` all missing at `h = 42` in both frames).
4. Fit, predict, save raw predictions. Each worker records a **hash of its training and validation frames**; `combine` refuses to proceed unless all three models saw identical hashes, identical row counts and identical feature names *and order* (all checks passed for all three windows).
5. Ensemble = `0.6·LightGBM + 0.1·CatBoost + 0.3·XGBoost` in raw demand space (CatBoost output converted with `expm1` first), then the closed-day rule (clip ≥ 0; `IsOpen == 0 → 0`) — applied identically to every model and the blend.
6. Metrics use the repository's own `regression_report`: RMSLE on all store-days; WAPE, MAE, RMSE on open days; ±10 % coverage on open days with actual > 0.

**Validity check:** the worker's LightGBM reproduces the production pipeline's corrected metrics **to all digits** in all three windows (`lightgbm_equals_production_pipeline_metrics: true`), so the harness is consistent with what is shipped.

## 4. Model configurations (the archived final recipes; the original experiment scripts are not part of this repository)

| | LightGBM | CatBoost | XGBoost |
|---|---|---|---|
| Objective / target | Tweedie p = 1.05, raw demand | RMSE on `log1p(demand)`; output `expm1` | `reg:tweedie` p = 1.1, raw demand |
| Rounds | 1,160 | 1,320 | 590 |
| Learning rate | 0.05 | 0.08 | 0.05 |
| Tree shape | 127 leaves, min_data_in_leaf 200 | depth 6 | depth 8, min_child_weight 50 |
| Regularisation / sampling | λ₂ 1.0; feature/bagging 0.85 | l2 3.0; Bernoulli subsample 0.8; Plain boosting | λ 1.0; subsample 0.85; colsample 0.85; `hist` |
| Seed | 42 | 42 | 42 |
| Categorical handling | pandas categories; levels stored in the model | strings (frame-independent) | pandas categories with **pinned levels** (§8) |
| Library | lightgbm 4.6.0 (production `src.models`) | catboost 1.2.10 | xgboost 2.1.4 |

## 5. Validation windows

| Window | Forecast period | Rows | Training rows |
|---|---|---|---|
| primary | 2015-05-09 → 2015-06-19 | 46,830 | 5,093,070 |
| earlier_1 | 2015-03-28 → 2015-05-08 | 46,830 | 4,812,090 |
| earlier_2 | 2015-02-14 → 2015-03-27 | 46,830 | 4,531,110 |

**Tuning caveat (precise).** The archived tree counts (1,160 / 1,320 / 590 — early-stopping best iterations plus ~10 %) and the 0.6 / 0.1 / 0.3 weights were chosen on the **primary window under the pre-correction methodology**. The primary-window scores below are therefore *validation* scores, **not** an untouched test result, and are the most favourable window for the archived recipe. The two earlier windows were not used to choose them; but they are also windows from the same period and dataset, so they are a robustness check, not an independent test. The archived validation-stage XGBoost script and the blend-weight grid-search script are missing; **those tuning procedures were not reproduced and are not claimed to be** — only the documented final recipes and the documented blend were evaluated.

## 6. Results

### 6.1 Primary window (2015-05-09 → 2015-06-19)

| Model | RMSLE | WAPE | MAE | RMSE | within ±10 % |
|---|---|---|---|---|---|
| LightGBM | 0.09472 | 6.865 % | 499.77 | 690.44 | 75.3 % |
| CatBoost | 0.10312 | 7.715 % | 561.65 | 783.79 | 70.3 % |
| XGBoost | 0.09550 | 6.955 % | 506.34 | 707.31 | 75.0 % |
| **Ensemble** | **0.09410** | **6.825 %** | **496.88** | **688.09** | **75.6 %** |

### 6.2 Per window

| | Model | RMSLE | WAPE | MAE | RMSE |
|---|---|---|---|---|---|
| **Window 1 (primary)** 2015-05-09 → 06-19 | LightGBM | 0.09472 | 6.865 % | 499.77 | 690.44 |
| | CatBoost | 0.10312 | 7.715 % | 561.65 | 783.79 |
| | XGBoost | 0.09550 | 6.955 % | 506.34 | 707.31 |
| | Ensemble | **0.09410** | **6.825 %** | **496.88** | **688.09** |
| **Window 2 (earlier_1)** 2015-03-28 → 05-08 | LightGBM | 0.10304 | 8.758 % | 670.20 | 974.69 |
| | CatBoost | 0.11167 | 9.432 % | 721.78 | 1045.17 |
| | XGBoost | **0.10110** | **8.589 %** | **657.26** | **952.29** |
| | Ensemble | 0.10111 | 8.604 % | 658.39 | 954.40 |
| **Window 3 (earlier_2)** 2015-02-14 → 03-27 | LightGBM | 0.11817 | 7.583 % | 509.44 | 780.52 |
| | CatBoost | 0.12463 | 8.509 % | 571.64 | 848.89 |
| | XGBoost | 0.11823 | 7.618 % | 511.78 | 784.17 |
| | Ensemble | **0.11729** | **7.512 %** | **504.67** | **773.82** |
| **Pooled** (3 windows, 140,490 rows) | LightGBM | 0.10576 | 7.749 % | 558.20 | 822.19 |
| | CatBoost | 0.11349 | 8.563 % | 616.87 | 897.75 |
| | XGBoost | 0.10539 | 7.732 % | 556.99 | 819.61 |
| | Ensemble | **0.10462** | **7.660 %** | **551.78** | **811.57** |

Bold marks the lowest error among the four models. **The ensemble is the lowest-error model in the primary window, `earlier_2` and pooled; XGBoost alone is lowest in `earlier_1`** (by 0.00001 RMSLE, 0.015 pp WAPE, 1.1 MAE, 2.1 RMSE). Under the corrected methodology XGBoost alone is also slightly lower-error than LightGBM in the pooled figures, so a single LightGBM is not the strongest single member.

### 6.3 Ensemble compared with the **corrected** LightGBM (ensemble − LightGBM; negative = ensemble better)

| | RMSLE abs | RMSLE rel | WAPE (pp) | MAE | RMSE |
|---|---|---|---|---|---|
| primary | −0.00062 | −0.66 % | −0.040 | −2.89 | −2.35 |
| earlier_1 | −0.00193 | −1.87 % | −0.154 | −11.81 | −20.28 |
| earlier_2 | −0.00088 | −0.74 % | −0.071 | −4.76 | −6.70 |
| pooled | −0.00114 | −1.07 % | −0.089 | −6.43 | −10.62 |
| mean of the 3 windows | −0.00114 | −1.08 % | −0.088 | −6.49 | −9.78 |

The ensemble had lower error than LightGBM in **3 of 3 windows on RMSLE, WAPE, MAE and RMSE**.

**Store-level bootstrap** (2,000 paired resamples of the 1,115 stores; ensemble − LightGBM; 95 % interval): RMSLE primary −0.00061 [−0.00101, −0.00021]; `earlier_1` −0.00193 [−0.00220, −0.00168]; `earlier_2` −0.00087 [−0.00114, −0.00061]. WAPE and MAE intervals also exclude zero in all windows; the primary-window RMSE interval does include zero ([−5.49, +0.54]; ensemble better in 94 % of resamples). **This interval only reflects which stores are in a window with the models held fixed. It does not capture training randomness, window-to-window variation, or the tuning-on-the-primary-window caveat, so it is not a general significance test.**

### 6.4 Shift relative to the old (pre-correction) numbers — primary window

| Model | RMSLE old → new | WAPE old → new | Δ RMSLE |
|---|---|---|---|
| LightGBM | 0.09476 → 0.09472 | 6.842 % → 6.865 % | −0.00004 |
| CatBoost | 0.10208 → 0.10312 | 7.591 % → 7.715 % | **+0.00105** |
| XGBoost | 0.09585 → 0.09550 | 6.983 % → 6.955 % | −0.00036 |
| Ensemble | 0.09414 → 0.09410 | 6.809 % → 6.825 % | −0.00003 |

Old figures are from the local reproduction of the pre-correction ensemble; they are shown only to quantify the shift, **not** as comparable results. The like-for-like gain of the ensemble over LightGBM on the primary window is about the same before and after correction (−0.00062 RMSLE both times). Old numbers exist only for the primary window, so no shift can be given for the earlier windows. CatBoost's deterioration under the correction was not investigated.

### 6.5 Diagnostic (not a candidate, not tuned)
The archived weights with CatBoost removed and the other two renormalised (0.6/0.9, 0.3/0.9) — declared before looking at results — gave RMSLE 0.09414 / 0.10146 / 0.11758 (pooled 0.10485; WAPE 7.674 %) against 0.09410 / 0.10111 / 0.11729 (pooled 0.10462; 7.660 %) for the three-model blend: CatBoost's marginal contribution is small but positive on RMSLE in every window.

## 7. Engineering measurements (this laptop; not Render)

All timings are wall-clock on the same macOS arm64 machine; models were trained on the window's 4.5–5.1 M rows (a final fit on all 5.37 M rows would be slightly larger/slower — **not run**).

**Training time (seconds)**

| Window | LightGBM | XGBoost | CatBoost | Ensemble total |
|---|---|---|---|---|
| primary | 126 | 209 | 1,715 | 2,049 |
| earlier_1 | 145 | 194 | 1,534 | 1,873 |
| earlier_2 | 123 | 179 | 1,520 | 1,822 |

The three-model set took 13–16× a LightGBM fit (16.3× / 12.9× / 14.8×); CatBoost is 82–84 % of it. Total for all nine fits: 5,744 s (95.7 min).

**Model files (primary-window models)**

| | LightGBM | CatBoost | XGBoost (JSON / UBJSON) | Total |
|---|---|---|---|---|
| Raw | 21.74 MB | 15.89 MB | 67.75 / 63.12 MB | 105.38 MB (JSON) / 100.75 MB (UBJSON) |
| gzip -9 | 8.36 MB | 5.83 MB | 7.24 / 5.88 MB | 21.43 MB (JSON) |

**Inference, fresh process each** (`bench_inference.py`; primary-window models)

| | LightGBM only | 3-model ensemble |
|---|---|---|
| Model load | 0.09–0.17 s | LightGBM 0.17 s + CatBoost 0.015 s + XGBoost **1.56 s (JSON) / 0.064 s (UBJSON)** |
| Micro-benchmark process peak RSS after loading models † | 267 MB | 1,065 MB † |
| Micro-benchmark process peak RSS after inference † | 328 MB | 1,195 MB † |
| 42-row single-store forecast, median (p95) | 2.66 ms (2.88) | **15.57 ms (16.72)** = LightGBM 3.56 + CatBoost 2.34 + XGBoost 8.67 + blend/encode |
| 46,830-row batch | 1.30 s | 2.06 s |

† **These two rows overstate what production uses.** The micro-benchmark loaded *both* XGBoost formats (JSON, then UBJSON) in one process, so its peak includes the transient JSON parse. The deployed API loads the UBJSON copy only. Measured after the migration on like-for-like real-data bundles, whole API process (including the 970k-row history): **548 MB with the single LightGBM, 710 MB with the ensemble (+162 MB)**; an uncached 42-day forecast over HTTP: **14 ms vs 23 ms**.

Latency varies by roughly 30 % between sessions on this machine (an earlier session measured 11.5 ms for the ensemble on the archived models), so read these as orders of magnitude. **Training memory:** process peak was 6.0–8.2 GB in every run, dominated by feature/training-frame preparation; fitting raised the high-water mark by at most ~1 GB. `ru_maxrss` is a process-wide high-water mark, so **per-model training memory is not cleanly measurable this way** and is not reported. **Render's memory limit was not verified and no cloud cost is estimated.**

## 8. XGBoost categorical-encoding issue and fix

**Symptom.** Scoring one store's 42 rows gave XGBoost predictions that differ from the same rows scored inside a full frame — up to **21.6 %** with the archived final model (10.5 % with the corrected primary-window model), mean ~7 %.

**Root cause (measured).**
1. XGBoost 2.1.4 stores **no category levels** in the model (JSON has only `feature_types: "c"`). It consumes the integer **codes** of the pandas Categorical, and its splits are partitions over those codes.
2. `astype("category")` derives the levels **from the values in that frame**. In a single-store frame, `HubID` has one level, so store 42 receives code **0** instead of its training code **41**; `AssortmentTier` 3 → code 0 (training 2); `LoyaltyProgram` 1 → code 0 (training 1). The model then behaves as if the row belonged to the store/tier/programme that owns code 0.
3. Column attribution: with all other columns pinned, un-pinning only `HubID` reproduces the whole error (max 21.6 %, mean 6.9 %); un-pinning `LoyaltyProgram` or `AssortmentTier` alone adds up to 0.2–0.7 %; `Weekday` and `HubFormat` are harmless (their codes coincide).
4. **Why the old ensemble evaluation was unaffected:** the validation and training frames both contained every level in the same sorted order, so per-frame codes equalled training codes (measured: max prediction difference 0.0 between pinned and per-frame encodings on the full frame). The hazard exists only for frames that lack some levels — i.e. API requests for one store.
5. LightGBM avoids the problem because it stores the training category mapping inside the model and re-maps at predict time; CatBoost avoids it because categoricals are passed as strings.

**Fix** (`experiments/ensemble_corrected/recipes.py::encode_xgboost`): encode with `pd.Categorical(values, categories=levels)` where `levels` are learned from the *training* frame (`learn_categories`, sorted unique values) and reused for validation and any request. An unseen level becomes code −1, which XGBoost treats as missing (verified: finite predictions, no crash). Training uses the same pinned levels, so train / validation / inference are code-compatible; the levels must be persisted with the model (the production model card already stores `categories`).

**Regression tests** (`tests/test_ensemble_encoding.py`, 7 tests, skipped if xgboost/catboost are not installed): pinned single-store scoring equals full-frame scoring for every store (XGBoost, CatBoost); naive per-frame encoding **reproduces** the failure (asserts the code collapses to 0 and predictions differ); full frames are unaffected by pinning; an unseen level yields finite predictions; blend weights sum to 1 and the blend is the exact weighted sum. On the real corrected models, single-store vs full-frame maximum relative difference over 60 random stores: **0.0 for LightGBM, CatBoost and XGBoost**; naive XGBoost encoding for store 42: 10.5 %.

## 9. Limitations

- **Tuning:** tree counts and weights were tuned on the primary window under the pre-correction methodology (§5); the primary window is not an untouched test set and favours the archived recipe. The tuning procedures were not reproduced.
- **Margins are small.** The ensemble's gain over LightGBM is 0.04–0.15 pp WAPE and 0.7–1.9 % RMSLE; per-window differences are consistent in sign but are one evaluation of one training run per model.
- **Determinism:** each model was trained once per window under the corrected methodology; CatBoost/XGBoost run-to-run determinism was **not** re-tested here (the earlier pre-correction reproduction agreed to 5 decimals).
- **Single dataset and period;** three overlapping-in-time windows from 2015; no prediction intervals; store-bootstrap intervals capture store sampling only.
- **Models trained on window training sets, not a final fit** on all 5.37 M rows; sizes/latency are for those models.
- **Engineering figures** are from one laptop; latency varies ~30 % between sessions; training memory not attributable per model; Render limits unverified; no cost estimate.
- CatBoost's deterioration under the correction (§6.4) was not investigated.
- The `experiments/` code is research code, not the production implementation; XGBoost/CatBoost are not part of `requirements.txt` and the tests importorskip them.

## 10. Decision

**What the numbers support:** under the corrected methodology the 0.6/0.1/0.3 ensemble has lower error than the corrected LightGBM in every window and on every metric, is the lowest-error of the four models in two of three windows and pooled, and is within 0.00001 RMSLE of XGBoost-alone in the third, where XGBoost is marginally ahead. The gain is modest (≈ −1 % RMSLE, ≈ −0.09 pp WAPE pooled) and is smallest on the window the archived recipe was tuned on.

**What they do not support:** describing the ensemble as uniformly best, presenting the primary-window score as an untouched test result, or claiming a large improvement or any statistical significance beyond the store-sampling interval above.

**Decision taken:** the ensemble was adopted as the deployed model, accepting the measured cost (§7): 13–16× the training time, +162 MB of process memory for the whole API (548 → 710 MB, real-data bundle), ≈ 6× the 42-row model time in a micro-benchmark (23 vs 14 ms end to end over HTTP), ≈ +448 MB of Linux dependencies, and an explanation method that had to be redesigned. Compatibility with a particular hosting plan is not established by this document. The §6.5 diagnostic suggests CatBoost — the most expensive dependency — contributes little; that was not acted on, because the deployed model is the documented ensemble.

## 11. Implementation status

- **Production port:** `src/models/ensemble.py` (recipes, pinned encodings, fit / predict / contributions, hash-verified save and load), model-type dispatch in the pipeline, back-tests, interpretability and service; the recipes are asserted equal to the evaluated ones by a test.
- **Serving:** all three models are loaded once at start-up; each forecast is validated (row count, horizon order, store, dates, no NaN/inf, non-negative) before it is returned; a failed load gives a degraded `/health`; the previous single-LightGBM path remains supported (`model.type: lightgbm_single`) as the rollback.
- **Explanations:** exact TreeSHAP per model; the ensemble-level view is a blend-weighted average, labelled approximate, with its reconstruction error reported per request (measured on 3,000 open store-days: mean 0.023 %, 99th percentile 0.25 %, worst 1.9 %). The three models agree on the single top driver for only 49 % of rows, so the per-model views carry information.
- **Tests:** 96 backend tests, including single-store equals full-frame for each member and the blend, pinned categories, unseen categories, NaN/inf rejection, blend arithmetic, hash-tamper detection and "no model loading or training inside a request".
- **Demo bundle** (synthetic data, same recipe): 22 MB of files; in the Docker image ≈ 254 MiB after start-up, ≈ 19 ms per uncached 42-day forecast, ≈ 77 ms per explanation. On synthetic data the ensemble is not the best member (CatBoost is), so its numbers are illustrative only.
