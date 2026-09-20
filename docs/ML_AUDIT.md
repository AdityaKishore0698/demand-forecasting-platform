# ML audit

A pre-publication review of the modelling code: is there temporal leakage, do training and serving agree, and can the reported numbers be reproduced? Everything below was run against the actual code and the real dataset (kept local; see `data/README.md`). Scripts were one-off checks and are described, not shipped.

**Outcome:** the audit found one real problem (§3) — training and back-testing could see a next-day schedule flag that the API cannot have for the last forecast day. It was fixed and every model was re-evaluated under the corrected methodology. §1–§3 record the audit of the pipeline as it was *before* that fix (measured on the single-LightGBM model of that time); **§4 records the final model decision: the deployed model is the 3-model LightGBM + CatBoost + XGBoost ensemble**, evaluated in [ENSEMBLE_CORRECTED_EVALUATION.md](ENSEMBLE_CORRECTED_EVALUATION.md).

## 1. Reproducibility (measured before the §3 correction)

The full pipeline was re-run from scratch into a separate directory and compared with the stored results.

| Claim | Stored | Re-run | Verdict |
|---|---|---|---|
| Training rows in the final model | 5,374,050 (119 origins) | 5,374,050 (119 origins) | reproduced |
| Features | 44 | 44 | reproduced |
| Horizon | 1–42 days, direct | (config) | as claimed |
| Primary RMSLE | 0.0947613532955437 | 0.0947613532955437 | **identical** |
| Primary WAPE / MAE / RMSE | 6.84% / 498.10 / 687.14 | identical to all digits | **identical** |
| Best-baseline (7-day rolling mean) WAPE | 17.78% | 17.78% | identical |
| Earlier windows RMSLE | 0.10350, 0.11898 | 0.10350, 0.11898 | identical |
| Model file | — | decompressed model text hash identical | **bit-identical model** |

One reproducibility defect was found and fixed: the gzip header embedded a timestamp and filename, so the same model produced a different `model_version` hash each run. `save_model` now writes a constant header; the model *content* was already identical.

## 2. Leakage audit

**Method 1 — independent recomputation.** For 25 randomly chosen stores × 12 random origins, lags (1, 7, 42), rolling mean/std (7/14/28/42) and store×weekday means were recomputed with plain pandas straight from the raw CSV, using only data on or before the origin, and compared with the repository's feature tables: **4,200 values, 0 mismatches** (including NaN placement for stores with gaps). Supervised rows were also checked: target date = origin + h, label equals the raw target, history features equal the origin row's.

**Method 2 — do the tests test what they claim? (mutation testing).** Four leaks were injected into scratch copies of the code, and the existing tests were run:

| Injected bug | Caught by |
|---|---|
| lag off by one (lag1 reads *tomorrow*) | `test_origin_features_are_causal` |
| centered rolling window (uses future days) | `test_origin_features_are_causal` |
| store×weekday mean shifted to include future days | `test_origin_features_are_causal` |
| training origins allowed up to the cut-off (targets reach into the validation window) | `test_training_targets_must_precede_validation_window` |

All four were detected. In addition `test_backtest_predictions_do_not_depend_on_window_actuals` rewrites every target inside the validation window, retrains, and requires identical predictions.

**Other checks.** Windows are chronological and non-overlapping (tested). Training/serving consistency: `tests/test_service.py` asserts that features rebuilt at request time equal the batch-training features; categorical levels are pinned; scoring one store alone equals scoring it in the full frame. `AppSessions` (train-only) is excluded. Seeds: LightGBM is seeded from the config; re-runs are bit-identical.

## 3. Finding and correction: the last-day schedule boundary

**What the feature is.** Each row carries the schedule flags (open, promotion, regional holiday, school closure) of the previous and next day (`*_prev1`, `*_next1`). The flags for the *target* day are known in advance; the neighbours are legitimate too — as long as they fall inside the published schedule.

**Is it available at prediction time?** The schedule is published `horizon` = 42 days ahead of the forecast origin (the API's snapshot ends at origin + 42). For the **42nd forecast day**, `*_next1` describes day 43 — which does **not** exist at inference. Days 1–41 are unaffected, and `*_prev1` looks backwards. Before the fix, training rows and back-test windows took the flag from data that continued past day 42, while the API passed a missing value: an information advantage in validation that the deployed system does not have.

**Size of the problem (old model, primary window).** Masking `*_next1` on day 42 at prediction time moved day-42 WAPE from 5.7% to 11.7% and overall WAPE from 6.84% to 7.03% (RMSLE 0.09476 → 0.09647).

**The fix.** `mask_unpublished_schedule` (`src/features/builder.py`) blanks every `*_nextK` feature whose day lies beyond the published schedule (`h + K > 42`). It runs inside the single dataset builder used by training, back-testing and the API, so all three see identical information by construction; the model now also *trains* with those values missing on `h = 42` rows and learns to handle them. Tests: `test_next_day_schedule_is_blank_when_it_would_not_be_published`, `test_masking_does_not_touch_features_when_disabled`, `test_last_forecast_day_uses_the_same_masked_schedule_as_training`.

**Complete re-run (all three windows, final model retrained) — before vs after.**

| Primary window | Pre-correction | Corrected |
|---|---|---|
| WAPE | 6.84% | **6.86%** |
| RMSLE | 0.09476 | **0.09472** |
| MAE | 498 | **500** |
| RMSE | 687 | **690** |
| Within ±10% | 76% | **75%** |
| Day-42 WAPE | 5.7% as validated (11.7% under the API's information) | **6.3%** |
| Earlier window 1 — RMSLE / WAPE | 0.10350 / 8.85% | 0.10304 / 8.76% |
| Earlier window 2 — RMSLE / WAPE | 0.11898 / 7.62% | 0.11817 / 7.58% |
| Best-baseline WAPE (primary) | 17.78% | 17.78% |

Because the corrected model is trained under the same restriction it is evaluated under, overall accuracy is essentially unchanged (WAPE 6.84% → 6.86%; within ±10%: 76% → 75%); the honest change is that day-42 accuracy is now measured the way the API delivers it, and it is in line with the other horizons (WAPE by week-ahead bucket ranges 6.3%–7.4%). The pre-correction artifacts are kept locally (not published).

## 4. Final model: the 3-model ensemble

Several boosting models were evaluated alone and as blends (LightGBM Tweedie, CatBoost on log1p demand, XGBoost Tweedie). The archived final recipe — blend weights 0.6 / 0.1 / 0.3, fixed round counts — was then evaluated under the **corrected** methodology on three chronological windows with all three models trained on identical data ([ENSEMBLE_CORRECTED_EVALUATION.md](ENSEMBLE_CORRECTED_EVALUATION.md)). No weights or tree counts were re-tuned.

| Corrected results | RMSLE | WAPE | MAE | RMSE |
|---|---|---|---|---|
| Primary window — LightGBM | 0.09472 | 6.865 % | 499.8 | 690.4 |
| Primary window — CatBoost | 0.10312 | 7.715 % | 561.7 | 783.8 |
| Primary window — XGBoost | 0.09550 | 6.955 % | 506.3 | 707.3 |
| **Primary window — ensemble** | **0.09410** | **6.825 %** | **496.9** | **688.1** |
| Pooled, 3 windows — LightGBM / ensemble | 0.10576 / **0.10462** | 7.749 % / **7.660 %** | 558.2 / **551.8** | 822.2 / **811.6** |

- **The ensemble had lower error than the corrected LightGBM in all three windows on all four metrics** (RMSLE −0.7 % to −1.9 %; WAPE −0.04 to −0.15 percentage points). It was the lowest-error of the four models in the primary window, in the oldest window and pooled; **in the middle window XGBoost alone was marginally lower** (RMSLE 0.10110 vs 0.10111). CatBoost was the highest-error model in every window.
- **Tuning caveat.** The round counts (1,160 / 1,320 / 590) and the weights were chosen on the primary window under the pre-correction methodology, so the primary-window score is a validation result, not an untouched test result. The scripts that produced those choices (the XGBoost validation run and the weight grid search) were not saved and were not reproduced; only the documented final recipes and blend were evaluated.
- **Uncertainty.** A paired store-level bootstrap (models fixed) gives 95 % intervals that exclude zero for the RMSLE, WAPE and MAE differences in every window; it reflects which stores are in a window only, not training randomness, window-to-window variation or the tuning caveat, so it is not evidence of a broader significance. No claim is made that the gain would persist elsewhere.
- **Cost (measured on one laptop).** Training ≈ 13–16× a LightGBM fit (CatBoost ≈ 84 % of it); the deployed real-data bundle's model files total 88.4 MB (LightGBM 8.4 MB gzipped, CatBoost 17.6 MB, XGBoost 62.4 MB UBJSON) versus about 21.7 MB raw / 8.4 MB gzipped for LightGBM alone; the whole API used 710 MB of process memory after start-up with the ensemble versus 548 MB with the single model (both include the 970k-row history); an uncached 42-day forecast took 23 ms versus 14 ms end to end (model time alone ≈ 15.6 vs 2.7 ms in a micro-benchmark); +448 MB of Linux dependencies. These were accepted in exchange for the lower validation error; compatibility with a given hosting plan must be checked against that plan's limits (not verified here).
- **Serving safety found during the evaluation.** XGBoost stores no category levels in the model; scoring a single-store request with per-frame `astype("category")` re-codes the levels and changed predictions by up to 21.6 % (10.5 % for the corrected model). The training levels are now persisted and pinned; tests assert single-store equals full-frame for every member and for the blend.
- **Reproducibility.** The archived ensemble was re-trained end to end with the unmodified archived code and reproduced its recorded validation RMSLE (0.09414 pre-correction) to five decimals; under the corrected methodology all nine (window × model) fits ran with identical training-frame hashes across models, and the worker's LightGBM equals the production pipeline's metrics to every digit.

## 5. Not audited / known gaps

Concept drift and real-world accuracy (no data after the history), prediction intervals, run-to-run determinism of CatBoost/XGBoost under the corrected methodology (one training run per model and window), and behaviour for brand-new stores (rejected by validation, no cold-start path).
