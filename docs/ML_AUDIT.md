# ML audit

A pre-publication review of the modelling code: is there temporal leakage, do training and serving agree, and can the reported numbers be reproduced? Everything below was run against the actual code and the real dataset (kept local; see `data/README.md`). Scripts were one-off checks and are described, not shipped.

**Outcome:** the audit found one real problem (§3) — training and back-testing could see a next-day schedule flag that the API cannot have for the last forecast day. It was fixed and the complete validation was re-run; the README and all documents carry the *corrected* numbers. §1–§2 were measured on the pipeline as it was *before* that fix; §3 records both.

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

## 4. Ensemble decision

An offline blend of LightGBM (Tweedie) + XGBoost (Tweedie) + CatBoost (log1p target) was tuned on the primary window. *(Measured with the pre-correction LightGBM model; the corrected LightGBM's primary RMSLE, 0.09472, differs from it by 0.00004, so the conclusion is unchanged.)*

| | RMSLE | WAPE |
|---|---|---|
| LightGBM Tweedie (pre-correction model) | 0.09476 | 6.84% |
| Blend 0.6 / 0.3 / 0.1 | 0.09411 | 6.81% |

- **Size of the gain:** 0.68% relative RMSLE, **0.03 percentage points** of WAPE.
- **Is it real?** Holding the models fixed and resampling *stores*, the RMSLE improvement is 0.0006 (95% CI 0.0003–0.0010). But that interval ignores the largest sources of noise: the weights were tuned on the same window they are scored on; with weights tuned on half the stores and scored on the other half the gain averages 0.60% and ranges from −0.14% to +1.06% over 40 splits; and single-model results vary far more than 0.0006 from training randomness alone — five seeds of an earlier single-LightGBM configuration scored 0.0964 to 0.1078 (a spread ~17× the blend's gain).
- **Cost:** three model runtimes and dependency sets; the archived XGBoost and CatBoost models are 68 MB and 18 MB against a 9 MB LightGBM model; slower training.
- **Recommendation: keep the single LightGBM.** The evidence is not strong enough to justify the deployment complexity; the production path was not modified.

## 5. Not audited / known gaps

Concept drift and real-world accuracy (no data after the history), prediction intervals, and behaviour for brand-new stores (rejected by validation, no cold-start path).
