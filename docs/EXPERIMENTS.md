# Experiment log

A condensed record of what was tried while building the model. **The RMSLE figures in this file were measured before the last-day schedule correction described in [ML_AUDIT.md](ML_AUDIT.md) §3** (which changed the primary-window RMSLE only from 0.09476 to 0.09472 and does not alter any conclusion here). **Every number is a validation RMSLE on a single chronological hold-out (9 May – 19 Jun 2015, 42 days); lower is better.** Because the same window was used to pick between these options, differences of ~0.0003 are within noise and the *direction* of large differences is what matters. The full machine-readable log (87 rows, including runtimes) was kept from the original development history and is not part of the served project.

## Baselines (no learning)

| Baseline | RMSLE |
|---|---|
| Global average | 0.3873 |
| Store average | 0.2961 |
| Last known day | 0.3344 |
| Same weekday last week | 0.3290 |
| 7-day rolling average | **0.2331** (best) |

## Progression of the model

| Step | What changed | RMSLE | Verdict |
|---|---|---|---|
| v1 | Origin-anchored LightGBM (log1p target): lags, rolling stats of demand *and* app sessions, calendar, store metadata | 0.1228 | First learned model; ~47% better than best baseline |
| v2 | Origin stride 7 → 3 days (2× more rows) | 0.1230 | No gain, 2× cost → keep stride 7 |
| v3 | Drop app-session history features | 0.1201 | Better and simpler; app sessions unavailable in the future anyway |
| tune | 127 leaves, min_data_in_leaf 200, L2 1.0 | 0.1198 | Adopted |
| **v4** | Add lags 21/35/42, 42-day rolling stats, store×weekday baseline, previous/next-day schedule flags | **0.0970** | Largest single gain (−19%) |
| v5 | Add store×school-closure and store×promo baselines | 0.1001 | Worse → rejected (overfits) |
| v6 (×3) | Add "momentum" ratio of 7-day to 28-day mean (continuous, bucketed, strict min-periods) | 0.115 | Worse in all variants → rejected |
| **v7** | Tweedie objective on the raw target instead of log1p + RMSE | **0.0947** (power 1.05) | Adopted. Power search: 1.01 → 0.0950, 1.05 → 0.0947, 1.1 → 0.0950, 1.2 → 0.0953, 1.5 → 0.0959 |
| tune (Tweedie) | 63/127/255 leaves, learning rate 0.03 | 0.0950 – 0.0960 | No improvement over 127 leaves, lr 0.05 |
| v10 | 7-day schedule-density window instead of ±1 day | 0.0971 | Worse → rejected |
| v11 | Year and linear-trend features | 0.0960 – 0.1005 | Worse → rejected |
| v12 | Cross-store (network-wide) demand-level features | 0.0969 | Worse → rejected |

**Adopted configuration = v7** (`configs/default.yaml`). Re-running the refactored, modular pipeline in this repository reproduces the primary-window RMSLE of **0.09476** and the best baseline's **0.23307** (verified when the code was restructured).

## Ensembles (offline only, not served)

| Combination | RMSLE |
|---|---|
| LightGBM Tweedie alone | 0.0947 |
| + CatBoost (log1p) weighted blend | 0.0944 |
| LightGBM + CatBoost + XGBoost (0.6 / 0.1 / 0.3) | 0.0941 |
| Geometric-mean or ridge-stacked blends | 0.0941 |
| 5-seed bagging of one LightGBM | 0.0971 vs 0.0970 single (no gain) |

The best blend improves RMSLE by ~0.6% over the single model. The served model is the single LightGBM: three runtimes, three sets of dependencies and slower training were not worth a gain that small (and measured on the same window used to select the blend weights).

## Post-hoc calibration (not adopted)

The model under-forecasts slightly and consistently: the ratio of total actual to total predicted demand was between 1.012 and 1.021 on all three windows (the served model's reported bias on the primary window is −1.2%). A global 1.01× scale factor improved the offline blend from 0.09414 to 0.09371 on the primary window. It was **not adopted**: the gain is ~0.5% on the same window used to choose the factor, it patches a symptom (a level bias, plausibly from growth not captured by any feature) rather than a cause, and a calibration tuned this way cannot be confirmed without independent data. Candidate causes (trend features) were tested separately and made validation worse (see v11).

## Lessons

1. The largest gains came from *information the model was missing* (store×weekday baseline, more lags, neighbouring-day schedule) and from *matching the loss to the target* (Tweedie), not from tuning or ensembling.
2. Many "obvious" additions (interactions, momentum, trend, network features) hurt on held-out data. Adding features is not free: with 5M correlated rows, trees overfit them.
3. Validate on more than one window before trusting a small improvement — the three windows differ by up to ~0.025 RMSLE.
