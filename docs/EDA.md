# Exploratory data analysis

The numbers below were computed on the original dataset (not redistributed; see `data/README.md`). `make eda` (`scripts/eda.py`) regenerates six figures from *your* copy of the data into `docs/figures/`; that folder is git-ignored, so the figures are not part of the published repository. In the raw files each store is identified by `HubID`.

## Dataset

| Fact | Value |
|---|---|
| Training records | 970,379 store-days |
| Stores | 1,115 |
| Period | 1 Jan 2013 – 19 Jun 2015 (900 calendar days) |
| Forecast window (schedule only, no demand) | 20 Jun – 31 Jul 2015 (42 days, 46,830 rows) |
| Stores with a gap in their history | 181 (almost all a single 184-day gap) |
| Closed store-days | 17.1% — demand is **always 0** on closed days |
| Open days with zero demand | 54 (negligible) |
| Mean / median open-day demand | 6,954 / 6,368 orders |
| Store-level average, P10 / P50 / P90 | 4,510 / 6,599 / 9,635 |
| Skewness of open-day demand | 1.59 (right-skewed) |

## Findings and what each meant for the model

**1. A large share of zeros — but they are structural, not random.** Every zero-demand row is a closed day (except 54 open days). `IsOpen` is known for the future, so closed days are forecast as exactly 0 by rule, and the model is evaluated on open days for the percentage metrics (dividing by zero is otherwise meaningless).

**2. Strong weekly rhythm, and Sundays are different.** Open-day demand is highest on Monday (8,216) and lowest on Saturday (5,886). Only 2.5% of Sunday store-days are open, and those stores behave differently (Sunday open-day mean 8,208). → `Weekday`, a store×weekday historical baseline (`OV_hubweekday_mean`) and seasonal lags (7, 14, … 42 days) are included.

**3. Promotions lift demand.** Open-day demand is 1.39× higher on promotion days. 35.7% of the forecast window's days have a promotion, and the schedule is known in advance → promotion flags (and neighbouring days' flags) are features.

**4. Stores differ a lot in level.** The coefficient of variation of store averages is 0.34; P10–P90 spans 4,510 to 9,635. → store identity, store attributes and a store-level baseline matter; errors are also reported by store-volume tercile.

**5. Total demand has seasonality and level shifts.** Aggregate demand shows recurring peaks and year-on-year variation.

**6. Not every store has a complete history.** Between 935 and 1,115 stores report on any given date; 181 stores are missing a stretch of ~6 months. → the pipeline works on a full daily calendar per store so lags and rolling windows never silently use the wrong dates, and lets missing values propagate as `NaN` (LightGBM handles them natively).

**7. One column is unusable for forecasting.** `AppSessions` (a usage signal) exists only in the training file. Even if predictive in-sample, it is not available for future days, so it is excluded. (An earlier experiment that used only its *lagged* history found no benefit — RMSLE 0.12281 with vs 0.12013 without.)

## Not investigated

Weather, prices, competitor events and macro-economic drivers are not in the data and are not modelled. Holidays appear in the future window at a rate of 0%, so their learned effect is only exercised in back-tests.
