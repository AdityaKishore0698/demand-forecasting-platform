/**
 * Offline validation results on the REAL dataset (Rossmann Store Sales, not included in this repository).
 * Static, reported figures from docs/ENSEMBLE_CORRECTED_EVALUATION.md — they are NOT computed by this app, NOT produced
 * by the synthetic demo bundle, and NOT live production accuracy. Shown only in demo mode, in a card that says so.
 * The primary window also informed the choice of round counts and blend weights (validation, not an untouched test).
 */
export interface ReportedRow { window: string; ensemble: { rmsle: number; wape: number }; lightgbm: { rmsle: number; wape: number }; baselineWape: number }

export const REPORTED_PRIMARY = { rmsle: 0.09410, wape: 0.06825, mae: 496.88, rmse: 688.09 };

export const REPORTED_WINDOWS: ReportedRow[] = [
  { window: "Primary · 9 May – 19 Jun 2015", ensemble: { rmsle: 0.09410, wape: 0.06825 }, lightgbm: { rmsle: 0.09472, wape: 0.06865 }, baselineWape: 0.1778 },
  { window: "Earlier · 28 Mar – 8 May 2015", ensemble: { rmsle: 0.10111, wape: 0.08604 }, lightgbm: { rmsle: 0.10304, wape: 0.08758 }, baselineWape: 0.254 },
  { window: "Earlier · 14 Feb – 27 Mar 2015", ensemble: { rmsle: 0.11729, wape: 0.07512 }, lightgbm: { rmsle: 0.11817, wape: 0.07583 }, baselineWape: 0.184 },
];
