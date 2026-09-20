import { Info } from "lucide-react";
import { useState } from "react";
import { useApp } from "../context/AppContext";
import { useBacktest, useHealth, useMetrics } from "../api/hooks";
import type { MetricMap, MetricsResponse } from "../api/types";
import { ActualVsPredictedChart } from "../components/charts/ActualVsPredictedChart";
import { ErrorHistogram } from "../components/charts/ErrorHistogram";
import { GroupBarChart } from "../components/charts/GroupBarChart";
import { Card } from "../components/ui/Card";
import { InfoTip } from "../components/ui/InfoTip";
import { Legend } from "../components/ui/Legend";
import { Reveal } from "../components/ui/Reveal";
import { Segmented } from "../components/ui/Segmented";
import { ChartSkeleton, ErrorState, QueryBoundary, Skeleton } from "../components/ui/States";
import { Switch } from "../components/ui/Switch";
import { useChartColors } from "../lib/chart";
import { REPORTED_PRIMARY, REPORTED_WINDOWS } from "../lib/reportedResults";
import { fmtDateYear, fmtInt, fmtPct, fmtSignedPct } from "../lib/format";

type Scope = "all" | "store";
type Breakdown = "by_horizon" | "by_weekday" | "by_hub_tier" | "by_promo";

const BREAKDOWNS: Array<{ value: Breakdown; label: string; note: string }> = [
  { value: "by_horizon", label: "Days ahead", note: "Error by how far ahead the forecast is (1–7, 8–14, … days)." },
  { value: "by_weekday", label: "Weekday", note: "Error by day of week." },
  { value: "by_hub_tier", label: "Store volume", note: "Stores split into terciles by historical average demand." },
  { value: "by_promo", label: "Promotion", note: "Error on promotion days versus regular days." },
];

function MetricTile({ label, value, base, info, lowerIsBetter = true }: {
  label: string; value: string; base?: string; info: string; lowerIsBetter?: boolean;
}) {
  return (
    <div className="card card--hover" style={{ padding: "14px 16px" }}>
      <div className="kpi__label">{label}<InfoTip text={info} /></div>
      <div className="kpi__value num" style={{ marginTop: 6 }}>{value}</div>
      {base && <div className="kpi__sub">{base}<span style={{ marginLeft: 6, opacity: 0.7 }}>{lowerIsBetter ? "lower is better" : ""}</span></div>}
    </div>
  );
}

function MetricGrid({ m, s }: { m: MetricsResponse; s: MetricMap }) {
  const b = m.baselines[m.best_baseline];
  const bl = `${b.label}: `;
  return (
    <div className="grid grid--3">
      <MetricTile label="MAE" value={`${fmtInt(s.mae)}`} base={`${bl}${fmtInt(b.mae)}`}
        info="Mean absolute error in orders per store-day (open days). The average size of a miss." />
      <MetricTile label="RMSE" value={`${fmtInt(s.rmse)}`} base={`${bl}${fmtInt(b.rmse)}`}
        info="Root mean squared error in orders per store-day (open days). Penalises large misses more than MAE." />
      <MetricTile label="WAPE" value={fmtPct(s.wape)} base={`${bl}${fmtPct(b.wape)}`}
        info="Total absolute error ÷ total actual orders (open days). A volume-weighted percentage error." />
      <MetricTile label="RMSPE" value={fmtPct(s.rmspe)} base={`${bl}${fmtPct(b.rmspe)}`}
        info="Root mean squared percentage error. Reported for comparison only: it is dominated by low-volume days and undefined at zero." />
      <MetricTile label="RMSLE" value={s.rmsle == null ? "—" : s.rmsle.toFixed(4)} base={`${bl}${b.rmsle?.toFixed(4)}`}
        info="Root mean squared log error over all store-days — the metric the model was tuned on. Measures relative error." />
      <MetricTile label="Bias" value={fmtSignedPct(s.bias)} lowerIsBetter={false} base="Total predicted ÷ total actual − 1"
        info="Negative means the model under-forecasts in total over the window." />
    </div>
  );
}

export default function ModelPerformance() {
  const { storeId } = useApp();
  const colors = useChartColors();
  const [scope, setScope] = useState<Scope>("all");
  const [showBase, setShowBase] = useState(true);
  const [bd, setBd] = useState<Breakdown>("by_horizon");
  const metrics = useMetrics();
  const demo = useHealth().data?.data_label === "synthetic-demo";
  const backtest = useBacktest(scope === "store" ? storeId : null);
  const bdInfo = BREAKDOWNS.find((x) => x.value === bd)!;

  return (
    <div className="page">
      <Reveal>
        <div className="page__head">
          <div>
            <h1>Model performance</h1>
            <p className="page__lede">
              How well the model forecast a period it had never seen: it was trained only on data before that period, then asked to
              predict the next 42 days.
            </p>
            <p style={{ marginTop: 8 }}>
              {demo
                ? <span className="chip chip--warn" style={{ whiteSpace: "normal", borderRadius: 12, lineHeight: 1.35, padding: "4px 10px" }}>Synthetic demo — the numbers below come from generated data and a model trained on it</span>
                : <span className="chip chip--accent" style={{ whiteSpace: "normal", borderRadius: 12, lineHeight: 1.35, padding: "4px 10px" }}>Offline validation on the dataset the model was trained on — not live production accuracy</span>}
            </p>
          </div>
        </div>
      </Reveal>

      {demo && (
        <Reveal delay={0.02}>
          <Card title="Reported offline validation on the real dataset"
            subtitle="Rossmann Store Sales (not included in this repository). Reported figures, not computed by this demo, and not live production accuracy.">
            <div className="table-wrap" style={{ maxHeight: "none", borderTop: 0 }}>
              <table className="table">
                <thead><tr><th>Window (42 days)</th><th className="r">Ensemble RMSLE</th><th className="r">Ensemble WAPE</th><th className="r">LightGBM alone WAPE</th><th className="r">Best simple baseline WAPE</th></tr></thead>
                <tbody>
                  {REPORTED_WINDOWS.map((w) => (
                    <tr key={w.window}>
                      <td>{w.window}</td><td className="r num"><b>{w.ensemble.rmsle.toFixed(4)}</b></td><td className="r num"><b>{fmtPct(w.ensemble.wape, 3)}</b></td>
                      <td className="r num">{fmtPct(w.lightgbm.wape, 3)}</td><td className="r num">{fmtPct(w.baselineWape)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="card__sub" style={{ marginTop: 8 }}>
              Primary window: MAE {fmtInt(REPORTED_PRIMARY.mae)}, RMSE {fmtInt(REPORTED_PRIMARY.rmse)} orders per store-day. The primary window also informed the choice of round counts and
              blend weights, so it is a validation result, not an untouched test. See the README for the full comparison.
            </p>
          </Card>
        </Reveal>
      )}

      <QueryBoundary query={metrics} skeleton={<div className="grid grid--3">{Array.from({ length: 6 }, (_, i) => <div key={i} className="card" style={{ padding: 16 }}><Skeleton h={62} r={10} /></div>)}</div>}>
        {(m) => {
          const w = m.primary_window;
          const overall = m.model.wape;
          return (
            <>
              <Reveal delay={0.04}>
                <div className="callout">
                  <Info aria-hidden />
                  <div>
                    <b>Hold-out window: {fmtDateYear(w.start)} → {fmtDateYear(w.end)}</b> ({w.horizon} days, {fmtInt(m.model.n_open_rows)} open store-days).
                    These are validation metrics on hold-out windows, not live production accuracy. Metrics are on open days unless noted. Comparison baseline: <b>{m.baselines[m.best_baseline].label}</b>, the strongest simple
                    method on this window. The model cut WAPE by <b>{m.improvement_vs_best_baseline_pct.wape?.toFixed(0)}%</b> relative to it.
                  </div>
                </div>
              </Reveal>

              <Reveal delay={0.06}><MetricGrid m={m} s={m.model} /></Reveal>

              {m.members && Object.keys(m.members).length > 0 && (
                <Reveal delay={0.07}>
                  <Card title="Ensemble members" subtitle="Each model scored on exactly the same hold-out rows; the ensemble is their fixed-weight blend (weights set in advance, not tuned on these numbers)">
                    <div className="table-wrap" style={{ maxHeight: "none", borderTop: 0 }}>
                      <table className="table">
                        <thead><tr><th>Model</th><th className="r">RMSLE</th><th className="r">WAPE</th><th className="r">MAE</th><th className="r">RMSE</th></tr></thead>
                        <tbody>
                          {[...Object.entries(m.members), ["ensemble", m.model] as [string, MetricMap]].map(([name, v]) => (
                            <tr key={name} className={name === "ensemble" ? "is-selected" : ""}>
                              <td><b>{name === "ensemble" ? "Ensemble" : name === "lightgbm" ? "LightGBM" : name === "catboost" ? "CatBoost" : name === "xgboost" ? "XGBoost" : name}</b></td>
                              <td className="r num">{v.rmsle?.toFixed(4)}</td><td className="r num">{fmtPct(v.wape)}</td>
                              <td className="r num">{fmtInt(v.mae)}</td><td className="r num">{fmtInt(v.rmse)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </Reveal>
              )}

              <Reveal delay={0.09}>
                <Card title="Actual vs predicted"
                  subtitle={backtest.data ? `${backtest.data.scope} · daily orders on the hold-out window` : "Daily orders on the hold-out window"}
                  actions={
                    <>
                      <Segmented ariaLabel="Scope" value={scope} onChange={setScope}
                        options={[{ value: "all", label: "All stores" }, { value: "store", label: storeId ? `Store ${storeId}` : "Store" }]} />
                      <Switch label="Baseline" checked={showBase} onChange={setShowBase} />
                    </>
                  }>
                  {backtest.isPending ? <ChartSkeleton height={320} />
                    : backtest.isError ? <ErrorState error={backtest.error} onRetry={() => backtest.refetch()} />
                    : (
                      <>
                        <Legend items={[
                          { label: "Actual", color: colors.actual },
                          { label: "Model", color: colors.forecast },
                          ...(showBase ? [{ label: backtest.data.baseline_label, color: colors.baseline, kind: "dash" as const }] : []),
                        ]} />
                        <ActualVsPredictedChart points={backtest.data.points} showBaseline={showBase} baselineLabel={backtest.data.baseline_label} />
                        <p className="card__sub" style={{ marginTop: 8 }}>
                          {scope === "all" ? "Each point sums all stores for that day." : "Closed days are zero for both series."}{" "}
                          WAPE for this view: <b>{fmtPct(backtest.data.metrics.wape)}</b> (baseline {fmtPct(backtest.data.baseline_metrics.wape)}).
                        </p>
                      </>
                    )}
                </Card>
              </Reveal>

              <Reveal delay={0.12} className="grid grid--2">
                <Card title="Error distribution" subtitle="Per-forecast percentage error, open days"
                  info="Each forecast's (predicted − actual) ÷ actual. A centred, narrow histogram means unbiased and consistent.">
                  <ErrorHistogram edges={m.error_distribution.bin_edges} counts={m.error_distribution.counts} />
                  <p className="card__sub" style={{ marginTop: 8 }}>
                    <b>{fmtPct(m.error_distribution.share_within_10pct, 0)}</b> of forecasts are within ±10% of actual and{" "}
                    <b>{fmtPct(m.error_distribution.share_within_20pct, 0)}</b> within ±20%. Median error{" "}
                    {fmtSignedPct(m.error_distribution.median_pct_error)}. Extremes beyond the axis are clipped into the edge bins.
                  </p>
                </Card>
                <Card title="Where the errors are" subtitle={bdInfo.note}
                  actions={<Segmented ariaLabel="Breakdown" value={bd} onChange={setBd} options={BREAKDOWNS.map((x) => ({ value: x.value, label: x.label }))} />}>
                  <GroupBarChart data={m.breakdowns[bd]} overall={overall} />
                </Card>
              </Reveal>

              <Reveal delay={0.15} className="grid grid--2">
                <Card title="Against simple baselines" subtitle="WAPE, primary window (lower is better)"
                  info="A forecasting model should be judged against simple rules, not against nothing.">
                  {(() => {
                    const rows = [
                      ...Object.entries(m.baselines).map(([k, v]) => ({ k, label: v.label, v: v.wape ?? 0, model: false })),
                      { k: "model", label: m.model_label ?? "Model", v: m.model.wape ?? 0, model: true },
                    ].sort((a, b) => b.v - a.v);
                    const max = Math.max(...rows.map((r) => r.v));
                    return rows.map((r) => (
                      <div key={r.k} className={`bar-row ${r.model ? "is-model" : ""}`}>
                        <span>{r.label}</span>
                        <div className="bar-row__track"><div className="bar-row__fill" style={{ width: `${(r.v / max) * 100}%` }} /></div>
                        <span className="num" style={{ textAlign: "right" }}>{fmtPct(r.v)}</span>
                      </div>
                    ));
                  })()}
                </Card>

                <Card title="Robustness across windows" subtitle="The same procedure repeated on earlier 6-week periods"
                  info="Each window trains only on data before it. The primary window was also used for tuning, so the earlier windows are the fairer robustness check.">
                  <div className="table-wrap" style={{ maxHeight: "none", borderTop: 0 }}>
                    <table className="table">
                      <thead><tr><th>Window</th><th className="r">RMSLE</th><th className="r">WAPE</th><th className="r">Best baseline WAPE</th></tr></thead>
                      <tbody>
                        {m.windows.map((x) => (
                          <tr key={x.name}>
                            <td>{fmtDateYear(x.start)} → {fmtDateYear(x.end)}{x.name === "primary" && <span className="chip chip--accent" style={{ marginLeft: 8 }}>primary</span>}</td>
                            <td className="r num">{x.model.rmsle?.toFixed(4)}</td>
                            <td className="r num"><b>{fmtPct(x.model.wape)}</b></td>
                            <td className="r num">{fmtPct(x.best_baseline_metrics.wape)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </Reveal>

              <Reveal delay={0.18}>
                <Card title="How this was validated" subtitle="Time-aware, no shuffling">
                  <div className="timeline" style={{ ["--train" as string]: "6fr", ["--hold" as string]: "1fr" }}>
                    <div className="timeline__seg timeline__train"><b>Training data</b>Only days before {fmtDateYear(w.start)}: the last training example's target date is on or before the cut-off</div>
                    <div className="timeline__seg timeline__hold"><b>Hold-out</b>{w.horizon} days</div>
                  </div>
                  <ul className="list" style={{ marginTop: 12 }}>
                    {m.validation_notes.map((n) => <li key={n}>{n}</li>)}
                    <li>
                      In-sample check: on a {fmtInt(m.training_in_sample.n_rows)}-row sample of training data the model reaches WAPE{" "}
                      {fmtPct(m.training_in_sample.wape)} vs {fmtPct(m.model.wape)} on the hold-out — a modest gap between training and unseen data.
                    </li>
                  </ul>
                </Card>
              </Reveal>
            </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
