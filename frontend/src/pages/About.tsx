import { ArrowRight, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { useDatasetInfo, useMetrics, useModelInfo } from "../api/hooks";
import { Card } from "../components/ui/Card";
import { Reveal } from "../components/ui/Reveal";
import { Skeleton } from "../components/ui/States";
import { fmtDateYear, fmtInt, fmtPct } from "../lib/format";

const STACK: Array<{ title: string; items: string[] }> = [
  { title: "Modelling", items: ["Python", "pandas / NumPy", "LightGBM + CatBoost + XGBoost", "TreeSHAP + permutation importance"] },
  { title: "API", items: ["FastAPI", "Pydantic v2", "Uvicorn", "Model and data loaded once at start-up"] },
  { title: "Frontend", items: ["React 18 + TypeScript", "Vite", "Recharts", "TanStack Query", "Framer Motion"] },
  { title: "Quality", items: ["pytest (features, leakage, API)", "Vitest (formatting utilities)", "Central YAML config, fixed seed"] },
];

const FLOW = [
  ["Raw history", "Daily orders per store plus known schedule (open, promotion, holidays)"],
  ["Features", "Lags, rolling means, weekday baseline — frozen at the last known day"],
  ["Model", "Three boosted-tree models, blended, predict step 1…42 directly"],
  ["Validation", "Chronological hold-outs, compared to simple baselines"],
  ["API + dashboard", "FastAPI serves forecasts; React visualises them"],
];

function Section({ title, children, delay }: { title: string; children: ReactNode; delay: number }) {
  return (
    <Reveal delay={delay}>
      <Card title={title}><div className="prose">{children}</div></Card>
    </Reveal>
  );
}

export default function About() {
  const model = useModelInfo();
  const data = useDatasetInfo();
  const metrics = useMetrics();
  const d = data.data;
  const m = model.data;
  const mm = metrics.data;

  return (
    <div className="page">
      <Reveal>
        <div className="page__head">
          <div>
            <h1>About this project</h1>
            <p className="page__lede">
              An end-to-end demand forecasting workflow: from raw order history to a validated model, a REST API and this dashboard.
            </p>
          </div>
        </div>
      </Reveal>

      <Reveal delay={0.03}>
        <Card title="How it fits together">
          <div className="flow">
            {FLOW.map(([t, s], i) => (
              <div key={t} style={{ display: "contents" }}>
                <div className="flow__step"><b>{t}</b>{s}</div>
                {i < FLOW.length - 1 && <ArrowRight className="flow__arrow" size={16} aria-hidden />}
              </div>
            ))}
          </div>
        </Card>
      </Reveal>

      <div className="grid grid--2">
        <Section delay={0.06} title="The problem">
          <p>
            Given the daily order history of many stores, predict how many orders each store will receive on each of the next
            <strong> 1–42 days</strong>. Accurate forecasts let a business plan staffing and stock; forecasting badly means either idle
            capacity or missed orders.
          </p>
          <p>
            Stores follow strong weekly rhythms, are sometimes closed, and respond to promotions and holidays — so the model is given the
            calendar and promotion schedule that is known in advance.
          </p>
        </Section>

        <Section delay={0.08} title="The data">
          {!d ? <Skeleton h={120} r={10} /> : (
            <>
              <p>
                <strong>{fmtInt(d.train_rows)}</strong> daily records for <strong>{fmtInt(d.n_hubs)}</strong> stores, from{" "}
                {fmtDateYear(d.train_start)} to {fmtDateYear(d.train_end)}. About {fmtPct(d.closed_day_share, 0)} of store-days are closed
                (and demand is always zero on those days); {d.hubs_with_history_gaps} stores have a gap in their history.
              </p>
              <p>
                Average demand on open days is about {fmtInt(d.mean_open_day_demand)} orders. Promotion days average{" "}
                {d.promo_uplift_ratio ? `${((d.promo_uplift_ratio - 1) * 100).toFixed(0)}% more` : "a different level"} than regular days.
                The forecast window is {fmtDateYear(d.forecast_start)} – {fmtDateYear(d.forecast_end)}.
              </p>
              <p style={{ color: "var(--text-muted)" }}>
                This is a fixed historical dataset, so the dashboard forecasts the six weeks after the data ends — it is not connected to live orders.
              </p>
            </>
          )}
        </Section>

        <Section delay={0.1} title="The model">
          {!m ? <Skeleton h={120} r={10} /> : (
            <>
              {m.model_type_id === "boosting_ensemble" && m.weights && m.components ? (
                <p>
                  A blend of <strong>three gradient-boosted tree models</strong> ({m.n_features} features):{" "}
                  {m.components.map((c, i) => (
                    <span key={c.name}>{i > 0 && ", "}<strong>{c.family}</strong> ({(c.weight * 100).toFixed(0)}%{c.n_trees ? `, ${fmtInt(c.n_trees)} trees` : ""})</span>
                  ))}
                  , combined as a weighted sum of their demand forecasts. LightGBM and XGBoost use a <strong>Tweedie</strong> loss, which suits
                  non-negative, right-skewed counts with many zeros; CatBoost learns log(1 + demand). The ensemble adds model footprint and inference
                  work compared with a single model, in exchange for a small validation-error reduction.
                </p>
              ) : (
                <p>
                  A <strong>LightGBM</strong> gradient-boosted tree model ({m.n_estimators} trees, {m.n_features} features) with a
                  <strong> Tweedie</strong> loss, which suits non-negative, right-skewed counts that include many zeros.
                </p>
              )}
              <p>
                It forecasts <strong>directly</strong>: all history features are computed once at the last known day, and the number of days
                ahead is itself a feature. That avoids compounding errors from feeding predictions back in as inputs. Stores scheduled to be
                closed are forecast as zero.
              </p>
              <p style={{ color: "var(--text-muted)" }}>
                Version <span className="mono">{m.model_version}</span> · trained on {fmtInt(m.training.n_rows)} rows from weekly forecast origins.
              </p>
            </>
          )}
        </Section>

        <Section delay={0.12} title="Validation">
          {!mm ? <Skeleton h={120} r={10} /> : (
            <>
              <p>
                Random splits would leak the future into training, so validation is <strong>chronological</strong>: train on data before a cut-off
                and forecast the following 42 days. This is repeated on {mm.windows.length} consecutive windows.
              </p>
              <p>
                On the most recent window the model's WAPE is <strong>{fmtPct(mm.model.wape)}</strong> versus{" "}
                <strong>{fmtPct(mm.baselines[mm.best_baseline].wape)}</strong> for the best simple baseline. Across windows it ranges from{" "}
                {fmtPct(Math.min(...mm.windows.map((w) => w.model.wape ?? 1)))} to {fmtPct(Math.max(...mm.windows.map((w) => w.model.wape ?? 0)))}.
              </p>
            </>
          )}
        </Section>
      </div>

      <Reveal delay={0.14}>
        <Card title="Tech stack">
          <div className="tech-grid">
            {STACK.map((s) => (
              <div className="tech-card" key={s.title}>
                <h3>{s.title}</h3>
                <div className="tag-row">{s.items.map((i) => <span className="chip" key={i}>{i}</span>)}</div>
              </div>
            ))}
          </div>
        </Card>
      </Reveal>

      <Reveal delay={0.16}>
        <Card title="Limitations" info="What this project does not claim.">
          <ul className="list">
            <li>Future accuracy cannot be measured: actuals for the forecast window are not in the data. The reported error comes from earlier hold-out windows.</li>
            <li>The most recent hold-out window was also used to tune the model, so it is a validation figure, not a fully untouched test.</li>
            <li>Forecasts use only the schedule known in advance (open days, promotions, holidays). Weather, competitor actions and price changes are not modelled.</li>
            <li>Forecasts are point estimates. There are no prediction intervals.</li>
            <li>The ensemble's blended explanation is an approximation (each model's own TreeSHAP is exact); the ensemble adds model footprint and inference work compared with a single model.</li>
            <li>Horizon is capped at 42 days — the length of the known schedule. Longer horizons would require inventing inputs.</li>
            <li>One global model (the ensemble) serves every store. Stores with short or gapped history rely more on the weekday baseline.</li>
            <li>No live drift monitoring or automated retraining is implemented.</li>
          </ul>
          <div className="callout callout--warn" style={{ marginTop: 14 }}>
            <TriangleAlert aria-hidden />
            <div>Treat the model as a strong, well-validated baseline — not as a guarantee for any single store on any single day.</div>
          </div>
        </Card>
      </Reveal>
    </div>
  );
}
