import { Activity, CalendarRange, Gauge, Store, TrendingUp } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useForecast, useHistory } from "../api/hooks";
import { ForecastChart } from "../components/charts/ForecastChart";
import { RecentDemandChart } from "../components/charts/RecentDemandChart";
import { TrendChart } from "../components/charts/TrendChart";
import { WeeklyPatternChart } from "../components/charts/WeeklyPatternChart";
import { Card } from "../components/ui/Card";
import { KpiCard } from "../components/ui/KpiCard";
import { Legend } from "../components/ui/Legend";
import { Reveal } from "../components/ui/Reveal";
import { ChartSkeleton, EmptyState, ErrorState } from "../components/ui/States";
import { useApp } from "../context/AppContext";
import { useChartColors } from "../lib/chart";
import { meanOpen, rollingOpenMean, weeklyPattern } from "../lib/analytics";
import { fmtDateYear, fmtInt, fmtPct } from "../lib/format";

const CONTEXT_DAYS = 120;

export default function Overview() {
  const { storeId, store, horizon, stores, storesError, refetchStores } = useApp();
  const history = useHistory(storeId);
  const forecast = useForecast(storeId, horizon);
  const colors = useChartColors();

  const points = history.data?.points;
  const recentPoints = useMemo(() => (points ?? []).slice(-CONTEXT_DAYS), [points]);
  const last28 = useMemo(() => (points ?? []).slice(-28), [points]);
  const weekly = useMemo(() => weeklyPattern(points ?? []), [points]);
  const trend = useMemo(() => rollingOpenMean(points ?? [], 28), [points]);
  const recentAvg = useMemo(() => meanOpen(last28), [last28]);

  const hist = store?.avg_open_day_demand ?? null;
  const fcMean = forecast.data?.summary.mean_open_day ?? null;
  const lift = hist && fcMean != null ? fcMean / hist - 1 : null;

  if (storesError) {
    return (
      <div className="page">
        <div className="page__head"><div><h1>Overview</h1></div></div>
        <ErrorState error={storesError} onRetry={refetchStores} />
      </div>
    );
  }

  return (
    <div className="page">
      <Reveal>
        <div className="page__head">
          <div>
            <h1>Overview</h1>
            <p className="page__lede">
              Recent demand for the selected store, followed by the model's forecast for the next {horizon} days.
            </p>
          </div>
        </div>
      </Reveal>

      <Reveal delay={0.04} className="grid grid--kpi">
        <KpiCard label="Forecast horizon" icon={CalendarRange} tone="forecast" value={horizon} format={(v) => `${Math.round(v)}`} unit="days"
          sub={forecast.data ? `${fmtDateYear(forecast.data.forecast[0].date)} → ${fmtDateYear(forecast.data.forecast[forecast.data.forecast.length - 1].date)}` : "—"}
          loading={forecast.isPending} info="How many days ahead the forecast covers. The model supports 1–42 days." />
        <KpiCard label="Selected store" icon={Store} text={store ? `Store ${store.store_id}` : "—"} loading={!store}
          sub={store ? `Ranks #${store.demand_rank} of ${fmtInt(stores.length)} by average demand` : undefined}
          info="Stores are ranked by average orders on days they are open." />
        <KpiCard label="Avg historical demand" icon={Activity} value={hist} format={fmtInt} unit="orders" loading={!store}
          sub={recentAvg != null ? `per open day · last 28 days: ${fmtInt(recentAvg)}` : "per open day"}
          info="Mean daily orders over the store's full recorded history, counting open days only." />
        <KpiCard label="Forecasted demand" icon={TrendingUp} tone="forecast" value={fcMean} format={fmtInt} unit="orders"
          loading={forecast.isPending} delta={lift != null ? { value: lift, label: "vs history" } : undefined}
          sub="per open day, vs historical avg"
          info="Average predicted orders over the open days in the forecast window." />
        <KpiCard label="Model error" icon={Gauge} tone="positive" value={store?.backtest_wape ?? null} format={(v) => fmtPct(v, 1)}
          loading={!store} sub="WAPE on a 6-week hold-out"
          info="Weighted absolute percentage error for this store on the most recent chronological hold-out window (open days). Lower is better. It is a validation figure, not a guarantee for future dates." />
      </Reveal>

      <Reveal delay={0.08}>
        <Card title="Demand history and forecast"
          subtitle={`Last ${CONTEXT_DAYS} days of recorded demand, then the ${horizon}-day forecast`}
          actions={<Legend items={[
            { label: "Recorded demand", color: colors.history },
            { label: "Forecast", color: colors.forecast, kind: "dash" },
          ]} />}>
          {history.isPending || forecast.isPending ? <ChartSkeleton height={340} />
            : history.isError ? <ErrorState error={history.error} onRetry={() => history.refetch()} />
            : forecast.isError ? <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
            : !forecast.data || !history.data ? <EmptyState title="No data for this store" />
            : <ForecastChart history={recentPoints} forecast={forecast.data.forecast} showClosed={false} />}
          {forecast.data && (
            <p className="card__sub" style={{ marginTop: 10 }}>
              Closed days are hidden here (demand is zero by rule).{" "}
              <Link to="/forecast" style={{ color: "var(--accent)", fontWeight: 560 }}>See the day-by-day forecast →</Link>
            </p>
          )}
        </Card>
      </Reveal>

      <Reveal delay={0.12} className="grid grid--2">
        <Card title="Demand trend" subtitle="28-day average of open-day demand"
          info="Smooths out day-to-day noise so level shifts and seasonality are visible. The dashed line is the forecast average.">
          {history.isPending ? <ChartSkeleton height={230} />
            : history.isError ? <ErrorState error={history.error} onRetry={() => history.refetch()} />
            : <TrendChart data={trend} forecastMean={fcMean} />}
        </Card>
        <Card title="Weekly pattern" subtitle="Average orders per weekday (open days)"
          info="Some stores are closed on particular weekdays; those bars are faded.">
          {history.isPending ? <ChartSkeleton height={230} />
            : history.isError ? <ErrorState error={history.error} onRetry={() => history.refetch()} />
            : <WeeklyPatternChart data={weekly} />}
        </Card>
      </Reveal>

      <Reveal delay={0.16}>
        <Card title="Recent demand" subtitle="Last 28 recorded days — promotion days highlighted, closed days faded">
          {history.isPending ? <ChartSkeleton height={230} />
            : history.isError ? <ErrorState error={history.error} onRetry={() => history.refetch()} />
            : <RecentDemandChart data={last28} />}
        </Card>
      </Reveal>
    </div>
  );
}
