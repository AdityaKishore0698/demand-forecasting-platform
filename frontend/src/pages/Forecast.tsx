import { ArrowDownToLine, CalendarCheck2, Download, Info, Sigma, TrendingDown, TrendingUp } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { useForecast, useHistory } from "../api/hooks";
import type { ForecastPoint } from "../api/types";
import { DriversPanel } from "../components/charts/DriversPanel";
import { ForecastChart } from "../components/charts/ForecastChart";
import { Card } from "../components/ui/Card";
import { Drawer } from "../components/ui/Drawer";
import { Legend } from "../components/ui/Legend";
import { Reveal } from "../components/ui/Reveal";
import { Segmented } from "../components/ui/Segmented";
import { ChartSkeleton, EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { Switch } from "../components/ui/Switch";
import { useApp } from "../context/AppContext";
import { useChartColors } from "../lib/chart";
import { downloadCsv } from "../lib/csv";
import { fmtDate, fmtDateLong, fmtInt } from "../lib/format";

const CONTEXT_OPTIONS = [
  { value: 60, label: "60 days" },
  { value: 120, label: "120 days" },
  { value: 365, label: "1 year" },
];

function Tile({ icon: Icon, label, value, sub, tone }: {
  icon: typeof TrendingUp; label: string; value: string; sub?: string; tone: "peak" | "low" | "avg" | "total";
}) {
  const bg = { peak: "var(--forecast-soft)", low: "var(--accent-soft)", avg: "var(--surface-3)", total: "var(--surface-3)" }[tone];
  const fg = { peak: "var(--forecast)", low: "var(--accent)", avg: "var(--text-2)", total: "var(--text-2)" }[tone];
  return (
    <div className="card tile card--hover">
      <span className="tile__ico" style={{ background: bg, color: fg }}><Icon aria-hidden /></span>
      <div>
        <div className="tile__label">{label}</div>
        <div className="tile__value num">{value}</div>
        {sub && <div className="tile__sub">{sub}</div>}
      </div>
    </div>
  );
}

function ContextChips({ p }: { p: ForecastPoint }) {
  if (!p.is_open) return <span className="chips"><span className="chip chip--closed">Closed</span></span>;
  const chips = [
    p.promo_active && <span key="p" className="chip chip--promo">Promotion</span>,
    p.holiday && <span key="h" className="chip chip--holiday">Holiday</span>,
    p.school_closure && <span key="s" className="chip chip--school">School closure</span>,
  ].filter(Boolean);
  return <span className="chips">{chips.length ? chips : <span className="chip chip--open">Regular day</span>}</span>;
}

export default function Forecast() {
  const { storeId, horizon, storesError, refetchStores } = useApp();
  const colors = useChartColors();
  const [context, setContext] = useState(120);
  const [showClosed, setShowClosed] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const history = useHistory(storeId);
  const forecast = useForecast(storeId, horizon);
  const points = history.data?.points;
  const shown = useMemo(() => (points ?? []).slice(-context), [points, context]);
  const close = useCallback(() => setSelected(null), []);

  const rows = forecast.data?.forecast ?? [];
  const s = forecast.data?.summary;
  const selectedRow = rows.find((r) => r.date === selected);

  if (storesError) {
    return (
      <div className="page">
        <div className="page__head"><div><h1>Forecast</h1></div></div>
        <ErrorState error={storesError} onRetry={refetchStores} />
      </div>
    );
  }

  const exportCsv = () => {
    if (!forecast.data) return;
    downloadCsv(`forecast_store${storeId}_${horizon}d.csv`, rows.map((r) => ({
      date: r.date, weekday: r.weekday, predicted_demand: r.predicted_demand, open: r.is_open,
      promotion: r.promo_active, holiday: r.holiday, school_closure: r.school_closure,
    })));
  };

  return (
    <div className="page">
      <Reveal>
        <div className="page__head">
          <div>
            <h1>Forecast</h1>
            <p className="page__lede">
              Day-by-day predicted demand for Store {storeId ?? "—"}. Days the store is scheduled to be closed are forecast as zero.
            </p>
          </div>
          <button className="btn" onClick={exportCsv} disabled={!forecast.data}><Download aria-hidden /> Export CSV</button>
        </div>
      </Reveal>

      <Reveal delay={0.04} className="grid grid--kpi4">
        {forecast.isPending ? Array.from({ length: 4 }, (_, i) => <div key={i} className="card tile"><Skeleton h={38} w="100%" /></div>) : s && (
          <>
            <Tile tone="peak" icon={TrendingUp} label="Peak day" value={s.peak ? fmtInt(s.peak.value) : "—"} sub={s.peak ? fmtDate(s.peak.date) : undefined} />
            <Tile tone="low" icon={TrendingDown} label="Lowest open day" value={s.lowest_open_day ? fmtInt(s.lowest_open_day.value) : "—"} sub={s.lowest_open_day ? fmtDate(s.lowest_open_day.date) : undefined} />
            <Tile tone="avg" icon={Sigma} label="Average (open days)" value={fmtInt(s.mean_open_day)} sub={`${s.open_days} open · ${s.closed_days} closed · ${s.promo_days} promo`} />
            <Tile tone="total" icon={CalendarCheck2} label={`Total, ${horizon} days`} value={fmtInt(s.total_predicted)} sub="predicted orders" />
          </>
        )}
      </Reveal>

      <Reveal delay={0.08}>
        <Card title="Forecast chart" subtitle="Recorded demand followed by the model forecast"
          actions={
            <>
              <Legend items={[
                { label: "Recorded", color: colors.history },
                { label: "Forecast", color: colors.forecast, kind: "dash" },
              ]} />
              <Segmented ariaLabel="History shown" value={context} onChange={setContext} options={CONTEXT_OPTIONS} />
              <Switch label="Show closed days" checked={showClosed} onChange={setShowClosed} />
            </>
          }>
          {history.isPending || forecast.isPending ? <ChartSkeleton height={380} />
            : history.isError ? <ErrorState error={history.error} onRetry={() => history.refetch()} />
            : forecast.isError ? <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
            : <ForecastChart history={shown} forecast={rows} showClosed={showClosed} height={380} />}
        </Card>
      </Reveal>

      <Reveal delay={0.12}>
        <Card flush title="Forecast table" subtitle="Select a row to see what pushes that day's forecast up or down"
          actions={<span className="chip chip--accent"><ArrowDownToLine size={12} aria-hidden /> {rows.length} days</span>}>
          {forecast.isPending ? <div style={{ padding: 16 }}><Skeleton h={220} r={12} /></div>
            : forecast.isError ? <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
            : !rows.length ? <EmptyState title="No forecast rows" message="The API returned an empty forecast for this store." />
            : (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Date</th><th className="r">Predicted orders</th><th>Context</th><th className="r">Step</th></tr></thead>
                  <tbody>
                    {rows.map((r) => {
                      const cls = [
                        s?.peak?.date === r.date ? "is-peak" : "", s?.lowest_open_day?.date === r.date ? "is-low" : "",
                        !r.is_open ? "is-closed" : "", selected === r.date ? "is-selected" : "",
                      ].join(" ");
                      return (
                        <tr key={r.date} className={cls} onClick={() => r.is_open && setSelected(r.date)}
                          style={{ cursor: r.is_open ? "pointer" : "default" }} tabIndex={r.is_open ? 0 : -1}
                          onKeyDown={(e) => { if (r.is_open && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); setSelected(r.date); } }}>
                          <td>{fmtDateLong(r.date)}</td>
                          <td className="r num"><b>{r.is_open ? fmtInt(r.predicted_demand) : "0"}</b></td>
                          <td><ContextChips p={r} /></td>
                          <td className="r num" style={{ color: "var(--text-muted)" }}>+{r.horizon_step}d</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
        </Card>
      </Reveal>

      {forecast.data && forecast.data.notes.length > 0 && (
        <Reveal delay={0.14}>
          <div className="callout callout--warn">
            <Info aria-hidden />
            <div>{forecast.data.notes.map((n) => <p key={n}>{n}</p>)}</div>
          </div>
        </Reveal>
      )}

      <Drawer open={!!selectedRow && storeId != null} onClose={close}
        title={selectedRow ? fmtDateLong(selectedRow.date) : ""} subtitle={`Store ${storeId} · step +${selectedRow?.horizon_step ?? ""}d`}>
        {selectedRow && storeId != null && <DriversPanel storeId={storeId} date={selectedRow.date} />}
      </Drawer>
    </div>
  );
}
