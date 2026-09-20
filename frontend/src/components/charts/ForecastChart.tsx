import { useMemo } from "react";
import { Area, CartesianGrid, ComposedChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ForecastPoint, HistoricalPoint } from "../../api/types";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtCompact, fmtDateLong, fmtInt, fmtTsShort, toTs } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

interface Row {
  ts: number; date: string; hist: number | null; fc: number | null; kind: "history" | "forecast";
  open: boolean | null; promo: boolean | null; holiday?: boolean; school?: boolean; step?: number;
}

interface Props { history: HistoricalPoint[]; forecast: ForecastPoint[]; showClosed: boolean; height?: number }

export function ForecastChart({ history, forecast, showClosed, height = 340 }: Props) {
  const c = useChartColors();

  const { rows, fStart, fEnd, bridgeTs } = useMemo(() => {
    const rows: Row[] = [];
    for (const p of history) {
      const ts = toTs(p.date);
      if (p.demand == null) { rows.push({ ts, date: p.date, hist: null, fc: null, kind: "history", open: null, promo: null }); continue; }
      if (!p.is_open && !showClosed) continue;
      rows.push({ ts, date: p.date, hist: p.demand, fc: null, kind: "history", open: p.is_open, promo: p.promo_active });
    }
    let bridgeTs: number | null = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i].hist != null) { rows[i].fc = rows[i].hist; bridgeTs = rows[i].ts; break; }   // join the two lines
    }
    for (const f of forecast) {
      if (!f.is_open && !showClosed) continue;
      rows.push({ ts: toTs(f.date), date: f.date, hist: null, fc: f.predicted_demand, kind: "forecast", open: f.is_open, promo: f.promo_active, holiday: f.holiday, school: f.school_closure, step: f.horizon_step });
    }
    const fStart = forecast.length ? toTs(forecast[0].date) : null;
    const fEnd = forecast.length ? toTs(forecast[forecast.length - 1].date) : null;
    return { rows, fStart, fEnd, bridgeTs };
  }, [history, forecast, showClosed]);

  const promoDot = (colorKey: "history" | "forecast", valueKey: "hist" | "fc") =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (p: any) => {
      const row: Row | undefined = p.payload;
      if (!row?.promo || row[valueKey] == null || (valueKey === "fc" && row.kind === "history")) return <g key={`d-${p.index}`} />;
      return <circle key={`d-${p.index}`} cx={p.cx} cy={p.cy} r={3.4} fill={c[colorKey]} stroke={c.surface} strokeWidth={1.5} />;
    };

  const Tip = (t: TipInput<Row>) => {
    const r = tipPoint(t);
    if (!r) return null;
    const isFc = r.kind === "forecast";
    const v = isFc ? r.fc : r.hist;
    return (
      <TipShell title={fmtDateLong(r.date)}
        rows={[{ label: isFc ? "Forecast" : "Actual", value: r.open === false ? "0 (closed)" : fmtInt(v), color: isFc ? c.forecast : c.history }]
          .concat(isFc && r.step ? [{ label: "Days ahead", value: String(r.step), color: "" }] : [])}
        footer={<>
          {r.open === false && <span className="chip chip--closed">Closed</span>}
          {r.promo && <span className="chip chip--promo">Promotion</span>}
          {r.holiday && <span className="chip chip--holiday">Holiday</span>}
          {r.school && <span className="chip chip--school">School closure</span>}
        </>} />
    );
  };

  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 14, right: 14, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="gHist" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.history} stopOpacity={0.28} /><stop offset="100%" stopColor={c.history} stopOpacity={0.02} /></linearGradient>
            <linearGradient id="gFc" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.forecast} stopOpacity={0.3} /><stop offset="100%" stopColor={c.forecast} stopOpacity={0.02} /></linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} vertical={false} />
          {fStart != null && fEnd != null && <ReferenceArea x1={fStart} x2={fEnd} fill={c.forecast} fillOpacity={0.06} ifOverflow="visible" />}
          <XAxis dataKey="ts" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmtTsShort} tick={axisTick(c.muted)}
            axisLine={false} tickLine={false} minTickGap={36} />
          <YAxis tickFormatter={fmtCompact} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={44} domain={[0, "auto"]} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<Row>)} />} cursor={{ stroke: c.muted, strokeDasharray: "3 3" }} />
          {bridgeTs != null && <ReferenceLine x={bridgeTs} stroke={c.muted} strokeDasharray="4 4" label={{ value: "Forecast starts →", position: "insideTopRight", fill: c.muted, fontSize: 11 }} />}
          <Area dataKey="hist" type="monotone" stroke={c.history} strokeWidth={2} fill="url(#gHist)" connectNulls={false}
            dot={promoDot("history", "hist")} activeDot={{ r: 4.5 }} isAnimationActive animationDuration={700} />
          <Area dataKey="fc" type="monotone" stroke={c.forecast} strokeWidth={2.4} strokeDasharray="6 4" fill="url(#gFc)" connectNulls={false}
            dot={promoDot("forecast", "fc")} activeDot={{ r: 5 }} isAnimationActive animationDuration={900} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
