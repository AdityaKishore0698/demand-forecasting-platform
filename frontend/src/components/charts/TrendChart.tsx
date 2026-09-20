import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtCompact, fmtDateLong, fmtInt, fmtTsMonth, toTs } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

interface Row { ts: number; date: string; value: number | null }

export function TrendChart({ data, forecastMean, height = 230 }: { data: Array<{ date: string; value: number | null }>; forecastMean: number | null; height?: number }) {
  const c = useChartColors();
  const rows: Row[] = useMemo(() => data.map((d) => ({ ...d, ts: toTs(d.date) })), [data]);
  const Tip = (t: TipInput<Row>) => {
    const r = tipPoint(t);
    return r && r.value != null ? <TipShell title={fmtDateLong(r.date)} rows={[{ label: "28-day average", value: fmtInt(r.value), color: c.history }]} /> : null;
  };
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <defs><linearGradient id="gTrend" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c.history} stopOpacity={0.25} /><stop offset="100%" stopColor={c.history} stopOpacity={0.02} /></linearGradient></defs>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="ts" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmtTsMonth} tick={axisTick(c.muted)} axisLine={false} tickLine={false} minTickGap={48} />
          <YAxis tickFormatter={fmtCompact} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={40} domain={["auto", "auto"]} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<Row>)} />} cursor={{ stroke: c.muted, strokeDasharray: "3 3" }} />
          {forecastMean != null && <ReferenceLine y={forecastMean} stroke={c.forecast} strokeDasharray="5 4" label={{ value: `Forecast avg ${fmtInt(forecastMean)}`, position: "insideTopRight", fill: c.forecast, fontSize: 11 }} />}
          <Area dataKey="value" type="monotone" stroke={c.history} strokeWidth={2} fill="url(#gTrend)" connectNulls={false} dot={false} isAnimationActive animationDuration={800} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
