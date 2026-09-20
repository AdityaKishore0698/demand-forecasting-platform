import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { BacktestPoint } from "../../api/types";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtCompact, fmtDateLong, fmtInt, fmtTsShort, toTs } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

type Row = BacktestPoint & { ts: number };

export function ActualVsPredictedChart({ points, showBaseline, baselineLabel, height = 320 }: {
  points: BacktestPoint[]; showBaseline: boolean; baselineLabel: string; height?: number;
}) {
  const c = useChartColors();
  const rows: Row[] = useMemo(() => points.map((p) => ({ ...p, ts: toTs(p.date) })), [points]);
  const Tip = (t: TipInput<Row>) => {
    const r = tipPoint(t);
    if (!r) return null;
    const err = r.actual > 0 ? (r.predicted - r.actual) / r.actual : null;
    const list = [
      { label: "Actual", value: fmtInt(r.actual), color: c.actual },
      { label: "Model", value: fmtInt(r.predicted), color: c.forecast },
    ];
    if (showBaseline) list.push({ label: baselineLabel, value: fmtInt(r.baseline), color: c.baseline });
    if (err != null) list.push({ label: "Model error", value: `${err >= 0 ? "+" : "−"}${Math.abs(err * 100).toFixed(1)}%`, color: "" });
    return <TipShell title={fmtDateLong(r.date)} rows={list} />;
  };
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 12, right: 14, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="ts" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmtTsShort} tick={axisTick(c.muted)} axisLine={false} tickLine={false} minTickGap={30} />
          <YAxis tickFormatter={fmtCompact} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={46} domain={[0, "auto"]} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<Row>)} />} cursor={{ stroke: c.muted, strokeDasharray: "3 3" }} />
          {showBaseline && <Line dataKey="baseline" stroke={c.baseline} strokeWidth={1.6} strokeDasharray="5 4" dot={false} isAnimationActive animationDuration={700} />}
          <Line dataKey="actual" stroke={c.actual} strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} isAnimationActive animationDuration={800} />
          <Line dataKey="predicted" stroke={c.forecast} strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} isAnimationActive animationDuration={900} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
