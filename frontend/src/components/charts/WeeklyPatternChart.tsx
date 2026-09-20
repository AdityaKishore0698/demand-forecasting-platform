import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtCompact, fmtInt, fmtPct } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

export interface WeekRow { day: string; avg: number; openDays: number; closedShare: number }

export function WeeklyPatternChart({ data, height = 230 }: { data: WeekRow[]; height?: number }) {
  const c = useChartColors();
  const max = Math.max(...data.map((d) => d.avg));
  const Tip = (t: TipInput<WeekRow>) => {
    const r = tipPoint(t);
    return r ? <TipShell title={r.day} rows={[
      { label: "Avg orders (open days)", value: fmtInt(r.avg), color: c.bar },
      { label: "Closed on this weekday", value: fmtPct(r.closedShare, 0) },
      { label: "Open days observed", value: fmtInt(r.openDays) },
    ]} /> : null;
  };
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="day" tick={axisTick(c.muted)} axisLine={false} tickLine={false} />
          <YAxis tickFormatter={fmtCompact} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={40} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<WeekRow>)} />} cursor={{ fill: c.grid }} />
          <Bar dataKey="avg" radius={[7, 7, 2, 2]} animationDuration={700}>
            {data.map((d) => <Cell key={d.day} fill={c.bar} fillOpacity={d.avg === max ? 1 : d.openDays === 0 ? 0.15 : 0.55} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
