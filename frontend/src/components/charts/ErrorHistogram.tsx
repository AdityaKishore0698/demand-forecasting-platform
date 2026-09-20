import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtInt } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

interface Row { label: string; mid: number; count: number; lo: number; hi: number; near: boolean }

export function ErrorHistogram({ edges, counts, height = 240 }: { edges: number[]; counts: number[]; height?: number }) {
  const c = useChartColors();
  const total = counts.reduce((a, b) => a + b, 0);
  const rows: Row[] = counts.map((count, i) => {
    const lo = edges[i], hi = edges[i + 1], mid = (lo + hi) / 2;
    return { label: `${mid > 0 ? "+" : ""}${(mid * 100).toFixed(1).replace(/\.0$/, "")}%`, mid, count, lo, hi, near: Math.abs(mid) <= 0.1 };
  });
  const Tip = (t: TipInput<Row>) => {
    const r = tipPoint(t);
    return r ? <TipShell title={`Error ${(r.lo * 100).toFixed(0)}% to ${(r.hi * 100).toFixed(0)}%`} rows={[
      { label: "Store-days", value: fmtInt(r.count), color: r.near ? c.bar : c.baseline },
      { label: "Share", value: `${((r.count / total) * 100).toFixed(1)}%` },
    ]} /> : null;
  };
  const zeroLabel = rows.find((r) => Math.abs(r.mid) < 1e-9)?.label ?? rows[Math.floor(rows.length / 2)].label;
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="label" tick={axisTick(c.muted)} axisLine={false} tickLine={false} interval={3} />
          <YAxis tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={44} tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<Row>)} />} cursor={{ fill: c.grid }} />
          <ReferenceLine x={zeroLabel} stroke={c.muted} strokeDasharray="3 3" />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} animationDuration={700}>
            {rows.map((r) => <Cell key={r.label} fill={r.near ? c.bar : c.baseline} fillOpacity={r.near ? 0.95 : 0.6} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
