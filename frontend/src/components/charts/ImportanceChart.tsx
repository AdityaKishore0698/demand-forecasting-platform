import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ImportanceItem } from "../../api/types";
import { axisTick, useChartColors } from "../../lib/chart";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

export function ImportanceChart({ items, valueLabel, height }: { items: ImportanceItem[]; valueLabel: string; height?: number }) {
  const c = useChartColors();
  const h = height ?? Math.max(240, items.length * 30 + 30);
  const Tip = (t: TipInput<ImportanceItem>) => {
    const r = tipPoint(t);
    return r ? (
      <div className="chart-tip" style={{ maxWidth: 280 }}>
        <div className="chart-tip__title">{r.label}</div>
        <div className="chart-tip__row"><span className="chart-tip__key"><i className="chart-tip__dot" style={{ background: c.groups[r.group] }} />{r.group_label}</span><b className="num">{(r.share * 100).toFixed(1)}%</b></div>
        <div className="chart-tip__row"><span>{valueLabel}</span><b className="num">{r.value.toPrecision(3)}</b></div>
        {r.description && <p style={{ marginTop: 6, color: "var(--text-muted)", fontSize: 12, lineHeight: 1.45 }}>{r.description}</p>}
      </div>
    ) : <TipShell title="" rows={[]} />;
  };
  return (
    <div className="chart" style={{ height: h }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={items} layout="vertical" margin={{ top: 4, right: 20, bottom: 0, left: 4 }} barCategoryGap={6}>
          <CartesianGrid stroke={c.grid} horizontal={false} />
          <XAxis type="number" tick={axisTick(c.muted)} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <YAxis type="category" dataKey="label" width={210} tick={{ fill: c.text, fontSize: 12 }} axisLine={false} tickLine={false} interval={0} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<ImportanceItem>)} />} cursor={{ fill: c.grid }} />
          <Bar dataKey="share" radius={[0, 7, 7, 0]} animationDuration={700}>
            {items.map((i) => <Cell key={i.feature} fill={c.groups[i.group] ?? c.bar} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
