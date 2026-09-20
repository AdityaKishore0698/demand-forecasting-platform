import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HistoricalPoint } from "../../api/types";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtCompact, fmtDate, fmtDateLong, fmtInt } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

export function RecentDemandChart({ data, height = 230 }: { data: HistoricalPoint[]; height?: number }) {
  const c = useChartColors();
  const rows = data.map((p) => ({ ...p, v: p.demand ?? 0, label: fmtDate(p.date).split(" ").slice(0, 2).join(" ") }));
  const Tip = (t: TipInput<(typeof rows)[number]>) => {
    const r = tipPoint(t);
    return r ? (
      <TipShell title={fmtDateLong(r.date)} rows={[{ label: "Orders", value: r.demand == null ? "no data" : r.is_open ? fmtInt(r.demand) : "0 (closed)", color: c.bar }]}
        footer={<>{r.is_open === false && <span className="chip chip--closed">Closed</span>}{r.promo_active && <span className="chip chip--promo">Promotion</span>}</>} />
    ) : null;
  };
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="label" tick={axisTick(c.muted)} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={18} />
          <YAxis tickFormatter={fmtCompact} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={40} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<(typeof rows)[number]>)} />} cursor={{ fill: c.grid }} />
          <Bar dataKey="v" radius={[6, 6, 2, 2]} animationDuration={700}>
            {rows.map((r) => <Cell key={r.date} fill={r.promo_active ? c.forecast : c.bar} fillOpacity={r.is_open === false ? 0.25 : 0.9} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
