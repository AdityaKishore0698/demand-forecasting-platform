import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GroupError } from "../../api/types";
import { axisTick, useChartColors } from "../../lib/chart";
import { fmtInt, fmtPct } from "../../lib/format";
import { TipShell, tipPoint, type TipInput } from "./ChartTooltip";

export function GroupBarChart({ data, overall, height = 240 }: { data: GroupError[]; overall: number | null; height?: number }) {
  const c = useChartColors();
  const rows = data.map((d) => ({ ...d, v: d.wape ?? 0 }));
  const Tip = (t: TipInput<GroupError & { v: number }>) => {
    const r = tipPoint(t);
    return r ? <TipShell title={r.group} rows={[
      { label: "WAPE", value: fmtPct(r.wape), color: c.bar },
      { label: "MAE (orders)", value: fmtInt(r.mae) },
      { label: "RMSLE", value: r.rmsle == null ? "—" : r.rmsle.toFixed(4) },
      { label: "Open store-days", value: fmtInt(r.n_open_rows) },
    ]} /> : null;
  };
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 22, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="group" tick={axisTick(c.muted)} axisLine={false} tickLine={false} />
          <YAxis tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={axisTick(c.muted)} axisLine={false} tickLine={false} width={40} />
          <Tooltip content={(t: unknown) => <Tip {...(t as TipInput<GroupError & { v: number }>)} />} cursor={{ fill: c.grid }} />
          <Bar dataKey="v" radius={[7, 7, 2, 2]} animationDuration={700}>
            {rows.map((r) => <Cell key={r.group} fill={overall != null && r.v > overall ? c.forecast : c.bar} fillOpacity={0.9} />)}
            <LabelList dataKey="v" position="top" formatter={(v: unknown) => fmtPct(Number(v), 1)} fill={c.text} fontSize={11.5} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
