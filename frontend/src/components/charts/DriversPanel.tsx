import { useExplain } from "../../api/hooks";
import { fmtDateLong, fmtInt, fmtNum } from "../../lib/format";
import { ChartSkeleton, ErrorState } from "../ui/States";
import { Skeleton } from "../ui/States";

const GROUPS: Record<string, string> = {
  schedule: "Operating schedule", promo: "Promotions & holidays", calendar: "Calendar", recent: "Recent demand",
  baseline: "Store baseline", hub: "Store attributes", horizon: "Forecast horizon",
};
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%`;
const fmtValue = (v: number | null) => (v == null ? "n/a" : Number.isInteger(v) ? String(v) : fmtNum(v, 1));

export function DriversPanel({ storeId, date }: { storeId: number; date: string }) {
  const q = useExplain(storeId, date);
  if (q.isPending) return <div style={{ display: "grid", gap: 14 }}><Skeleton h={64} r={13} /><ChartSkeleton height={220} /></div>;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const d = q.data;
  const max = Math.max(...d.drivers.map((x) => Math.abs(x.effect_pct)), Math.abs(d.other_features_effect_pct), 1);
  const bar = (v: number) => ({ width: `${(Math.abs(v) / max) * 50}%` });

  return (
    <>
      <div className="pair">
        <div><small>Model starting point</small><div className="hero-num num">{fmtInt(d.baseline_demand)}</div></div>
        <div><small>Forecast for {fmtDateLong(d.date).split(",")[0]}</small><div className="hero-num num" style={{ color: "var(--forecast)" }}>{fmtInt(d.predicted_demand)}</div></div>
      </div>
      {!d.is_open && <div className="callout callout--warn">The store is closed that day, so the forecast is forced to 0. The model would otherwise predict {fmtInt(d.model_prediction)}.</div>}
      <div>
        <h3 style={{ marginBottom: 4 }}>What moves this forecast</h3>
        <p className="card__sub" style={{ marginBottom: 8 }}>Each bar is the multiplicative effect of a feature on expected demand, relative to the model's starting point — its typical output across all training days, closed days included, which is why it is far below a normal open-day forecast.</p>
        <div className="drivers">
          {d.drivers.map((x) => (
            <div className="driver" key={x.feature}>
              <div className="driver__name"><b title={x.label}>{x.label}</b><small>{GROUPS[x.group] ?? x.group} · value {fmtValue(x.value)}</small></div>
              <div className="driver__track"><div className={`driver__fill ${x.effect_pct >= 0 ? "driver__fill--pos" : "driver__fill--neg"}`} style={bar(x.effect_pct)} /></div>
              <div className="driver__pct num" style={{ color: x.effect_pct >= 0 ? "var(--forecast)" : "var(--accent)" }}>{signed(x.effect_pct)}</div>
            </div>
          ))}
          <div className="driver" style={{ opacity: 0.8 }}>
            <div className="driver__name"><b>All other features</b><small>combined</small></div>
            <div className="driver__track"><div className={`driver__fill ${d.other_features_effect_pct >= 0 ? "driver__fill--pos" : "driver__fill--neg"}`} style={bar(d.other_features_effect_pct)} /></div>
            <div className="driver__pct num">{signed(d.other_features_effect_pct)}</div>
          </div>
        </div>
      </div>
      <div className="callout"><span className="muted">{d.note}</span></div>
    </>
  );
}
