import { ArrowDownRight, ArrowUpRight, Minus, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useCountUp } from "../../lib/motion";
import { InfoTip } from "./InfoTip";
import { Skeleton } from "./States";

interface Delta { value: number | null; goodWhen?: "up" | "down"; label?: string }

interface KpiProps {
  label: string;
  icon: LucideIcon;
  tone?: "accent" | "forecast" | "positive";
  value?: number | null;                        // animated number
  format?: (v: number) => string;
  text?: string;                                // OR a static string value
  unit?: string;
  sub?: ReactNode;
  delta?: Delta;
  info?: string;
  loading?: boolean;
}

function DeltaBadge({ delta }: { delta: Delta }) {
  if (delta.value == null || Number.isNaN(delta.value)) return null;
  const v = delta.value;
  const flat = Math.abs(v) < 0.0005;
  const up = v > 0;
  const neutral = !delta.goodWhen;
  const good = delta.goodWhen === "up" ? up : !up;
  const cls = flat || neutral ? "delta--flat" : good ? "delta--up" : "delta--down";
  const Icon = flat ? Minus : up ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={`delta ${cls}`}>
      <Icon aria-hidden />
      {(Math.abs(v) * 100).toFixed(1)}%
    </span>
  );
}

function Animated({ value, format }: { value: number; format: (v: number) => string }) {
  const v = useCountUp(value);
  return <>{format(v)}</>;
}

export function KpiCard({ label, icon: Icon, tone = "accent", value, format = String, text, unit, sub, delta, info, loading }: KpiProps) {
  return (
    <div className="card kpi card--hover">
      <div className="kpi__top">
        <span className="kpi__label">{label}{info && <InfoTip text={info} />}</span>
        <span className={`kpi__icon ${tone === "forecast" ? "kpi__icon--forecast" : tone === "positive" ? "kpi__icon--positive" : ""}`}>
          <Icon aria-hidden />
        </span>
      </div>
      {loading ? (
        <>
          <Skeleton h={30} w="62%" />
          <Skeleton h={14} w="80%" />
        </>
      ) : (
        <>
          <div className="kpi__value num">
            {text != null ? text : value == null ? "—" : <Animated value={value} format={format} />}
            {unit && <small>{unit}</small>}
          </div>
          <div className="kpi__sub">
            {delta && <DeltaBadge delta={delta} />}
            {sub}
          </div>
        </>
      )}
    </div>
  );
}
