import type { ReactNode } from "react";

export interface TipRow { label: string; value: string; color?: string }

export function TipShell({ title, rows, footer }: { title: string; rows: TipRow[]; footer?: ReactNode }) {
  return (
    <div className="chart-tip">
      <div className="chart-tip__title">{title}</div>
      {rows.map((r) => (
        <div className="chart-tip__row" key={r.label}>
          <span className="chart-tip__key">
            {r.color && <i className="chart-tip__dot" style={{ background: r.color }} />}
            {r.label}
          </span>
          <b className="num">{r.value}</b>
        </div>
      ))}
      {footer && <div className="chart-tip__foot">{footer}</div>}
    </div>
  );
}

/** Minimal shape of the props Recharts passes to a custom tooltip. */
export interface TipInput<P = Record<string, unknown>> { active?: boolean; payload?: Array<{ payload: P }> }
export const tipPoint = <P,>(t: TipInput<P>): P | null => (t.active && t.payload && t.payload.length ? t.payload[0].payload : null);
