const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const oneFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

export const fmtInt = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? "—" : intFmt.format(Math.round(v)));
export const fmtNum = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(v);
export const fmtOne = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? "—" : oneFmt.format(v));

export function fmtCompact(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1_000_000) return `${(v / 1_000_000).toFixed(a >= 10_000_000 ? 0 : 1)}M`;
  if (a >= 1_000) return `${(v / 1_000).toFixed(a >= 10_000 ? 0 : 1)}k`;
  return intFmt.format(Math.round(v));
}

/** value is a fraction (0.068 -> "6.8%") */
export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
export const fmtSignedPct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;

const parse = (iso: string) => new Date(`${iso}T00:00:00`);
export const toTs = (iso: string) => parse(iso).getTime();
export const fmtDate = (iso: string) => parse(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
export const fmtDateLong = (iso: string) => parse(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
export const fmtDateShort = (iso: string) => parse(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
export const fmtDateYear = (iso: string) => parse(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
export const fmtTsShort = (ts: number) => new Date(ts).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
export const fmtTsMonth = (ts: number) => new Date(ts).toLocaleDateString("en-GB", { month: "short", year: "2-digit" });

export const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const;
/** JS getDay(): 0=Sun -> our Mon=0 index */
export const weekdayIndex = (iso: string) => (parse(iso).getDay() + 6) % 7;
