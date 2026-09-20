import type { HistoricalPoint } from "../api/types";
import { WEEKDAYS, weekdayIndex } from "./format";

/** Average demand per weekday on days the hub was open. */
export function weeklyPattern(points: HistoricalPoint[]) {
  const sum = new Array(7).fill(0);
  const n = new Array(7).fill(0);
  const total = new Array(7).fill(0);
  const closed = new Array(7).fill(0);
  for (const p of points) {
    if (p.demand == null) continue;
    const i = weekdayIndex(p.date);
    total[i] += 1;
    if (p.is_open) { sum[i] += p.demand; n[i] += 1; } else closed[i] += 1;
  }
  return WEEKDAYS.map((day, i) => ({
    day, avg: n[i] ? sum[i] / n[i] : 0, openDays: n[i], closedShare: total[i] ? closed[i] / total[i] : 0,
  }));
}

/** Trailing mean of open-day demand (requires a minimum number of open days in the window). */
export function rollingOpenMean(points: HistoricalPoint[], window = 28, minOpen = 8) {
  const out: Array<{ date: string; value: number | null }> = [];
  for (let i = 0; i < points.length; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      const p = points[j];
      if (p.demand != null && p.is_open) { s += p.demand; c += 1; }
    }
    out.push({ date: points[i].date, value: c >= minOpen ? s / c : null });
  }
  return out;
}

export function meanOpen(points: HistoricalPoint[]): number | null {
  let s = 0, c = 0;
  for (const p of points) if (p.demand != null && p.is_open) { s += p.demand; c += 1; }
  return c ? s / c : null;
}
