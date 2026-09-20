import { describe, expect, it } from "vitest";
import { fmtCompact, fmtInt, fmtPct, fmtSignedPct, weekdayIndex } from "./format";

describe("format helpers", () => {
  it("formats integers and compacts large numbers", () => {
    expect(fmtInt(6234.6)).toBe("6,235");
    expect(fmtCompact(1250)).toBe("1.3k");
    expect(fmtCompact(48200)).toBe("48k");
    expect(fmtCompact(2_300_000)).toBe("2.3M");
  });
  it("handles missing values without printing NaN", () => {
    expect(fmtInt(null)).toBe("—");
    expect(fmtPct(undefined)).toBe("—");
    expect(fmtCompact(Number.NaN)).toBe("—");
  });
  it("formats fractions as percentages", () => {
    expect(fmtPct(0.0684)).toBe("6.8%");
    expect(fmtSignedPct(0.052)).toBe("+5.2%");
    expect(fmtSignedPct(-0.031)).toBe("−3.1%");
  });
  it("maps dates to a Monday-first weekday index", () => {
    expect(weekdayIndex("2015-06-22")).toBe(0); // Monday
    expect(weekdayIndex("2015-06-21")).toBe(6); // Sunday
  });
});
