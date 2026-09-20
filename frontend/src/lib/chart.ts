import { useMemo } from "react";
import { useTheme } from "../context/ThemeContext";

export interface ChartColors {
  history: string; forecast: string; actual: string; baseline: string; bar: string;
  grid: string; text: string; muted: string; surface: string; positive: string; negative: string;
  groups: Record<string, string>;
}

/** Resolve the active theme's CSS variables into concrete colours for SVG charts. */
export function useChartColors(): ChartColors {
  const { theme } = useTheme();
  return useMemo(() => {
    const cs = getComputedStyle(document.documentElement);
    const v = (n: string) => cs.getPropertyValue(n).trim();
    return {
      history: v("--c-history"), forecast: v("--c-forecast"), actual: v("--c-actual"), baseline: v("--c-baseline"),
      bar: v("--c-bar"), grid: v("--grid"), text: v("--text"), muted: v("--text-muted"), surface: v("--surface"),
      positive: v("--positive"), negative: v("--negative"),
      groups: {
        schedule: v("--g-schedule"), promo: v("--g-promo"), calendar: v("--g-calendar"), recent: v("--g-recent"),
        baseline: v("--g-baseline"), hub: v("--g-hub"), horizon: v("--g-horizon"),
      },
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}

export const axisTick = (fill: string) => ({ fill, fontSize: 11.5 });
