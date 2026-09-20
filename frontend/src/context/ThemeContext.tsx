import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export type Theme = "light" | "dark";
interface ThemeCtx { theme: Theme; toggle: () => void; setTheme: (t: Theme) => void }
const Ctx = createContext<ThemeCtx | null>(null);

const readInitial = (): Theme =>
  document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"; // set pre-paint by index.html

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readInitial);

  const setTheme = useCallback((next: Theme) => {
    // update the DOM first so anything reading CSS variables during the next render sees the new theme
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("df-theme", next); } catch { /* storage unavailable */ }
    setThemeState(next);
  }, []);

  const value = useMemo(
    () => ({ theme, setTheme, toggle: () => setTheme(theme === "dark" ? "light" : "dark") }),
    [theme, setTheme],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTheme must be used inside ThemeProvider");
  return v;
}
