import { motion } from "framer-motion";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "../../context/ThemeContext";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === "dark";
  return (
    <button className="theme-toggle" onClick={toggle} role="switch" aria-checked={dark}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} title={dark ? "Light theme" : "Dark theme"}>
      <span className="theme-toggle__icons" aria-hidden><Sun /><Moon /></span>
      <motion.span className="theme-toggle__thumb" initial={false} animate={{ x: dark ? 28 : 0 }}
        transition={{ type: "spring", stiffness: 520, damping: 34 }}>
        {dark ? <Moon aria-hidden /> : <Sun aria-hidden />}
      </motion.span>
    </button>
  );
}
