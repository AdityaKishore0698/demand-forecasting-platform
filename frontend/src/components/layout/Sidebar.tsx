import { motion } from "framer-motion";
import { Gauge, Info, LayoutDashboard, Sparkles, TrendingUp } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useHealth } from "../../api/hooks";
import { fmtDateYear } from "../../lib/format";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/forecast", label: "Forecast", icon: TrendingUp },
  { to: "/performance", label: "Model Performance", icon: Gauge },
  { to: "/insights", label: "Feature Insights", icon: Sparkles },
  { to: "/about", label: "About", icon: Info },
] as const;

export function LogoMark() {
  return (
    <svg className="brand__mark" viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#3550d6" /><stop offset="1" stopColor="#e8590c" /></linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#logo-g)" />
      <rect x="7" y="17" width="4" height="8" rx="1.3" fill="#fff" fillOpacity=".8" />
      <rect x="14" y="12" width="4" height="13" rx="1.3" fill="#fff" fillOpacity=".8" />
      <rect x="21" y="7" width="4" height="18" rx="1.3" fill="#fff" />
    </svg>
  );
}

export function Sidebar({ open, onNavigate }: { open: boolean; onNavigate: () => void }) {
  const health = useHealth();
  const ok = health.data?.model_loaded === true;
  return (
    <aside className={`sidebar ${open ? "is-open" : ""}`} aria-label="Primary">
      <div className="brand">
        <LogoMark />
        <div className="brand__text">
          <span className="brand__title">Demand Forecasting</span>
          <span className="brand__sub">Analytics Platform</span>
        </div>
      </div>
      <nav className="nav" aria-label="Sections">
        <div className="nav__label">Workspace</div>
        {NAV.map(({ to, label, icon: Icon, ...rest }) => (
          <NavLink key={to} to={to} end={"end" in rest} onClick={onNavigate} className={({ isActive }) => `nav__item ${isActive ? "is-active" : ""}`}>
            {({ isActive }) => (
              <>
                {isActive && <motion.span layoutId="nav-pill" className="nav__pill" transition={{ type: "spring", stiffness: 520, damping: 42 }} />}
                <Icon aria-hidden />
                <span>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar__foot">
        <div className="snapshot">
          <div className="snapshot__row"><span><span className={`status-dot ${ok ? "" : "is-bad"}`} />API</span><b>{health.isPending ? "checking…" : ok ? "online" : "offline"}</b></div>
          {health.data?.data_through && <div className="snapshot__row"><span>Data through</span><b>{fmtDateYear(health.data.data_through)}</b></div>}
          {health.data?.model_version && <div className="snapshot__row"><span>Model</span><b className="num">{health.data.model_version.split("-")[1] ?? health.data.model_version}</b></div>}
        </div>
      </div>
    </aside>
  );
}
