import { Menu } from "lucide-react";
import { useLocation } from "react-router-dom";
import { HorizonSelect } from "../controls/HorizonSelect";
import { StoreSelect } from "../controls/StoreSelect";
import { ThemeToggle } from "../controls/ThemeToggle";
import { NAV } from "./Sidebar";

export function Topbar({ onMenu }: { onMenu: () => void }) {
  const { pathname } = useLocation();
  const current = NAV.find((n) => ("end" in n ? pathname === n.to : pathname.startsWith(n.to))) ?? NAV[0];
  const showControls = pathname !== "/about";
  return (
    <header className="topbar">
      <button className="btn btn--icon menu-btn" onClick={onMenu} aria-label="Open navigation"><Menu aria-hidden /></button>
      <div className="topbar__title">
        <span className="topbar__eyebrow">Demand Forecasting Platform</span>
        <span className="topbar__name">{current.label}</span>
      </div>
      <div className="topbar__controls">
        {showControls && (<><StoreSelect /><HorizonSelect /></>)}
        <ThemeToggle />
      </div>
    </header>
  );
}
