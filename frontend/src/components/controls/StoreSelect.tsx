import { ChevronDown, Search, Store as StoreIcon } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../context/AppContext";
import { fmtInt } from "../../lib/format";

const MAX_ROWS = 60;

export function StoreSelect() {
  const { stores, store, setStoreId, storesLoading } = useApp();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const t = q.trim();
    return t ? stores.filter((s) => String(s.store_id).includes(t)) : stores;
  }, [stores, q]);
  const shown = filtered.slice(0, MAX_ROWS);

  useEffect(() => { setCursor(0); }, [q, open]);
  useEffect(() => {
    if (!open) return;
    input.current?.focus();
    const onDown = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const choose = (id: number) => { setStoreId(id); setOpen(false); setQ(""); };
  const jump = (kind: "top" | "median" | "smallest") => {
    if (!stores.length) return;
    const sorted = [...stores].sort((a, b) => a.demand_rank - b.demand_rank);
    choose(kind === "top" ? sorted[0].store_id : kind === "smallest" ? sorted[sorted.length - 1].store_id : sorted[Math.floor(sorted.length / 2)].store_id);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, shown.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === "Enter" && shown[cursor]) { e.preventDefault(); choose(shown[cursor].store_id); }
    else if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="select" ref={root}>
      <button className="select__btn" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open} disabled={storesLoading && !store}>
        <StoreIcon aria-hidden />
        <span className="select__value">
          {store ? <>Store {store.store_id} <span className="select__hint">· Fmt {store.hub_format ?? "?"} · Tier {store.assortment_tier ?? "?"}</span></> : "Loading hubs…"}
        </span>
        <ChevronDown aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div className="select__panel" initial={{ opacity: 0, y: -6, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }} transition={{ duration: 0.16 }}>
            <div className="select__search">
              <Search aria-hidden />
              <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey} inputMode="numeric"
                placeholder={`Search ${stores.length.toLocaleString()} hubs by number…`} aria-label="Search hubs" />
            </div>
            <div style={{ display: "flex", gap: 6, padding: "8px 10px 2px", flexWrap: "wrap" }}>
              <button className="chip chip--accent" style={{ border: 0, cursor: "pointer" }} onClick={() => jump("top")}>Highest demand</button>
              <button className="chip chip--accent" style={{ border: 0, cursor: "pointer" }} onClick={() => jump("median")}>Median hub</button>
              <button className="chip chip--accent" style={{ border: 0, cursor: "pointer" }} onClick={() => jump("smallest")}>Lowest demand</button>
            </div>
            <div className="select__list" role="listbox">
              {shown.length === 0 && <div className="select__empty">No hub matches “{q}”.</div>}
              {shown.map((s, i) => (
                <button key={s.store_id} role="option" aria-selected={s.store_id === store?.store_id}
                  className={`select__opt ${s.store_id === store?.store_id ? "is-selected" : ""} ${i === cursor ? "is-cursor" : ""}`}
                  onMouseEnter={() => setCursor(i)} onClick={() => choose(s.store_id)}>
                  <span><b>Store {s.store_id}</b> <small>Fmt {s.hub_format ?? "?"} · Tier {s.assortment_tier ?? "?"}</small></span>
                  <small className="num">{fmtInt(s.avg_open_day_demand)} / day</small>
                </button>
              ))}
              {filtered.length > MAX_ROWS && <div className="select__empty">Showing {MAX_ROWS} of {filtered.length.toLocaleString()} — keep typing to narrow.</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
