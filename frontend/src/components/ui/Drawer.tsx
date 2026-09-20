import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";

export function Drawer({ open, onClose, title, subtitle, children }: {
  open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="drawer-scrim" onClick={onClose}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} />
          <motion.aside className="drawer" role="dialog" aria-modal="true" aria-label="Forecast explanation"
            initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 340, damping: 38 }}>
            <div className="drawer__head">
              <div><h2>{title}</h2>{subtitle && <p className="card__sub">{subtitle}</p>}</div>
              <button className="btn btn--icon btn--ghost" onClick={onClose} aria-label="Close"><X aria-hidden /></button>
            </div>
            <div className="drawer__body">{children}</div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
