import { motion } from "framer-motion";
import { useId } from "react";

interface Option<T> { value: T; label: string }

export function Segmented<T extends string | number>({
  value, onChange, options, ariaLabel,
}: { value: T; onChange: (v: T) => void; options: Option<T>[]; ariaLabel: string }) {
  const id = useId();
  return (
    <div className="seg" role="tablist" aria-label={ariaLabel}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button key={String(o.value)} role="tab" aria-selected={active} type="button"
            className={`seg__btn ${active ? "is-active" : ""}`} onClick={() => onChange(o.value)}>
            {active && <motion.span layoutId={`seg-${id}`} className="seg__thumb" transition={{ type: "spring", stiffness: 500, damping: 38 }} />}
            <span>{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}
