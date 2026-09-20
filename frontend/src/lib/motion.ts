import { animate, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export const EASE = [0.22, 1, 0.36, 1] as const;

/** Smoothly count a number up/down when it changes (respects reduced-motion). */
export function useCountUp(target: number | null | undefined, duration = 0.85): number {
  const reduce = useReducedMotion();
  const latest = useRef(target ?? 0);
  const [value, setValue] = useState(target ?? 0);

  useEffect(() => {
    if (target == null || Number.isNaN(target)) return;
    if (reduce) { latest.current = target; setValue(target); return; }
    const controls = animate(latest.current, target, {
      duration, ease: EASE, onUpdate: (v) => { latest.current = v; setValue(v); },
    });
    return () => controls.stop();
  }, [target, duration, reduce]);

  return target == null ? 0 : value;
}
