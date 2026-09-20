import { CircleHelp } from "lucide-react";

export function InfoTip({ text, align = "center" }: { text: string; align?: "center" | "left" }) {
  return (
    <span className={`tip ${align === "left" ? "tip--left" : ""}`} tabIndex={0} role="note" aria-label={text}>
      <CircleHelp aria-hidden />
      <span className="tip__bubble" role="tooltip">{text}</span>
    </span>
  );
}
