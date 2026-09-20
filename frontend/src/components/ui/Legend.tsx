export interface LegendItem { label: string; color: string; kind?: "line" | "dash" | "box" }

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span className="legend__item" key={i.label} style={{ color: i.color }}>
          <span className={`legend__swatch ${i.kind === "dash" ? "legend__swatch--dash" : i.kind === "box" ? "legend__swatch--box" : ""}`}
            style={{ background: i.color }} />
          <span style={{ color: "var(--text-2)" }}>{i.label}</span>
        </span>
      ))}
    </div>
  );
}
