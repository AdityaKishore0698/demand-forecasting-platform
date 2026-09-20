import { TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { useImportance } from "../api/hooks";
import type { ImportanceItem, ImportanceResponse, ImportanceView } from "../api/types";
import { ImportanceChart } from "../components/charts/ImportanceChart";
import { Card } from "../components/ui/Card";
import { Reveal } from "../components/ui/Reveal";
import { Segmented } from "../components/ui/Segmented";
import { ChartSkeleton, QueryBoundary, Skeleton } from "../components/ui/States";
import { useChartColors } from "../lib/chart";
import { fmtPct } from "../lib/format";

const VIEWS: Array<{ value: ImportanceView; label: string; unit: string; blurb: string }> = [
  { value: "permutation", label: "Permutation", unit: "RMSLE increase",
    blurb: "Shuffle one feature on the hold-out window and measure how much the forecast error grows. Bigger = the model leans on it more." },
  { value: "shap", label: "SHAP", unit: "mean |SHAP| (log scale)",
    blurb: "Average size of each feature's push on the prediction, using exact TreeSHAP contributions." },
  { value: "gain", label: "Gain", unit: "loss reduction",
    blurb: "How much each feature reduced training loss across all trees in the final model. Can overstate high-cardinality features." },
];

const TOP = 15;

function groupTotals(items: ImportanceItem[]) {
  const acc = new Map<string, { group: string; label: string; share: number; n: number }>();
  for (const i of items) {
    const g = acc.get(i.group) ?? { group: i.group, label: i.group_label, share: 0, n: 0 };
    g.share += i.share;
    g.n += 1;
    acc.set(i.group, g);
  }
  return [...acc.values()].sort((a, b) => b.share - a.share);
}

function Content({ data }: { data: ImportanceResponse }) {
  const colors = useChartColors();
  const [view, setView] = useState<ImportanceView>("permutation");
  const v = VIEWS.find((x) => x.value === view)!;
  const items = data[view];
  const top = useMemo(() => items.slice(0, TOP), [items]);
  const groups = useMemo(() => groupTotals(items), [items]);

  return (
    <>
      <Reveal delay={0.04}>
        <Card title="Most influential features" subtitle={v.blurb}
          actions={<Segmented ariaLabel="Importance method" value={view} onChange={setView} options={VIEWS.map((x) => ({ value: x.value, label: x.label }))} />}>
          <ImportanceChart items={top} valueLabel={v.unit} />
          <p className="card__sub" style={{ marginTop: 8 }}>
            Top {TOP} of {items.length} features. Bars show each feature's share of the total; colour shows its group.{" "}
            {String(data.meta[view] ?? "")}
          </p>
        </Card>
      </Reveal>

      <Reveal delay={0.08} className="grid grid--split-rev">
        <Card title="By feature group" subtitle="Share of total importance, summed within each group">
          <div className="stack-bar" role="img" aria-label="Importance share by feature group">
            {groups.map((g) => <div key={g.group} style={{ width: `${g.share * 100}%`, background: colors.groups[g.group] }} title={`${g.label} ${fmtPct(g.share)}`} />)}
          </div>
          <div className="legend-groups" style={{ marginTop: 14 }}>
            {groups.map((g) => (
              <div className="group-row" key={g.group}><i style={{ background: colors.groups[g.group] }} />{g.label}<b>{fmtPct(g.share, g.share < 0.01 ? 1 : 0)}</b></div>
            ))}
          </div>
        </Card>

        <Card title="What the top features mean" subtitle="In plain English">
          {top.slice(0, 7).map((f, i) => (
            <div className="feature-card" key={f.feature}>
              <span className="feature-card__rank">{i + 1}</span>
              <div>
                <b>{f.label}</b> <span className="chip chip--accent" style={{ marginLeft: 6 }}>{f.group_label}</span>
                <p>{f.description}</p>
              </div>
            </div>
          ))}
        </Card>
      </Reveal>

      <Reveal delay={0.12}>
        <div className="callout callout--warn">
          <TriangleAlert aria-hidden />
          <div>
            <b>Read importance as "what the model uses", not "what causes demand".</b> Features are correlated (for example, the store's usual demand
            on a weekday overlaps with recent lags), so shuffling one can under-state its true information. Permutation and SHAP are computed on a
            model trained before the hold-out window; gain comes from the final model trained on all history.
          </div>
        </div>
      </Reveal>
    </>
  );
}

export default function FeatureInsights() {
  const q = useImportance();
  return (
    <div className="page">
      <Reveal>
        <div className="page__head">
          <div>
            <h1>Feature insights</h1>
            <p className="page__lede">What the model pays attention to when it forecasts demand, measured three different ways.</p>
          </div>
        </div>
      </Reveal>
      <QueryBoundary query={q} skeleton={<><Skeleton h={420} r={16} /><ChartSkeleton height={200} /></>}>
        {(data) => <Content data={data} />}
      </QueryBoundary>
    </div>
  );
}
