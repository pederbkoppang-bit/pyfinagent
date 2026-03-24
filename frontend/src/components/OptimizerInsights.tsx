"use client";

import { useEffect, useState } from "react";
import type { OptimizerInsights, OptimizerExperimentFull } from "@/lib/types";
import { BentoCard } from "@/components/BentoCard";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  CartesianGrid,
} from "recharts";
import { ArrowClockwise } from "@phosphor-icons/react";

// ── Section 1: Training Data Scope ──────────────────────────────

function DataScopeSection({ scope }: { scope: OptimizerInsights["data_scope"] }) {
  if (!scope || !scope.windows?.length) {
    return (
      <BentoCard>
        <h3 className="mb-3 text-lg font-semibold text-slate-300">Training Data Scope</h3>
        <p className="text-sm text-slate-500">Run a backtest first to see data scope.</p>
      </BentoCard>
    );
  }

  // Compute date range for timeline
  const allDates = scope.windows.flatMap((w) => [w.train_start, w.train_end, w.test_start, w.test_end].filter(Boolean));
  const minDate = allDates.length ? new Date(allDates.sort()[0]).getTime() : 0;
  const maxDate = allDates.length ? new Date(allDates.sort().reverse()[0]).getTime() : 0;
  const range = maxDate - minDate || 1;

  return (
    <BentoCard>
      <h3 className="mb-3 text-lg font-semibold text-slate-300">Training Data Scope</h3>
      <div className="mb-4 flex flex-wrap gap-4 text-xs text-slate-400">
        <span>Date range: <span className="font-mono text-slate-300">{scope.start_date} → {scope.end_date}</span></span>
        <span>Windows: <span className="font-mono text-slate-300">{scope.n_windows}</span></span>
        <span>Features: <span className="font-mono text-slate-300">{scope.n_features}</span></span>
        <span>Strategy: <span className="font-mono text-slate-300">{scope.strategy}</span></span>
      </div>

      {/* Walk-forward Gantt chart */}
      <div className="space-y-2">
        {scope.windows.map((w) => {
          const trainStart = ((new Date(w.train_start).getTime() - minDate) / range) * 100;
          const trainWidth = ((new Date(w.train_end).getTime() - new Date(w.train_start).getTime()) / range) * 100;
          const testStart = ((new Date(w.test_start).getTime() - minDate) / range) * 100;
          const testWidth = ((new Date(w.test_end).getTime() - new Date(w.test_start).getTime()) / range) * 100;

          return (
            <div key={w.id} className="flex items-center gap-2">
              <span className="w-8 text-right font-mono text-xs text-slate-500">W{w.id}</span>
              <div className="relative h-5 flex-1 rounded bg-slate-800">
                <div
                  className="absolute top-0 h-full rounded-l bg-sky-600/60"
                  style={{ left: `${trainStart}%`, width: `${Math.max(trainWidth, 0.5)}%` }}
                  title={`Train: ${w.train_start} → ${w.train_end} (${w.n_train_samples} samples)`}
                />
                <div
                  className="absolute top-0 h-full rounded-r bg-emerald-500/60"
                  style={{ left: `${testStart}%`, width: `${Math.max(testWidth, 0.5)}%` }}
                  title={`Test: ${w.test_start} → ${w.test_end} (${w.n_candidates} candidates)`}
                />
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 flex items-center gap-4 text-[11px] text-slate-500">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded bg-sky-600/60" /> Train</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded bg-emerald-500/60" /> Test</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded bg-slate-800" /> Embargo gap</span>
      </div>
    </BentoCard>
  );
}


// ── Section 2: Parameter Slice Plots ────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  keep: "#22c55e",
  BASELINE: "#38bdf8",
  dsr_reject: "#f59e0b",
  discard: "#64748b",
  crash: "#ef4444",
};

function SlicePlotsSection({
  experiments,
  paramBounds,
  intParams,
}: {
  experiments: OptimizerExperimentFull[];
  paramBounds: Record<string, [number, number]>;
  intParams: string[];
}) {
  const paramNames = Object.keys(paramBounds);

  // Build per-param data: only experiments that have params_full
  const withParams = experiments.filter((e) => e.params_full);
  if (withParams.length < 2) {
    return (
      <BentoCard>
        <h3 className="mb-3 text-lg font-semibold text-slate-300">Parameter Slice Plots</h3>
        <p className="text-sm text-slate-500">Need ≥2 experiments with full param snapshots. Run optimizer with updated backend to populate.</p>
      </BentoCard>
    );
  }

  return (
    <BentoCard>
      <h3 className="mb-4 text-lg font-semibold text-slate-300">Parameter Slice Plots</h3>
      <p className="mb-4 text-xs text-slate-500">
        Each plot shows one param vs Sharpe. Green = kept · Amber = DSR reject · Gray = discard
      </p>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {paramNames.map((param) => {
          const data = withParams.map((e) => ({
            x: Number(e.params_full?.[param] ?? 0),
            y: e.metric_after,
            status: e.status,
            fill: STATUS_COLORS[e.status] ?? "#64748b",
          }));
          const [lo, hi] = paramBounds[param];
          const isInt = intParams.includes(param);

          return (
            <div key={param} className="rounded-lg border border-slate-700/50 bg-navy-800/40 p-2">
              <p className="mb-1 truncate text-[10px] font-medium text-slate-400" title={param}>
                {param}
              </p>
              <ResponsiveContainer width="100%" height={110}>
                <ScatterChart margin={{ top: 4, right: 4, bottom: 12, left: 0 }}>
                  <XAxis
                    type="number"
                    dataKey="x"
                    domain={[lo, hi]}
                    tick={{ fontSize: 8, fill: "#64748b" }}
                    tickFormatter={(v: number) => isInt ? String(Math.round(v)) : v.toFixed(2)}
                    tickCount={3}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    tick={{ fontSize: 8, fill: "#64748b" }}
                    tickFormatter={(v: number) => v.toFixed(2)}
                    tickCount={3}
                    width={32}
                  />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
                    formatter={(val: number, name: string) => [
                      name === "x" ? (isInt ? Math.round(val) : val.toFixed(4)) : val.toFixed(4),
                      name === "x" ? param : "Sharpe",
                    ]}
                  />
                  <Scatter data={data} fill="#64748b">
                    {data.map((d, i) => (
                      <Cell key={i} fill={d.fill} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </BentoCard>
  );
}


// ── Section 3: Parameter Importance ─────────────────────────────

function ParamImportanceSection({ experiments, paramBounds }: {
  experiments: OptimizerExperimentFull[];
  paramBounds: Record<string, [number, number]>;
}) {
  const withParams = experiments.filter((e) => e.params_full);
  if (withParams.length < 3) {
    return (
      <BentoCard>
        <h3 className="mb-3 text-lg font-semibold text-slate-300">Parameter Importance</h3>
        <p className="text-sm text-slate-500">Need ≥3 experiments to compute importance.</p>
      </BentoCard>
    );
  }

  // For each param that was changed, compute variance of Sharpe
  const paramVariance: Record<string, { values: number[]; sharpes: number[] }> = {};
  for (const param of Object.keys(paramBounds)) {
    paramVariance[param] = { values: [], sharpes: [] };
  }
  for (const e of withParams) {
    if (!e.params_full) continue;
    for (const param of Object.keys(paramBounds)) {
      const val = Number(e.params_full[param] ?? 0);
      paramVariance[param].values.push(val);
      paramVariance[param].sharpes.push(e.metric_after);
    }
  }

  // Compute importance as std of Sharpe for experiments where this param was modified
  const importances = Object.entries(paramVariance)
    .map(([param, { sharpes }]) => {
      if (sharpes.length < 2) return { param, importance: 0 };
      const mean = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
      const variance = sharpes.reduce((a, b) => a + (b - mean) ** 2, 0) / sharpes.length;
      return { param, importance: Math.sqrt(variance) };
    })
    .sort((a, b) => b.importance - a.importance);

  const maxImp = importances[0]?.importance || 1;

  return (
    <BentoCard>
      <h3 className="mb-4 text-lg font-semibold text-slate-300">Parameter Importance</h3>
      <p className="mb-3 text-xs text-slate-500">
        Higher variance in Sharpe across experiments = more sensitivity to this parameter
      </p>
      <ResponsiveContainer width="100%" height={importances.length * 28 + 20}>
        <BarChart data={importances} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} />
          <YAxis type="category" dataKey="param" tick={{ fontSize: 10, fill: "#94a3b8" }} width={140} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
            formatter={(val: number) => [val.toFixed(4), "Importance"]}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
            {importances.map((d, i) => (
              <Cell key={i} fill={d.importance > maxImp * 0.6 ? "#10b981" : d.importance > maxImp * 0.3 ? "#6366f1" : "#475569"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </BentoCard>
  );
}


// ── Section 4: Feature Stability Matrix ─────────────────────────

function FeatureStabilitySection({ experiments }: { experiments: OptimizerExperimentFull[] }) {
  // Only kept + baseline experiments with MDA data
  const relevant = experiments.filter(
    (e) => (e.status === "keep" || e.status === "BASELINE") && e.top5_mda.length > 0
  );

  if (relevant.length < 2) {
    return (
      <BentoCard>
        <h3 className="mb-3 text-lg font-semibold text-slate-300">Feature Stability Matrix</h3>
        <p className="text-sm text-slate-500">Need ≥2 kept experiments with MDA data.</p>
      </BentoCard>
    );
  }

  // Collect all features that ever appeared in top5
  const featureSet = new Set<string>();
  relevant.forEach((e) => e.top5_mda.forEach((f) => { if (f) featureSet.add(f); }));
  const features = Array.from(featureSet);

  // Build rank matrix: for each experiment, rank features by their position in top5_mda
  // If not present → rank = features.length (worst)
  const matrix = relevant.map((e) => {
    const ranks: Record<string, number> = {};
    features.forEach((f) => {
      const idx = e.top5_mda.indexOf(f);
      ranks[f] = idx >= 0 ? idx + 1 : features.length + 1;
    });
    return { experiment: e, ranks };
  });

  const maxRank = features.length + 1;
  const shadeForRank = (rank: number): string => {
    if (rank === 1) return "bg-emerald-500";
    if (rank === 2) return "bg-emerald-600";
    if (rank === 3) return "bg-emerald-700";
    if (rank <= 5) return "bg-emerald-800";
    return "bg-slate-700";
  };

  return (
    <BentoCard>
      <h3 className="mb-4 text-lg font-semibold text-slate-300">Feature Stability Matrix</h3>
      <p className="mb-3 text-xs text-slate-500">
        Rows = features, columns = kept experiments. Even color = stable model. Wild variation = overfitting risk.
      </p>
      <div className="overflow-x-auto">
        <table className="text-[10px]">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-slate-500">Feature</th>
              {matrix.map((m, i) => (
                <th key={i} className="px-1 py-1 text-center text-slate-600" title={m.experiment.param_changed}>
                  {m.experiment.status === "BASELINE" ? "BL" : `#${i}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.slice(0, 10).map((feature) => (
              <tr key={feature}>
                <td className="truncate px-2 py-0.5 font-mono text-slate-400" style={{ maxWidth: 120 }} title={feature}>
                  {feature}
                </td>
                {matrix.map((m, i) => {
                  const rank = m.ranks[feature];
                  return (
                    <td key={i} className="px-1 py-0.5 text-center">
                      <span
                        className={`inline-block h-5 w-5 rounded text-[9px] font-bold leading-5 ${
                          rank <= maxRank ? shadeForRank(rank) : "bg-slate-800"
                        } text-white`}
                        title={`Rank ${rank} in experiment ${i}`}
                      >
                        {rank <= features.length ? rank : "—"}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BentoCard>
  );
}


// ── Section 5: Decision Log ─────────────────────────────────────

function DecisionLogSection({ experiments }: { experiments: OptimizerExperimentFull[] }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? experiments : experiments.slice(-20);

  if (experiments.length === 0) {
    return (
      <BentoCard>
        <h3 className="mb-3 text-lg font-semibold text-slate-300">Decision Log</h3>
        <p className="text-sm text-slate-500">No experiments yet.</p>
      </BentoCard>
    );
  }

  const borderColor = (status: string) => {
    if (status === "keep") return "border-l-emerald-500";
    if (status === "BASELINE") return "border-l-sky-500";
    if (status === "dsr_reject") return "border-l-amber-500";
    return "border-l-slate-600";
  };

  return (
    <BentoCard>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-300">Decision Log</h3>
        {experiments.length > 20 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-sky-400 hover:text-sky-300"
          >
            {expanded ? "Show recent" : `Show all ${experiments.length}`}
          </button>
        )}
      </div>
      <div className="max-h-96 space-y-1.5 overflow-y-auto scrollbar-thin">
        {visible.map((e, i) => (
          <div
            key={i}
            className={`flex items-start gap-3 rounded-lg border-l-2 bg-navy-800/40 py-2 pl-3 pr-2 ${borderColor(e.status)}`}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className={`inline-block rounded-full px-1.5 py-0.5 text-[9px] font-bold ${
                  e.status === "keep" ? "bg-emerald-500/20 text-emerald-400"
                  : e.status === "BASELINE" ? "bg-sky-500/20 text-sky-400"
                  : e.status === "dsr_reject" ? "bg-amber-500/20 text-amber-400"
                  : "bg-slate-700 text-slate-400"
                }`}>
                  {e.status.toUpperCase()}
                </span>
                <span className="truncate text-xs text-slate-300" title={e.param_changed}>{e.param_changed}</span>
              </div>
              <div className="mt-0.5 flex items-center gap-3 text-[10px] text-slate-500">
                <span>Sharpe: {e.metric_before.toFixed(3)} → <span className={e.delta > 0 ? "text-emerald-400" : "text-slate-400"}>{e.metric_after.toFixed(3)}</span></span>
                <span>DSR: {e.dsr.toFixed(3)}</span>
                {e.delta !== 0 && (
                  <span className={e.delta > 0 ? "text-emerald-400" : "text-rose-400"}>
                    Δ{e.delta > 0 ? "+" : ""}{e.delta.toFixed(4)}
                  </span>
                )}
              </div>
            </div>
            <span className="shrink-0 text-[9px] text-slate-600">{e.timestamp.slice(11, 19)}</span>
          </div>
        ))}
      </div>
    </BentoCard>
  );
}


// ── Main Component ──────────────────────────────────────────────

export function OptimizerInsightsView({
  insights,
  onRefresh,
}: {
  insights: OptimizerInsights | null;
  onRefresh: () => Promise<void>;
}) {
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!insights) {
      setLoading(true);
      onRefresh().finally(() => setLoading(false));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading && !insights) {
    return (
      <div className="flex items-center gap-2 py-12 text-slate-500">
        <ArrowClockwise size={16} className="animate-spin" />
        Loading optimizer insights...
      </div>
    );
  }

  if (!insights) {
    return <p className="text-slate-500">No optimizer data available. Run the optimizer first.</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {insights.experiments.length} experiments · {Object.keys(insights.param_bounds).length} tunable params
        </p>
        <button
          onClick={async () => { setLoading(true); await onRefresh(); setLoading(false); }}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:border-slate-600"
        >
          <ArrowClockwise size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      <DataScopeSection scope={insights.data_scope} />

      <SlicePlotsSection
        experiments={insights.experiments}
        paramBounds={insights.param_bounds}
        intParams={insights.int_params}
      />

      <ParamImportanceSection
        experiments={insights.experiments}
        paramBounds={insights.param_bounds}
      />

      <FeatureStabilitySection experiments={insights.experiments} />

      <DecisionLogSection experiments={insights.experiments} />
    </div>
  );
}
