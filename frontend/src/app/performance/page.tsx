"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { evaluateOutcomes, getPerformanceStats, getCostHistory } from "@/lib/api";
import type { PerformanceStats, CostHistoryEntry } from "@/lib/types";
import { IconDeepThink, TabCost } from "@/lib/icons";
// phase-25.B12: replace bare <p> loading/error with canonical PageSkeleton + rose error banner
import { PageSkeleton } from "@/components/Skeleton";

export default function PerformancePage() {
  const [stats, setStats] = useState<PerformanceStats | null>(null);
  const [costHistory, setCostHistory] = useState<CostHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getPerformanceStats().then(setStats),
      getCostHistory().then(setCostHistory),
    ])
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleEvaluate = async () => {
    setEvaluating(true);
    try {
      const result = await evaluateOutcomes();
      // Refresh stats after evaluation
      const updated = await getPerformanceStats();
      setStats(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* phase-16.48: canonical two-zone shell -- header pinned, content scrolls */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">
                Recommendation Performance
              </h2>
              <p className="text-sm text-slate-500">
                Track whether past recommendations were correct (learning loop)
              </p>
            </div>
            <button
              onClick={handleEvaluate}
              disabled={evaluating}
              className="rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
            >
              {evaluating ? "Evaluating..." : "Evaluate Outcomes"}
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
        {/* phase-25.B12: canonical loading + error states per frontend.md rules */}
        {loading && <PageSkeleton />}
        {error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
            <p className="text-sm text-rose-300">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-2 rounded-md border border-rose-500/40 px-3 py-1 text-xs text-rose-300 transition-colors hover:bg-rose-950/50"
            >
              Retry
            </button>
          </div>
        )}

        {stats && (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 md:col-span-4">
              <BentoCard glow>
                <p className="text-sm text-slate-400">Win Rate</p>
                <p className="mt-2 font-mono text-5xl font-bold text-emerald-400">
                  {((stats.win_rate ?? 0) * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {stats.wins}W / {stats.losses}L of{" "}
                  {stats.total_recommendations} total
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12 md:col-span-4">
              <BentoCard>
                <p className="text-sm text-slate-400">Avg Return</p>
                <p
                  className={`mt-2 font-mono text-5xl font-bold ${
                    (stats.avg_return ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {(stats.avg_return ?? 0) >= 0 ? "+" : ""}
                  {(stats.avg_return ?? 0).toFixed(2)}%
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12 md:col-span-4">
              <BentoCard>
                <p className="text-sm text-slate-400">Beat Benchmark (SPY)</p>
                <p className="mt-2 font-mono text-5xl font-bold text-sky-400">
                  {((stats.benchmark_beat_rate ?? 0) * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  of recommendations outperformed S&P 500
                </p>
              </BentoCard>
            </div>

            <div className="col-span-12">
              <BentoCard>
                <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold text-slate-300">
                  <IconDeepThink size={20} weight="duotone" /> How the Learning Loop Works
                </h3>
                <p className="text-sm leading-relaxed text-slate-400">
                  Click <strong>Evaluate Outcomes</strong> to compare each past
                  recommendation against actual price performance. The system
                  checks: did the stock go up after a Buy? Did it go down after a
                  Sell? Did it beat the S&P 500? Over time, this data reveals
                  which agents and pillars are most predictive — allowing pillar
                  weights to be calibrated empirically.
                </p>
              </BentoCard>
            </div>
          </div>
        )}

        {/* ── Cost History ──────────────────────────────────────── */}
        {/* phase-16.48: loading + empty states added so the section
            doesn't silently disappear before data arrives. */}
        {loading && costHistory.length === 0 && (
          <div className="mt-8 flex items-center gap-3 py-8 text-slate-400">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
            Loading cost history...
          </div>
        )}
        {!loading && costHistory.length === 0 && !error && (
          <div className="mt-8 flex flex-col items-center justify-center py-16 text-center">
            <TabCost size={36} weight="duotone" className="text-slate-600" />
            <p className="mt-3 text-sm text-slate-400">No cost history yet</p>
            <p className="mt-1 text-xs text-slate-600">
              Costs appear here after the first analysis runs
            </p>
          </div>
        )}
        {costHistory.length > 0 && (
          <div className="mt-8">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-300">
              <TabCost size={20} weight="duotone" /> LLM Cost History
            </h3>
            <div className="grid grid-cols-12 gap-6 mb-6">
              <div className="col-span-12 md:col-span-4">
                <BentoCard>
                  <p className="text-sm text-slate-400">Total Spend</p>
                  <p className="mt-2 font-mono text-4xl font-bold text-amber-400">
                    $
                    {costHistory
                      .reduce((s, r) => s + (r.total_cost_usd ?? 0), 0)
                      .toFixed(4)}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    across {costHistory.length} analyses
                  </p>
                </BentoCard>
              </div>
              <div className="col-span-12 md:col-span-4">
                <BentoCard>
                  <p className="text-sm text-slate-400">Avg Cost / Analysis</p>
                  <p className="mt-2 font-mono text-4xl font-bold text-sky-400">
                    $
                    {(
                      costHistory.reduce(
                        (s, r) => s + (r.total_cost_usd ?? 0),
                        0
                      ) / costHistory.length
                    ).toFixed(4)}
                  </p>
                </BentoCard>
              </div>
              <div className="col-span-12 md:col-span-4">
                <BentoCard>
                  <p className="text-sm text-slate-400">Total Tokens</p>
                  <p className="mt-2 font-mono text-4xl font-bold text-violet-400">
                    {(
                      costHistory.reduce(
                        (s, r) => s + (r.total_tokens ?? 0),
                        0
                      ) / 1_000_000
                    ).toFixed(2)}
                    M
                  </p>
                </BentoCard>
              </div>
            </div>

            <BentoCard>
              <h4 className="mb-3 text-sm font-semibold text-slate-400">
                Per-Analysis Cost
              </h4>
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-500">
                      <th className="pb-2 pr-4">Ticker</th>
                      <th className="pb-2 pr-4">Date</th>
                      <th className="pb-2 pr-4">Model</th>
                      <th className="pb-2 pr-4 text-right">Tokens</th>
                      <th className="pb-2 pr-4 text-right">Cost</th>
                      <th className="pb-2 text-right">Deep Think</th>
                    </tr>
                  </thead>
                  <tbody>
                    {costHistory.map((row, i) => (
                      <tr
                        key={`${row.ticker}-${row.analysis_date}-${i}`}
                        className="border-b border-slate-800 text-slate-300"
                      >
                        <td className="py-2 pr-4 font-medium text-slate-200">
                          {row.ticker}
                        </td>
                        <td className="py-2 pr-4 text-slate-400">
                          {row.analysis_date}
                        </td>
                        <td className="py-2 pr-4 text-xs text-slate-400">
                          {row.standard_model || "—"}
                          {row.deep_think_model && row.deep_think_model !== row.standard_model && (
                            <span className="ml-1 text-violet-400">+ {row.deep_think_model}</span>
                          )}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono">
                          {row.total_tokens != null
                            ? (row.total_tokens / 1000).toFixed(1) + "K"
                            : "—"}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono text-amber-400">
                          {row.total_cost_usd != null
                            ? "$" + row.total_cost_usd.toFixed(4)
                            : "—"}
                        </td>
                        <td className="py-2 text-right font-mono">
                          {row.deep_think_calls ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </BentoCard>
          </div>
        )}
        </div>
      </main>
    </div>
  );
}
