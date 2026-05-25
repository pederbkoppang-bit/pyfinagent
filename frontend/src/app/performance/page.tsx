"use client";

import { useEffect, useMemo, useState } from "react";
import { AreaChart } from "@tremor/react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { EmptyState } from "@/components/states/EmptyState";
import { TimeRangeSelector, filterByTimeRange, type TimeRange } from "@/components/TimeRangeSelector";
import { evaluateOutcomes, getPerformanceStats, getCostHistory, listReports, getReport } from "@/lib/api";
import type { PerformanceStats, CostHistoryEntry, ReportSummary, SynthesisReport } from "@/lib/types";
import { IconDeepThink, TabCost } from "@/lib/icons";
// phase-25.B12: replace bare <p> loading/error with canonical PageSkeleton + rose error banner
import { PageSkeleton } from "@/components/Skeleton";

export default function PerformancePage() {
  const [stats, setStats] = useState<PerformanceStats | null>(null);
  const [costHistory, setCostHistory] = useState<CostHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // phase-44.4: time-range filter for cost history (segmented control).
  const [timeRange, setTimeRange] = useState<TimeRange>("30d");
  // phase-44.4: per-pillar averages aggregated from recent reports.
  const [pillarAverages, setPillarAverages] = useState<Record<string, number> | null>(null);

  // Fetch recent reports + aggregate per-pillar averages for the
  // per-pillar performance bars criterion. Fail-soft: leave null if
  // any fetch fails so the bars section silently omits.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const summary: ReportSummary[] = await listReports(20);
        const tickerLatest = new Map<string, string>();
        for (const r of summary) {
          // most recent per ticker (listReports already sorts desc)
          if (!tickerLatest.has(r.ticker)) tickerLatest.set(r.ticker, r.analysis_date);
        }
        const fullReports = await Promise.all(
          Array.from(tickerLatest.entries()).slice(0, 10).map(async ([t, d]) => {
            try {
              const full = (await getReport(t, d)) as Record<string, unknown>;
              const synth =
                ((full.full_report_json as Record<string, unknown>)?.final_synthesis ??
                  full.full_report_json ??
                  {}) as SynthesisReport;
              return synth;
            } catch {
              return null;
            }
          }),
        );
        const synths = fullReports.filter((s): s is SynthesisReport => s !== null);
        if (synths.length === 0) {
          if (!cancelled) setPillarAverages(null);
          return;
        }
        const keys = [
          "pillar_1_corporate",
          "pillar_2_industry",
          "pillar_3_valuation",
          "pillar_4_sentiment",
          "pillar_5_governance",
        ] as const;
        const sums: Record<string, number> = {};
        const counts: Record<string, number> = {};
        for (const s of synths) {
          const sm = s.scoring_matrix as unknown as Record<string, number>;
          if (!sm) continue;
          for (const k of keys) {
            const v = sm[k];
            if (typeof v === "number" && !Number.isNaN(v)) {
              sums[k] = (sums[k] ?? 0) + v;
              counts[k] = (counts[k] ?? 0) + 1;
            }
          }
        }
        const avgs: Record<string, number> = {};
        for (const k of keys) {
          if (counts[k] > 0) avgs[k] = sums[k] / counts[k];
        }
        if (!cancelled) setPillarAverages(Object.keys(avgs).length > 0 ? avgs : null);
      } catch {
        if (!cancelled) setPillarAverages(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Filter + cumulative-transform cost history per the selected range.
  const filteredCostHistory = useMemo(
    () => filterByTimeRange(costHistory, timeRange, "analysis_date"),
    [costHistory, timeRange],
  );

  // Cumulative cost series for the Tremor AreaChart (chronological).
  const cumulativeCostSeries = useMemo(() => {
    const sorted = [...filteredCostHistory].sort((a, b) =>
      a.analysis_date.localeCompare(b.analysis_date),
    );
    let running = 0;
    return sorted.map((r) => {
      running += r.total_cost_usd ?? 0;
      return {
        date: r.analysis_date,
        Cumulative: Number(running.toFixed(4)),
      };
    });
  }, [filteredCostHistory]);

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

            {/* phase-44.4: per-pillar performance bars aggregated from
                SynthesisReport.scoring_matrix across recent reports. Renders
                only when data exists (fail-soft per researcher Option B). */}
            {pillarAverages && (
              <div className="col-span-12">
                <BentoCard>
                  <h3 className="mb-4 text-lg font-semibold text-slate-300">
                    Per-Pillar Average Score
                  </h3>
                  <ul className="space-y-2">
                    {[
                      ["pillar_1_corporate", "Corporate"],
                      ["pillar_2_industry", "Industry"],
                      ["pillar_3_valuation", "Valuation"],
                      ["pillar_4_sentiment", "Sentiment"],
                      ["pillar_5_governance", "Governance"],
                    ].map(([k, label]) => {
                      const v = pillarAverages[k] ?? 0;
                      const widthPct = Math.min(100, Math.max(0, (v / 10) * 100));
                      const bar = v >= 7 ? "bg-emerald-500/80" : v >= 5 ? "bg-sky-500/80" : v >= 3 ? "bg-amber-500/80" : "bg-rose-500/80";
                      return (
                        <li key={k} className="flex items-center gap-3">
                          <span className="w-28 text-xs text-slate-400">{label}</span>
                          <div
                            className="relative h-5 flex-1 overflow-hidden rounded bg-zinc-800"
                            role="progressbar"
                            aria-valuenow={Number(v.toFixed(2))}
                            aria-valuemin={0}
                            aria-valuemax={10}
                            aria-label={`${label} average ${v.toFixed(2)} of 10`}
                          >
                            <div className={`h-full ${bar}`} style={{ width: `${widthPct}%` }} />
                          </div>
                          <span className="w-12 text-right font-mono text-xs text-slate-300">
                            {v.toFixed(2)}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                  <p className="mt-2 text-xs text-slate-500">
                    Averaged across the latest report per ticker (max 10 tickers).
                  </p>
                </BentoCard>
              </div>
            )}

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
          <div className="mt-8">
            <EmptyState
              icon={TabCost}
              title="No cost history yet"
              description="Costs appear here after the first analysis runs."
            />
          </div>
        )}
        {costHistory.length > 0 && (
          <div className="mt-8">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-300">
                <TabCost size={20} weight="duotone" /> LLM Cost History
              </h3>
              {/* phase-44.4: TimeRangeSelector segmented control */}
              <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
            </div>

            {/* phase-44.4: Tremor AreaChart -- cumulative cost above the table.
                colors={["amber"]} overrides Tremor's hardcoded-blue default
                (verified vs vendor source cycle 63). */}
            {cumulativeCostSeries.length > 0 && (
              <BentoCard>
                <h4 className="mb-3 text-sm font-semibold text-slate-400">
                  Cumulative Cost ({timeRange})
                </h4>
                <AreaChart
                  data={cumulativeCostSeries}
                  index="date"
                  categories={["Cumulative"]}
                  colors={["amber"]}
                  yAxisWidth={48}
                  valueFormatter={(n: number) => `$${n.toFixed(4)}`}
                  className="h-48"
                  showAnimation={false}
                  showLegend={false}
                />
              </BentoCard>
            )}

            <div className="mt-6 grid grid-cols-12 gap-6 mb-6">
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
                    {filteredCostHistory.map((row, i) => (
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
