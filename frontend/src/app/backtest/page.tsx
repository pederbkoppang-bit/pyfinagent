"use client";

import { useCallback, useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { PageSkeleton } from "@/components/Skeleton";
import {
  runBacktest,
  getBacktestStatus,
  getBacktestResults,
  getIngestionStatus,
  runDataIngestion,
  startOptimizer,
  stopOptimizer,
  getOptimizerStatus,
  getOptimizerExperiments,
  getOptimizerBest,
} from "@/lib/api";
import type {
  BacktestStatus,
  BacktestResults,
  IngestionStatus,
  OptimizerStatus,
  OptimizerExperiment,
  OptimizerBest,
} from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import {
  Play,
  Stop,
  ArrowClockwise,
  Database,
  ChartLineUp,
  Table,
  TrendUp,
  Lightning,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

/* ── Helpers ── */
function Metric({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-xl font-bold ${color ?? "text-slate-100"}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

type Tab = "results" | "equity" | "features" | "optimizer";
const TABS: { id: Tab; label: string; icon: Icon }[] = [
  { id: "results", label: "Results", icon: Table },
  { id: "equity", label: "Equity Curve", icon: ChartLineUp },
  { id: "features", label: "Features", icon: TrendUp },
  { id: "optimizer", label: "Optimizer", icon: Lightning },
];

export default function BacktestPage() {
  const [btStatus, setBtStatus] = useState<BacktestStatus | null>(null);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [ingestion, setIngestion] = useState<IngestionStatus | null>(null);
  const [optStatus, setOptStatus] = useState<OptimizerStatus | null>(null);
  const [optExperiments, setOptExperiments] = useState<OptimizerExperiment[]>([]);
  const [optBest, setOptBest] = useState<OptimizerBest | null>(null);
  const [tab, setTab] = useState<Tab>("results");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ingestResult, setIngestResult] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, ing, opt] = await Promise.all([
        getBacktestStatus().catch(() => null),
        getIngestionStatus().catch(() => null),
        getOptimizerStatus().catch(() => null),
      ]);
      if (s) setBtStatus(s);
      if (ing) setIngestion(ing);
      if (opt) setOptStatus(opt);

      // Parallel fetch of conditional data
      const [r, exp, best] = await Promise.all([
        s?.has_result ? getBacktestResults().catch(() => null) : Promise.resolve(null),
        getOptimizerExperiments().catch(() => null),
        getOptimizerBest().catch(() => null),
      ]);
      if (r) setResults(r);
      if (exp) setOptExperiments(exp.experiments);
      if (best) setOptBest(best);

      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load backtest data");
    } finally {
      setLoading(false);
    }
  }, []);

  // Lightweight status-only poll (no heavy data re-fetches)
  const refreshStatus = useCallback(async () => {
    try {
      const [s, opt] = await Promise.all([
        getBacktestStatus().catch(() => null),
        getOptimizerStatus().catch(() => null),
      ]);
      if (s) setBtStatus(s);
      if (opt) setOptStatus(opt);
    } catch { /* swallow poll errors */ }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while running — lightweight status only, full refresh on completion
  const prevRunning = useState({ bt: false, opt: false })[0];
  useEffect(() => {
    const btRunning = btStatus?.status === "running";
    const optRunning = optStatus?.status === "running";
    if (!btRunning && !optRunning) {
      // If something just finished, do a full refresh to load results
      if (prevRunning.bt || prevRunning.opt) {
        refresh();
      }
      prevRunning.bt = false;
      prevRunning.opt = false;
      return;
    }
    prevRunning.bt = btRunning;
    prevRunning.opt = optRunning;
    const id = setInterval(refreshStatus, 5000);
    return () => clearInterval(id);
  }, [btStatus?.status, optStatus?.status, refresh, refreshStatus, prevRunning]);

  const handleRunBacktest = async () => {
    setActionLoading("backtest");
    try {
      await runBacktest();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start backtest");
    } finally {
      setActionLoading(null);
    }
  };

  const handleIngest = async () => {
    setActionLoading("ingest");
    setIngestResult(null);
    try {
      const res = await runDataIngestion();
      const r = res?.result ?? {};
      const prices = r.prices_inserted ?? 0;
      const fundamentals = r.fundamentals_inserted ?? 0;
      const macro = r.macro_inserted ?? 0;
      setIngestResult({
        type: "success",
        message: `Ingestion complete — ${Number(prices).toLocaleString()} prices, ${Number(fundamentals).toLocaleString()} fundamentals, ${Number(macro).toLocaleString()} macro rows inserted`,
      });
      await refresh();
    } catch (e) {
      setIngestResult({
        type: "error",
        message: e instanceof Error ? e.message : "Ingestion failed",
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleStartOptimizer = async () => {
    setActionLoading("optimizer");
    try {
      await startOptimizer({ max_iterations: 100 });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start optimizer");
    } finally {
      setActionLoading(null);
    }
  };

  const handleStopOptimizer = async () => {
    try {
      await stopOptimizer();
      await refresh();
    } catch {
      /* ignore */
    }
  };

  const a = results?.analytics;
  const isRunning = btStatus?.status === "running";
  const isOptRunning = optStatus?.status === "running";

  // Top MDA features for the features tab
  const topFeatures = results?.per_window
    ?.flatMap((w) => {
      const mda = w.feature_importance_mda ?? {};
      return Object.entries(mda).map(([name, imp]) => ({ name, importance: imp, window: w.window_id }));
    })
    .reduce<Record<string, { total: number; count: number }>>((acc, f) => {
      acc[f.name] = acc[f.name] ?? { total: 0, count: 0 };
      acc[f.name].total += f.importance;
      acc[f.name].count += 1;
      return acc;
    }, {});

  const avgFeatures = topFeatures
    ? Object.entries(topFeatures)
        .map(([name, { total, count }]) => ({ name, importance: total / count }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 20)
    : [];

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Walk-Forward Backtest</h2>
            <p className="text-sm text-slate-500">
              ML-driven backtesting with Triple Barrier labels &amp; Deflated Sharpe Ratio guard
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleIngest}
              disabled={!!actionLoading}
              title="Downloads S&P 500 prices (yfinance), fundamentals, and FRED macro data into BigQuery. Free data sources, BQ cost <$0.05. Takes ~5-15 min."
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 transition-colors hover:border-sky-500/50 hover:text-sky-300 disabled:opacity-50"
            >
              <Database size={16} />
              {actionLoading === "ingest" ? "Ingesting..." : "Ingest Data"}
            </button>
            <button
              onClick={handleRunBacktest}
              disabled={!!actionLoading || isRunning}
              title="Runs walk-forward ML backtest on ingested data. Uses GradientBoosting (no LLM cost). BQ reads <$0.01. Takes ~2-5 min."
              className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:opacity-50"
            >
              <Play size={16} weight="fill" />
              {actionLoading === "backtest" || isRunning ? "Running..." : "Run Backtest"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
            <p className="text-sm text-rose-300">{error}</p>
          </div>
        )}

        {/* Ingest result banner */}
        {ingestResult && (
          <div
            className={`mb-4 rounded-lg border p-3 ${
              ingestResult.type === "success"
                ? "border-emerald-500/30 bg-emerald-950/30"
                : "border-rose-500/30 bg-rose-950/30"
            }`}
          >
            <div className="flex items-center justify-between">
              <p className={`text-sm ${ingestResult.type === "success" ? "text-emerald-300" : "text-rose-300"}`}>
                {ingestResult.message}
              </p>
              <button onClick={() => setIngestResult(null)} className="ml-3 text-xs text-slate-500 hover:text-slate-300">
                dismiss
              </button>
            </div>
          </div>
        )}

        {/* Status banner */}
        {isRunning && (
          <div className="mb-4 rounded-lg border border-sky-500/30 bg-sky-950/30 p-3">
            <div className="flex items-center gap-2">
              <ArrowClockwise size={16} className="animate-spin text-sky-400" />
              <span className="text-sm text-sky-300">{btStatus?.progress || "Running backtest..."}</span>
            </div>
          </div>
        )}

        {/* Data ingestion summary */}
        {ingestion && (
          <div className="mb-6">
            <div className="grid grid-cols-3 gap-3">
              <Metric label="Price Rows" value={ingestion.historical_prices.toLocaleString()} />
              <Metric label="Fundamental Rows" value={ingestion.historical_fundamentals.toLocaleString()} />
              <Metric label="Macro Rows" value={ingestion.historical_macro.toLocaleString()} />
            </div>
            <p className="mt-2 text-xs text-slate-600">
              Data: yfinance + FRED (free) &middot; BQ storage &lt;$0.05 &middot; Backtest: ML only, $0 LLM cost
            </p>
          </div>
        )}

        {/* Analytics summary */}
        {a && (
          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            <Metric
              label="Sharpe Ratio"
              value={a.sharpe.toFixed(2)}
              color={a.sharpe >= 1 ? "text-emerald-400" : a.sharpe >= 0 ? "text-sky-300" : "text-rose-400"}
            />
            <Metric
              label="Deflated Sharpe"
              value={a.deflated_sharpe.toFixed(2)}
              color={a.deflated_sharpe >= 0.95 ? "text-emerald-400" : "text-amber-400"}
              sub={a.deflated_sharpe >= 0.95 ? "PASS" : "FAIL"}
            />
            <Metric
              label="Total Return"
              value={`${a.total_return_pct >= 0 ? "+" : ""}${a.total_return_pct.toFixed(1)}%`}
              color={a.total_return_pct >= 0 ? "text-emerald-400" : "text-rose-400"}
            />
            <Metric label="Hit Rate" value={`${(a.hit_rate * 100).toFixed(1)}%`} />
            <Metric
              label="Max Drawdown"
              value={`${a.max_drawdown.toFixed(1)}%`}
              color="text-rose-400"
            />
            <Metric
              label="Alpha"
              value={`${a.alpha >= 0 ? "+" : ""}${a.alpha.toFixed(1)}%`}
              color={a.alpha >= 0 ? "text-emerald-400" : "text-rose-400"}
            />
          </div>
        )}

        {loading && <PageSkeleton />}

        {/* Tab bar */}
        {!loading && (
          <>
            <div className="mb-6 flex gap-1 rounded-lg bg-navy-800/60 p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                    tab === t.id
                      ? "bg-sky-500/10 text-sky-400"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <t.icon size={16} weight={tab === t.id ? "fill" : "regular"} />
                  {t.label}
                </button>
              ))}
            </div>

            {/* ═══ RESULTS TAB ═══ */}
            {tab === "results" && (
              <div className="space-y-6">
                {/* Baselines comparison */}
                {results?.baselines && (
                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">Strategy vs Baselines</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-700 text-left">
                            <th className="px-3 py-2 text-slate-400">Strategy</th>
                            <th className="px-3 py-2 text-right text-slate-400">Return</th>
                            <th className="px-3 py-2 text-right text-slate-400">Sharpe</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr className="border-b border-slate-800 bg-sky-500/5">
                            <td className="px-3 py-2 font-medium text-sky-300">PyFinAgent ML</td>
                            <td className="px-3 py-2 text-right font-mono text-slate-200">
                              {a ? `${a.total_return_pct >= 0 ? "+" : ""}${a.total_return_pct.toFixed(1)}%` : "—"}
                            </td>
                            <td className="px-3 py-2 text-right font-mono text-slate-200">
                              {a?.sharpe.toFixed(2) ?? "—"}
                            </td>
                          </tr>
                          {Object.entries(results.baselines).map(([key, val]) => (
                            <tr key={key} className="border-b border-slate-800">
                              <td className="px-3 py-2 text-slate-400 capitalize">{key.replace(/_/g, " ")}</td>
                              <td className="px-3 py-2 text-right font-mono text-slate-300">
                                {val.total_return_pct >= 0 ? "+" : ""}{val.total_return_pct.toFixed(1)}%
                              </td>
                              <td className="px-3 py-2 text-right font-mono text-slate-300">
                                {val.sharpe.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </BentoCard>
                )}

                {/* Per-window summary */}
                {results?.per_window && results.per_window.length > 0 && (
                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">Walk-Forward Windows</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-700 text-left">
                            <th className="px-3 py-2 text-slate-400">#</th>
                            <th className="px-3 py-2 text-slate-400">Train Period</th>
                            <th className="px-3 py-2 text-slate-400">Test Period</th>
                            <th className="px-3 py-2 text-right text-slate-400">Candidates</th>
                            <th className="px-3 py-2 text-right text-slate-400">Samples</th>
                            <th className="px-3 py-2 text-right text-slate-400">Features</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.per_window.map((w) => (
                            <tr key={w.window_id} className="border-b border-slate-800">
                              <td className="px-3 py-2 font-mono text-slate-300">{w.window_id}</td>
                              <td className="px-3 py-2 text-xs text-slate-400">
                                {w.train_start} → {w.train_end}
                              </td>
                              <td className="px-3 py-2 text-xs text-slate-400">
                                {w.test_start} → {w.test_end}
                              </td>
                              <td className="px-3 py-2 text-right font-mono text-slate-300">{w.n_candidates}</td>
                              <td className="px-3 py-2 text-right font-mono text-slate-300">{w.n_train_samples}</td>
                              <td className="px-3 py-2 text-right font-mono text-slate-300">{w.n_features}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </BentoCard>
                )}

                {!results && !isRunning && (
                  <p className="text-slate-500">No backtest results yet. Click &quot;Run Backtest&quot; to start.</p>
                )}
              </div>
            )}

            {/* ═══ EQUITY CURVE TAB ═══ */}
            {tab === "equity" && (
              <BentoCard>
                <h3 className="mb-4 text-lg font-semibold text-slate-300">Portfolio Equity Curve</h3>
                {results?.equity_curve && results.equity_curve.length > 0 ? (
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={results.equity_curve} margin={{ top: 5, right: 20, left: 10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        interval="preserveStartEnd"
                        tickCount={10}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#0f172a",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        formatter={(val: number) => [`$${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, "Equity"]}
                      />
                      <Line type="monotone" dataKey="equity" stroke="#38bdf8" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="py-12 text-center text-slate-500">No equity data available</p>
                )}
              </BentoCard>
            )}

            {/* ═══ FEATURES TAB ═══ */}
            {tab === "features" && (
              <BentoCard>
                <h3 className="mb-4 text-lg font-semibold text-slate-300">
                  Mean Decrease Accuracy (MDA) — Top 20 Features
                </h3>
                {avgFeatures.length > 0 ? (
                  <ResponsiveContainer width="100%" height={avgFeatures.length * 32 + 40}>
                    <BarChart
                      data={avgFeatures}
                      layout="vertical"
                      margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fontSize: 11, fill: "#94a3b8" }}
                        width={160}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#0f172a",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        formatter={(val: number) => [val.toFixed(4), "MDA Importance"]}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={20}>
                        {avgFeatures.map((_, i) => (
                          <Cell key={i} fill={i < 5 ? "#38bdf8" : i < 10 ? "#6366f1" : "#475569"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="py-12 text-center text-slate-500">Run a backtest to see feature importance</p>
                )}
              </BentoCard>
            )}

            {/* ═══ OPTIMIZER TAB ═══ */}
            {tab === "optimizer" && (
              <div className="space-y-6">
                {/* Optimizer controls */}
                <div className="flex items-center gap-3">
                  {isOptRunning ? (
                    <button
                      onClick={handleStopOptimizer}
                      className="flex items-center gap-1.5 rounded-lg bg-rose-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-rose-500"
                    >
                      <Stop size={16} weight="fill" />
                      Stop Optimizer
                    </button>
                  ) : (
                    <button
                      onClick={handleStartOptimizer}
                      disabled={!!actionLoading}
                      className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:opacity-50"
                    >
                      <Play size={16} weight="fill" />
                      {actionLoading === "optimizer" ? "Starting..." : "Start Optimizer"}
                    </button>
                  )}
                  <button
                    onClick={refresh}
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:border-slate-600"
                  >
                    <ArrowClockwise size={16} />
                    Refresh
                  </button>
                </div>

                {/* Optimizer status */}
                {optStatus && optStatus.status !== "idle" && (
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
                    <Metric
                      label="Status"
                      value={optStatus.status.toUpperCase()}
                      color={
                        optStatus.status === "running"
                          ? "text-sky-400"
                          : optStatus.status === "completed"
                            ? "text-emerald-400"
                            : "text-slate-300"
                      }
                    />
                    <Metric label="Iterations" value={String(optStatus.iterations)} />
                    <Metric
                      label="Best Sharpe"
                      value={optStatus.best_sharpe?.toFixed(3) ?? "—"}
                      color="text-sky-300"
                    />
                    <Metric
                      label="Best DSR"
                      value={optStatus.best_dsr?.toFixed(3) ?? "—"}
                      color={optStatus.best_dsr != null && optStatus.best_dsr >= 0.95 ? "text-emerald-400" : "text-amber-400"}
                    />
                    <Metric label="Kept" value={String(optStatus.kept)} color="text-emerald-400" />
                    <Metric label="Discarded" value={String(optStatus.discarded)} color="text-rose-400" />
                  </div>
                )}

                {/* Best experiment */}
                {optBest && (
                  <BentoCard>
                    <h3 className="mb-3 text-lg font-semibold text-slate-300">Best Strategy</h3>
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                      <div>
                        <p className="text-xs text-slate-500">Sharpe</p>
                        <p className="font-mono text-lg font-bold text-emerald-400">{optBest.best_sharpe.toFixed(3)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500">DSR</p>
                        <p className="font-mono text-lg font-bold text-sky-300">{optBest.best_dsr.toFixed(3)}</p>
                      </div>
                    </div>
                  </BentoCard>
                )}

                {/* Experiments table */}
                {optExperiments.length > 0 && (
                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">Experiment Log</h3>
                    <div className="max-h-96 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-navy-800">
                          <tr className="border-b border-slate-700 text-left">
                            <th className="px-3 py-2 text-slate-400">#</th>
                            <th className="px-3 py-2 text-slate-400">Modification</th>
                            <th className="px-3 py-2 text-right text-slate-400">Before</th>
                            <th className="px-3 py-2 text-right text-slate-400">After</th>
                            <th className="px-3 py-2 text-right text-slate-400">DSR</th>
                            <th className="px-3 py-2 text-slate-400">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {optExperiments.map((exp, i) => (
                            <tr key={i} className="border-b border-slate-800">
                              <td className="px-3 py-2 font-mono text-xs text-slate-500">{exp.iteration}</td>
                              <td className="max-w-xs truncate px-3 py-2 text-xs text-slate-400" title={exp.modification}>
                                {exp.modification}
                              </td>
                              <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">{parseFloat(exp.metric_before).toFixed(3)}</td>
                              <td className="px-3 py-2 text-right font-mono text-xs text-slate-300">{parseFloat(exp.metric_after).toFixed(3)}</td>
                              <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">{parseFloat(exp.dsr).toFixed(3)}</td>
                              <td className="px-3 py-2">
                                <span
                                  className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                    exp.status === "keep"
                                      ? "bg-emerald-500/20 text-emerald-400"
                                      : exp.status === "BASELINE"
                                        ? "bg-sky-500/20 text-sky-400"
                                        : "bg-slate-700 text-slate-400"
                                  }`}
                                >
                                  {exp.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </BentoCard>
                )}

                {optExperiments.length === 0 && optStatus?.status === "idle" && (
                  <p className="text-slate-500">No experiments yet. Click &quot;Start Optimizer&quot; to begin.</p>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
