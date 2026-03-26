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
  getBacktestRuns,
  loadBacktestRun,
  getOptimizerInsights,
  deleteOptimizerHistory,
  deleteBacktestRun,
} from "@/lib/api";
import type {
  BacktestStatus,
  BacktestProgress,
  BacktestResults,
  BacktestRunSummary,
  IngestionStatus,
  OptimizerStatus,
  OptimizerExperiment,
  OptimizerBest,
  OptimizerInsights,
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
  MagnifyingGlass,
  Brain,
  ShoppingCart,
  ChartBarHorizontal,
  CloudArrowDown,
  CheckCircle,
  Trash,
  XCircle,
} from "@phosphor-icons/react";
import { OptimizerProgressChart } from "@/components/OptimizerProgressChart";
import { OptimizerInsightsView } from "@/components/OptimizerInsights";
import type { Icon } from "@phosphor-icons/react";

/* ── Backtest pipeline step definitions ── */
type PipelineStepKey =
  | "preloading" | "screening" | "building_features" | "training"
  | "computing_mda" | "predicting" | "trading" | "finalizing";

const STEP_ORDER: PipelineStepKey[] = [
  "preloading", "screening", "building_features", "training",
  "computing_mda", "predicting", "trading", "finalizing",
];

const STEP_META: Record<PipelineStepKey, { StepIcon: Icon; label: string }> = {
  preloading:        { StepIcon: CloudArrowDown,     label: "Load Market Data" },
  screening:         { StepIcon: MagnifyingGlass,    label: "Screen Universe" },
  building_features: { StepIcon: Database,           label: "Build Features" },
  training:          { StepIcon: Brain,              label: "Train ML Model" },
  computing_mda:     { StepIcon: ChartBarHorizontal, label: "Feature Importance (MDA)" },
  predicting:        { StepIcon: TrendUp,            label: "Predict Signals" },
  trading:           { StepIcon: ShoppingCart,       label: "Execute Trades" },
  finalizing:        { StepIcon: ChartLineUp,        label: "Finalize & Report" },
};

/* ── Helpers ── */
function formatRunTimestamp(ts: string): string {
  // Parse compact ISO like "20260323T081929Z" or full ISO like "2026-03-23T18:50:16.400976+00:00"
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z?)$/);
  if (m) {
    const d = m[7] === "Z"
      ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]))
      : new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    if (!isNaN(d.getTime())) return d.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
  }
  // Try standard ISO parse (handles full ISO 8601 with timezone)
  const d = new Date(ts);
  if (!isNaN(d.getTime())) return d.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
  return ts;
}

function Metric({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-xl font-bold ${color ?? "text-slate-100"}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

type Tab = "results" | "equity" | "features" | "optimizer" | "insights";
const TABS: { id: Tab; label: string; icon: Icon }[] = [
  { id: "results", label: "Results", icon: Table },
  { id: "equity", label: "Equity Curve", icon: ChartLineUp },
  { id: "features", label: "Features", icon: TrendUp },
  { id: "optimizer", label: "Optimizer", icon: Lightning },
  { id: "insights", label: "Insights", icon: MagnifyingGlass },
];

export default function BacktestPage() {
  const [btStatus, setBtStatus] = useState<BacktestStatus | null>(null);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [ingestion, setIngestion] = useState<IngestionStatus | null>(null);
  const [optStatus, setOptStatus] = useState<OptimizerStatus | null>(null);
  const [optExperiments, setOptExperiments] = useState<OptimizerExperiment[]>([]);
  const [optBest, setOptBest] = useState<OptimizerBest | null>(null);
  // optRuns kept for experiment count display; optRunIndex removed (always latest=0)
  const [runs, setRuns] = useState<BacktestRunSummary[]>([]);
  const [insights, setInsights] = useState<OptimizerInsights | null>(null);
  const [tab, setTab] = useState<Tab>("results");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ingestResult, setIngestResult] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [localElapsed, setLocalElapsed] = useState(0);
  const [tradePage, setTradePage] = useState(0);
  const [tradeSort, setTradeSort] = useState<{ col: string; asc: boolean }>({ col: "entry_date", asc: true });

  const refresh = useCallback(async (retryCount = 0) => {
    try {
      const [s, ing, opt, runsRes] = await Promise.all([
        getBacktestStatus().catch((e) => { console.warn("getBacktestStatus:", e.message); return null; }),
        getIngestionStatus().catch((e) => { console.warn("getIngestionStatus:", e.message); return null; }),
        getOptimizerStatus().catch((e) => { console.warn("getOptimizerStatus:", e.message); return null; }),
        getBacktestRuns().catch((e) => { console.warn("getBacktestRuns:", e.message); return null; }),
      ]);

      // If ALL primary calls failed, auto-retry once after 2s (covers backend restart)
      if (!s && !ing && !opt && !runsRes) {
        if (retryCount < 1) {
          await new Promise((r) => setTimeout(r, 2000));
          return refresh(retryCount + 1);
        }
        setError("Cannot load backtest data — backend may be down or unresponsive.");
        setLoading(false);
        return;
      }

      if (s) setBtStatus(s);
      if (ing) setIngestion(ing);
      if (opt) setOptStatus(opt);
      if (runsRes) setRuns(runsRes.runs);

      // Parallel fetch of conditional data
      const [r, exp, best] = await Promise.all([
        s?.has_result ? getBacktestResults().catch((e) => { console.warn("getBacktestResults:", e.message); return null; }) : Promise.resolve(null),
        opt?.status === "running"
          ? getOptimizerExperiments(opt.run_id).catch((e) => { console.warn("getOptimizerExperiments:", e.message); return null; })
          : getOptimizerExperiments(undefined, 0).catch((e) => { console.warn("getOptimizerExperiments:", e.message); return null; }),
        getOptimizerBest().catch((e) => { console.warn("getOptimizerBest:", e.message); return null; }),
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
  // When optimizer is running, also poll experiments for live chart updates
  const refreshStatus = useCallback(async () => {
    try {
      const [s, opt] = await Promise.all([
        getBacktestStatus().catch(() => null),
        getOptimizerStatus().catch(() => null),
      ]);
      if (s) setBtStatus(s);
      if (opt) setOptStatus(opt);

      // Live experiment updates while optimizer is running
      if (opt?.status === "running") {
        const [exp, best] = await Promise.all([
          getOptimizerExperiments(opt.run_id).catch(() => null),
          getOptimizerBest().catch(() => null),
        ]);
        if (exp) setOptExperiments(exp.experiments);
        if (best) setOptBest(best);
      }
    } catch {
      // Swallow poll errors — the initial refresh already set error state if needed
    }
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
    const id = setInterval(refreshStatus, 2000);
    return () => clearInterval(id);
  }, [btStatus?.status, optStatus?.status, refresh, refreshStatus, prevRunning]);

  // Client-side elapsed timer — ticks every second while running
  useEffect(() => {
    if (btStatus?.status !== "running") { setLocalElapsed(0); return; }
    const id = setInterval(() => setLocalElapsed((t: number) => t + 1), 1000);
    return () => clearInterval(id);
  }, [btStatus?.status]);

  // Sync elapsed from server on each poll (server value is authoritative)
  useEffect(() => {
    const p = btStatus?.progress;
    if (p && typeof p === "object") {
      const s = (p as BacktestProgress).elapsed_seconds;
      if (typeof s === "number") setLocalElapsed(s);
    }
  }, [btStatus?.progress]);

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
    setTab("optimizer");
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

  const handleDeleteRun = async (runId: string) => {
    try {
      await deleteBacktestRun(runId);
      setRuns((prev) => prev.filter((r) => r.run_id !== runId));
      // If we deleted the currently viewed run, load the next one or clear
      if (results?.run_id === runId) {
        const remaining = runs.filter((r) => r.run_id !== runId);
        if (remaining.length > 0) {
          const data = await loadBacktestRun(remaining[0].run_id);
          setResults(data);
        } else {
          setResults(null);
        }
      }
    } catch { /* ignore */ }
  };

  const handleClearHistory = async () => {
    if (!confirm("Delete all optimizer experiments and best params? This cannot be undone.")) return;
    setActionLoading("clear-history");
    try {
      await deleteOptimizerHistory();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to clear history");
    } finally {
      setActionLoading(null);
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
      <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
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
              disabled={!!actionLoading || isRunning || isOptRunning}
              title={isOptRunning ? "Optimizer is running - backtest unavailable" : "Runs walk-forward ML backtest on ingested data. Uses GradientBoosting (no LLM cost). BQ reads <$0.01. Takes ~2-5 min."}
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
            {error.includes("Cannot") && (
              <p className="mt-1 text-xs text-rose-300/60">
                Make sure the backend is running: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
              </p>
            )}
            <button onClick={() => { setError(null); setLoading(true); refresh(); }} className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60">
              Retry
            </button>
          </div>
        )}

        {/* Backtest error with traceback */}
        {btStatus?.status === "error" && btStatus.error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3">
            <p className="text-sm font-medium text-rose-400">Backtest Error</p>
            <p className="mt-1 font-mono text-xs text-rose-300/80">{btStatus.error}</p>
            {btStatus.traceback && (
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-rose-400/70 hover:text-rose-400">Show traceback</summary>
                <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-black/40 p-2 font-mono text-[11px] leading-relaxed text-rose-300/70">{btStatus.traceback}</pre>
              </details>
            )}
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

        {/* Progress panel — vertical Jira-style workflow timeline */}
        {isRunning && (() => {
          const p = btStatus?.progress;
          const prog: BacktestProgress | null =
            p && typeof p === "object" ? (p as BacktestProgress) : null;

          const mins = Math.floor(localElapsed / 60);
          const secs = Math.floor(localElapsed % 60);
          const elapsedStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

          if (!prog) {
            return (
              <div className="mb-4 rounded-xl border border-sky-500/30 bg-sky-950/20 p-4">
                <div className="flex items-center gap-2">
                  <ArrowClockwise size={16} className="animate-spin text-sky-400" />
                  <span className="text-sm text-sky-300">
                    {typeof p === "string" ? p : "Running backtest..."}
                  </span>
                  <span className="ml-auto font-mono text-xs text-slate-500">{elapsedStr} elapsed</span>
                </div>
              </div>
            );
          }

          const isFinalizing = prog.step === "finalizing";
          const totalW = prog.total_windows || 0;
          const currentW = isFinalizing ? totalW + 1 : (prog.window || 0);
          const windowPct = isFinalizing ? 100
            : totalW > 0 ? Math.round(Math.max(0, (currentW - 1) / totalW) * 100) : 0;
          const currentStepIdx = STEP_ORDER.indexOf(prog.step as PipelineStepKey);
          const totalCacheOps = (prog.cache_hits || 0) + (prog.cache_misses || 0);
          const cacheHitPct = totalCacheOps > 0
            ? Math.round((prog.cache_hits / totalCacheOps) * 100) : 0;

          return (
            <details open className="group mb-6 rounded-xl border border-slate-700/60 bg-[#080f1e] shadow-xl">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-5 [&::-webkit-details-marker]:hidden">
                <div className="flex min-w-0 flex-1 flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-60" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-sky-400" />
                    </span>
                    <span className="text-sm font-semibold text-slate-200">Walk-Forward Progress{btStatus?.engine_source === "optimizer" ? " (via Optimizer)" : ""}</span>
                    <span className="text-xs text-slate-500">Step {currentStepIdx + 1}/{STEP_ORDER.length}</span>
                  </div>
                  {/* Optimizer step subtitle + metric pills */}
                  {btStatus?.engine_source === "optimizer" && isOptRunning && optStatus && (
                    <div className="flex flex-wrap items-center gap-1.5 pl-[18px]">
                      {optStatus.current_step && (
                        <span className="text-xs text-slate-500">
                          {optStatus.current_step === "establishing_baseline" ? "Establishing Baseline"
                            : optStatus.current_step === "running_experiment" ? "Running Experiment"
                            : optStatus.current_step === "baseline_complete" ? "Baseline Complete"
                            : optStatus.current_step === "evaluated" ? "Evaluated"
                            : optStatus.current_step}
                          {optStatus.current_detail ? ` — ${optStatus.current_detail}` : ""}
                        </span>
                      )}
                      <span className="mx-1 hidden text-slate-700 sm:inline">|</span>
                      <span className="rounded-full border border-slate-700 bg-slate-800/60 px-2 py-0.5 font-mono text-[11px] text-slate-300">Iter {optStatus.iterations}</span>
                      <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 font-mono text-[11px] text-sky-300">{optStatus.best_sharpe?.toFixed(3) ?? "—"} Sharpe</span>
                      <span className={`rounded-full border px-2 py-0.5 font-mono text-[11px] ${optStatus.best_dsr != null && optStatus.best_dsr >= 0.95 ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-amber-500/30 bg-amber-500/10 text-amber-400"}`}>{optStatus.best_dsr?.toFixed(3) ?? "—"} DSR</span>
                      {optStatus.kept > 0 && <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] text-emerald-400">{optStatus.kept} kept</span>}
                      {optStatus.discarded > 0 && <span className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 font-mono text-[11px] text-rose-400">{optStatus.discarded} disc</span>}
                    </div>
                  )}
                </div>
                <div className="flex flex-shrink-0 items-center gap-3">
                  <span className="font-mono text-sm text-slate-400">{elapsedStr} elapsed</span>
                  {btStatus?.engine_source === "optimizer" && (
                    <button
                      onClick={(e) => { e.preventDefault(); handleStopOptimizer(); }}
                      className="flex items-center gap-1 rounded-md bg-rose-600/80 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-rose-500"
                    >
                      <Stop size={12} weight="fill" />
                      Stop
                    </button>
                  )}
                </div>
              </summary>
              <div className="px-5 pb-5">
              {/* Window rail + overall bar */}
              {totalW > 0 && (
                <div className="mb-5">
                  <div className="mb-2 flex items-center gap-2">
                    <div className="flex flex-1 flex-wrap items-center gap-1.5">
                      {Array.from({ length: totalW }, (_, i) => {
                        const wn = i + 1;
                        const wState = wn < currentW ? "done" : wn === currentW ? "active" : "pending";
                        return (
                          <div
                            key={wn}
                            title={`Window ${wn}`}
                            className={`h-2.5 w-2.5 flex-shrink-0 rounded-full transition-all duration-300 ${
                              wState === "done"
                                ? "bg-emerald-500"
                                : wState === "active"
                                ? "bg-sky-400 ring-2 ring-sky-400/40 ring-offset-1 ring-offset-[#080f1e]"
                                : "border border-slate-600 bg-transparent"
                            }`}
                          />
                        );
                      })}
                    </div>
                    <span className="ml-1 whitespace-nowrap font-mono text-xs font-medium text-slate-400">
                      {isFinalizing ? `${totalW} / ${totalW}` : `${Math.max(0, prog.window || 0)} / ${totalW}`}
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-1.5 rounded-full bg-sky-500 transition-all duration-700"
                      style={{ width: `${windowPct}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Vertical timeline */}
              <div>
                {STEP_ORDER.map((stepName, idx) => {
                  const isLastStep = idx === STEP_ORDER.length - 1;
                  const stepIdx = STEP_ORDER.indexOf(stepName);
                  const stepStatus: "done" | "active" | "pending" =
                    stepIdx < currentStepIdx ? "done"
                    : stepIdx === currentStepIdx ? "active"
                    : "pending";
                  const { StepIcon, label } = STEP_META[stepName];
                  const isDone   = stepStatus === "done";
                  const isActive = stepStatus === "active";

                  return (
                    <div key={stepName} className="relative flex gap-3">
                      {/* Vertical connector */}
                      {!isLastStep && (
                        <div
                          className={`absolute z-0 w-px ${
                            isDone ? "bg-emerald-500/25" : "bg-slate-700/50"
                          }`}
                          style={{ left: 15, top: 32, bottom: 0 }}
                        />
                      )}

                      {/* Step icon bubble */}
                      <div
                        className={`relative z-10 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border-2 transition-all duration-300 ${
                          isDone
                            ? "border-emerald-500/70 bg-emerald-950/50 text-emerald-400"
                            : isActive
                            ? "border-sky-400 bg-sky-950/60 text-sky-300 shadow-[0_0_14px_rgba(56,189,248,0.2)]"
                            : "border-slate-700/70 bg-[#060d1a] text-slate-700"
                        }`}
                      >
                        {isDone
                          ? <CheckCircle size={15} weight="fill" />
                          : <StepIcon size={13} weight={isActive ? "bold" : "regular"} />
                        }
                      </div>

                      {/* Step label + badge + detail */}
                      <div className={`flex-1 min-w-0${!isLastStep ? " pb-4" : ""}`}>
                        <div className="flex items-center gap-2">
                          <span className={`text-sm ${
                            isDone    ? "text-slate-500" :
                            isActive  ? "font-medium text-slate-100" :
                                        "text-slate-700"
                          }`}>
                            {label}
                          </span>
                          <div className="flex-1" />
                          {isDone && (
                            <span className="text-[11px] text-emerald-600">Complete</span>
                          )}
                          {isActive && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-sky-500/25 bg-sky-500/10 px-2 py-0.5 text-[10px] font-semibold text-sky-400">
                              <ArrowClockwise size={9} className="animate-spin" />
                              In Progress
                            </span>
                          )}
                          {stepStatus === "pending" && (
                            <span className="text-[11px] text-slate-700">Queued</span>
                          )}
                        </div>

                        {isActive && prog.step_detail && (
                          <p className="mt-0.5 truncate text-xs text-slate-500">{prog.step_detail}</p>
                        )}

                        {isActive && stepName === "building_features" && (prog.samples_built ?? 0) > 0 && (
                          <div className="mt-2 flex items-center gap-2">
                            <div className="h-1 flex-1 overflow-hidden rounded-full bg-slate-800">
                              <div
                                className="h-1 rounded-full bg-sky-600/80 transition-all duration-300"
                                style={{
                                  width: `${Math.min(100, Math.round(
                                    ((prog.samples_built ?? 0) / Math.max(1, prog.samples_total ?? 1)) * 100
                                  ))}%`,
                                }}
                              />
                            </div>
                            <span className="whitespace-nowrap text-xs tabular-nums text-slate-600">
                              {(prog.samples_built ?? 0).toLocaleString()} / {(prog.samples_total ?? 0).toLocaleString()} samples
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Footer: cache stats */}
              {totalCacheOps > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-800 pt-3 text-xs text-slate-600">
                  <span className="text-slate-500">Cache</span>
                  <span className={cacheHitPct >= 90 ? "text-emerald-600" : cacheHitPct >= 70 ? "text-amber-600" : "text-rose-600"}>
                    {cacheHitPct}% hit rate
                  </span>
                  <span>·</span>
                  <span>{(prog.cache_hits || 0).toLocaleString()} hits</span>
                  <span>·</span>
                  <span>{(prog.cache_misses || 0).toLocaleString()} misses</span>
                </div>
              )}
              </div>
            </details>
          );
        })()}

        {loading && <PageSkeleton />}

        {/* Unified run selector (Tier 4) — hidden on optimizer/insights tabs per layout spec */}
        {runs.length > 0 && !loading && tab !== "optimizer" && tab !== "insights" && (() => {
          const baselines = runs.filter((r) => r.is_baseline);
          const experiments = runs.filter((r) => !r.is_baseline);

          return (
            <div className="mb-4 flex items-center gap-2">
              <span className="text-xs text-slate-500">Run:</span>
              <select
                className="max-w-md rounded-lg border border-slate-700 bg-navy-800/80 px-3 py-1.5 text-xs text-slate-300 focus:border-sky-500 focus:outline-none"
                value={results?.run_id ?? ""}
                onChange={async (e) => {
                  const rid = e.target.value;
                  if (!rid) return;
                  const run = runs.find((r) => r.run_id === rid);
                  if (!run?.has_detail) {
                    setResults(null);
                    setBtStatus((prev) => prev ? { ...prev, status: "completed", has_result: false, run_id: rid } : prev);
                    return;
                  }
                  try {
                    const data = await loadBacktestRun(rid);
                    setResults(data);
                    setBtStatus((prev) => prev ? { ...prev, status: "completed", has_result: true, run_id: rid } : prev);
                  } catch { /* ignore load errors */ }
                }}
              >
                {baselines.map((b) => {
                  const children = experiments.filter((e) => e.parent_run_id === b.run_id);
                  const bSharpe = b.sharpe?.toFixed(2) ?? "?";
                  return (
                    <optgroup key={b.run_id} label={`${b.strategy} -- Sharpe ${bSharpe}`}>
                      <option value={b.run_id}>
                        Baseline -- {formatRunTimestamp(b.timestamp)} -- Sharpe {bSharpe}{!b.has_detail ? " (summary)" : ""}
                      </option>
                      {children.map((c) => {
                        const delta = (b.sharpe != null && c.sharpe != null)
                          ? (c.sharpe - b.sharpe).toFixed(2)
                          : "?";
                        const deltaPrefix = c.sharpe != null && b.sharpe != null && c.sharpe >= b.sharpe ? "+" : "";
                        const statusTag = c.status === "keep" ? "[kept]" : c.status === "discard" ? "[disc]" : c.status === "dsr_reject" ? "[dsr]" : "";
                        return (
                          <option key={c.run_id} value={c.run_id}>
                            {statusTag} {c.param_changed || c.run_id} -- Sharpe {c.sharpe?.toFixed(2) ?? "?"} ({deltaPrefix}{delta}){!c.has_detail ? " (summary)" : ""}
                          </option>
                        );
                      })}
                    </optgroup>
                  );
                })}
                {/* Experiments without a matching baseline parent */}
                {experiments.filter((e) => !baselines.some((b) => b.run_id === e.parent_run_id)).length > 0 && (
                  <optgroup label="Unlinked">
                    {experiments
                      .filter((e) => !baselines.some((b) => b.run_id === e.parent_run_id))
                      .map((e) => (
                        <option key={e.run_id} value={e.run_id}>
                          {e.run_id} -- Sharpe {e.sharpe?.toFixed(2) ?? "?"}{!e.has_detail ? " (summary)" : ""}
                        </option>
                      ))}
                  </optgroup>
                )}
              </select>
              {results?.run_id && (
                <button
                  onClick={() => {
                    const rid = results?.run_id;
                    if (!rid) return;
                    if (!confirm("Delete this backtest run?")) return;
                    handleDeleteRun(rid);
                  }}
                  className="rounded p-1 text-rose-500/70 transition-colors hover:bg-rose-500/10 hover:text-rose-400"
                  title="Delete selected run"
                >
                  <XCircle size={16} weight="fill" />
                </button>
              )}
            </div>
          );
        })()}

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
                {/* Data ingestion summary */}
                {ingestion && (
                  <div>
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
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
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
                            <th className="px-3 py-2 text-right text-slate-400">Excess Return</th>
                            <th className="px-3 py-2 text-right text-slate-400">Sharpe Δ</th>
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
                            <td className="px-3 py-2 text-right text-slate-500">—</td>
                            <td className="px-3 py-2 text-right text-slate-500">—</td>
                          </tr>
                          {Object.entries(results.baselines).map(([key, val]) => {
                            const retDelta = a ? a.total_return_pct - val.total_return_pct : null;
                            const sharpeDelta = a ? a.sharpe - val.sharpe : null;
                            return (
                              <tr key={key} className="border-b border-slate-800">
                                <td className="px-3 py-2 text-slate-400 capitalize">{key.replace(/_/g, " ")}</td>
                                <td className="px-3 py-2 text-right font-mono text-slate-300">
                                  {val.total_return_pct >= 0 ? "+" : ""}{val.total_return_pct.toFixed(1)}%
                                </td>
                                <td className="px-3 py-2 text-right font-mono text-slate-300">
                                  {val.sharpe.toFixed(2)}
                                </td>
                                <td className={`px-3 py-2 text-right font-mono ${retDelta != null ? (retDelta >= 0 ? "text-emerald-400" : "text-rose-400") : "text-slate-500"}`}>
                                  {retDelta != null ? `${retDelta >= 0 ? "+" : ""}${retDelta.toFixed(1)}%` : "—"}
                                </td>
                                <td className={`px-3 py-2 text-right font-mono ${sharpeDelta != null ? (sharpeDelta >= 0 ? "text-emerald-400" : "text-rose-400") : "text-slate-500"}`}>
                                  {sharpeDelta != null ? `${sharpeDelta >= 0 ? "+" : ""}${sharpeDelta.toFixed(2)}` : "—"}
                                </td>
                              </tr>
                            );
                          })}
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

                {/* ── Trade Statistics ── */}
                {results?.trade_statistics && (() => {
                  const ts = results.trade_statistics;
                  return (
                    <BentoCard>
                      <h3 className="mb-4 text-lg font-semibold text-slate-300">Trade Statistics</h3>
                      <div className="grid gap-4 sm:grid-cols-3">
                        {/* Performance */}
                        <div className="space-y-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-400">Performance</p>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <span className="text-slate-400">Profit Factor</span>
                            <span className="text-right font-mono text-slate-200">{ts.profit_factor.toFixed(2)}</span>
                            <span className="text-slate-400">Win Rate</span>
                            <span className="text-right font-mono text-slate-200">{(ts.win_rate * 100).toFixed(1)}%</span>
                            <span className="text-slate-400">Payoff Ratio</span>
                            <span className="text-right font-mono text-slate-200">{ts.payoff_ratio.toFixed(2)}</span>
                            <span className="text-slate-400">Expectancy</span>
                            <span className={`text-right font-mono ${ts.expectancy >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                              ${ts.expectancy.toFixed(2)}
                            </span>
                            <span className="text-slate-400">SQN</span>
                            <span className="text-right font-mono text-slate-200">{ts.sqn.toFixed(2)}</span>
                          </div>
                        </div>
                        {/* Extremes */}
                        <div className="space-y-2 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-amber-400">Extremes &amp; Streaks</p>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <span className="text-slate-400">Best Trade</span>
                            <span className="text-right font-mono text-emerald-400">{(ts.best_trade * 100).toFixed(1)}%</span>
                            <span className="text-slate-400">Worst Trade</span>
                            <span className="text-right font-mono text-rose-400">{(ts.worst_trade * 100).toFixed(1)}%</span>
                            <span className="text-slate-400">Win Streak</span>
                            <span className="text-right font-mono text-slate-200">{ts.max_win_streak}</span>
                            <span className="text-slate-400">Loss Streak</span>
                            <span className="text-right font-mono text-slate-200">{ts.max_loss_streak}</span>
                            <span className="text-slate-400">Avg Days (W)</span>
                            <span className="text-right font-mono text-slate-200">{ts.avg_holding_days_win.toFixed(1)}</span>
                            <span className="text-slate-400">Avg Days (L)</span>
                            <span className="text-right font-mono text-slate-200">{ts.avg_holding_days_loss.toFixed(1)}</span>
                          </div>
                        </div>
                        {/* Cost Impact */}
                        <div className="space-y-2 rounded-lg border border-rose-500/20 bg-rose-500/5 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-rose-400">Cost Impact</p>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <span className="text-slate-400">Total Commission</span>
                            <span className="text-right font-mono text-slate-200">${ts.total_commission.toFixed(0)}</span>
                            <span className="text-slate-400">Comm % of Profit</span>
                            <span className="text-right font-mono text-slate-200">{ts.commission_pct_of_profit.toFixed(1)}%</span>
                            <span className="text-slate-400">Avg Cost/Trade</span>
                            <span className="text-right font-mono text-slate-200">${ts.avg_cost_per_trade.toFixed(2)}</span>
                            <span className="text-slate-400">Turnover</span>
                            <span className="text-right font-mono text-slate-200">{ts.turnover_rate.toFixed(1)}</span>
                            <span className="text-slate-400">Break-Even WR</span>
                            <span className="text-right font-mono text-slate-200">{(ts.break_even_win_rate * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    </BentoCard>
                  );
                })()}

                {/* ── Trade List ── */}
                {results?.trades && results.trades.length > 0 && (() => {
                  const TRADES_PER_PAGE = 25;
                  const sorted = [...results.trades].sort((a, b) => {
                    const va = a[tradeSort.col as keyof typeof a];
                    const vb = b[tradeSort.col as keyof typeof b];
                    if (typeof va === "number" && typeof vb === "number") return tradeSort.asc ? va - vb : vb - va;
                    return tradeSort.asc
                      ? String(va).localeCompare(String(vb))
                      : String(vb).localeCompare(String(va));
                  });
                  const totalPages = Math.ceil(sorted.length / TRADES_PER_PAGE);
                  const page = Math.min(tradePage, totalPages - 1);
                  const visible = sorted.slice(page * TRADES_PER_PAGE, (page + 1) * TRADES_PER_PAGE);
                  const toggleSort = (col: string) =>
                    setTradeSort((s) => ({ col, asc: s.col === col ? !s.asc : true }));
                  const hdr = (label: string, col: string, right = false) => (
                    <th
                      className={`cursor-pointer px-2 py-2 text-slate-400 hover:text-slate-200 ${right ? "text-right" : "text-left"}`}
                      onClick={() => toggleSort(col)}
                    >
                      {label}{tradeSort.col === col ? (tradeSort.asc ? " ▲" : " ▼") : ""}
                    </th>
                  );
                  return (
                    <BentoCard>
                      <div className="mb-3 flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-slate-300">
                          Trade List <span className="text-sm font-normal text-slate-500">({results.trades.length} round-trips)</span>
                        </h3>
                        {totalPages > 1 && (
                          <div className="flex items-center gap-2 text-sm text-slate-400">
                            <button
                              disabled={page === 0}
                              onClick={() => setTradePage((p) => Math.max(0, p - 1))}
                              className="rounded px-2 py-0.5 hover:bg-slate-700 disabled:opacity-30"
                            >
                              ← Prev
                            </button>
                            <span>{page + 1} / {totalPages}</span>
                            <button
                              disabled={page >= totalPages - 1}
                              onClick={() => setTradePage((p) => Math.min(totalPages - 1, p + 1))}
                              className="rounded px-2 py-0.5 hover:bg-slate-700 disabled:opacity-30"
                            >
                              Next →
                            </button>
                          </div>
                        )}
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-700">
                              <th className="px-2 py-2 text-left text-slate-400">#</th>
                              {hdr("Ticker", "ticker")}
                              {hdr("Entry", "entry_date")}
                              {hdr("Exit", "exit_date")}
                              {hdr("Entry $", "entry_price", true)}
                              {hdr("Exit $", "exit_price", true)}
                              {hdr("Qty", "quantity", true)}
                              {hdr("P&L $", "net_pnl", true)}
                              {hdr("P&L %", "pnl_pct", true)}
                              {hdr("Days", "holding_days", true)}
                              {hdr("Conf", "probability", true)}
                            </tr>
                          </thead>
                          <tbody>
                            {visible.map((t, i) => (
                              <tr
                                key={`${t.ticker}-${t.entry_date}-${i}`}
                                className={`border-b border-slate-800 ${t.net_pnl >= 0 ? "bg-emerald-500/5" : "bg-rose-500/5"}`}
                              >
                                <td className="px-2 py-1.5 font-mono text-slate-500">{page * TRADES_PER_PAGE + i + 1}</td>
                                <td className="px-2 py-1.5 font-medium text-slate-200">{t.ticker}</td>
                                <td className="px-2 py-1.5 text-slate-400">{t.entry_date}</td>
                                <td className="px-2 py-1.5 text-slate-400">{t.exit_date}</td>
                                <td className="px-2 py-1.5 text-right font-mono text-slate-300">${t.entry_price.toFixed(2)}</td>
                                <td className="px-2 py-1.5 text-right font-mono text-slate-300">${t.exit_price.toFixed(2)}</td>
                                <td className="px-2 py-1.5 text-right font-mono text-slate-300">{t.quantity}</td>
                                <td className={`px-2 py-1.5 text-right font-mono ${t.net_pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                  {t.net_pnl >= 0 ? "+" : ""}{t.net_pnl.toFixed(2)}
                                </td>
                                <td className={`px-2 py-1.5 text-right font-mono ${t.pnl_pct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                  {t.pnl_pct >= 0 ? "+" : ""}{(t.pnl_pct * 100).toFixed(1)}%
                                </td>
                                <td className="px-2 py-1.5 text-right font-mono text-slate-300">{t.holding_days}</td>
                                <td className="px-2 py-1.5 text-right font-mono text-slate-400">{(t.probability * 100).toFixed(0)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </BentoCard>
                  );
                })()}

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
                    <div className="flex items-center gap-2 rounded-lg border border-sky-500/20 bg-sky-500/5 px-4 py-2">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-sky-400" />
                      <span className="text-sm text-slate-400">Optimizer running — see progress above ↑</span>
                    </div>
                  ) : (
                    <button
                      onClick={handleStartOptimizer}
                      disabled={!!actionLoading || isRunning}
                      title={isRunning ? "Backtest is running - optimizer unavailable" : undefined}
                      className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:opacity-50"
                    >
                      <Play size={16} weight="fill" />
                      {actionLoading === "optimizer" ? "Starting..." : "Start Optimizer"}
                    </button>
                  )}
                  <button
                    onClick={() => refresh()}
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:border-slate-600"
                  >
                    <ArrowClockwise size={16} />
                    Refresh
                  </button>
                  {optExperiments.length > 0 && (
                    <button
                      onClick={handleClearHistory}
                      disabled={!!actionLoading || isOptRunning}
                      className="flex items-center gap-1.5 rounded-lg border border-rose-700/50 px-3 py-2 text-sm text-rose-400 hover:border-rose-600 hover:bg-rose-500/10 disabled:opacity-50"
                    >
                      <Trash size={16} />
                      {actionLoading === "clear-history" ? "Clearing..." : "Clear History"}
                    </button>
                  )}
                </div>

                {/* Optimizer run selector — shows optimization sessions from TSV */}
                {runs.length > 0 && (() => {
                  const optBaselines = runs.filter((r) => r.is_baseline);
                  return optBaselines.length > 1 ? (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">Optimization run:</span>
                      <select
                        className="max-w-md rounded-lg border border-slate-700 bg-navy-800/80 px-3 py-1.5 text-xs text-slate-300 focus:border-sky-500 focus:outline-none"
                        onChange={async (e) => {
                          const idx = parseInt(e.target.value, 10);
                          if (isNaN(idx)) return;
                          try {
                            const data = await getOptimizerExperiments(undefined, idx);
                            setOptExperiments(data.experiments);
                          } catch { /* ignore */ }
                        }}
                      >
                        {optBaselines.map((b, i) => {
                          // Count experiments belonging to this baseline from current optExperiments
                          const expCount = optExperiments.filter((e) => e.parent_run_id === b.run_id && e.status !== "BASELINE").length;
                          return (
                            <option key={b.run_id} value={i}>
                              {b.strategy} -- {formatRunTimestamp(b.timestamp)} -- Sharpe {b.sharpe?.toFixed(2) ?? "?"} -- {expCount || "loading..."} experiments
                            </option>
                          );
                        })}
                      </select>
                    </div>
                  ) : null;
                })()}

                {/* Optimizer status — full cards only when NOT running (completed/error/stopped) */}
                {optStatus && optStatus.status !== "idle" && optStatus.status !== "running" && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
                      <Metric
                        label="Status"
                        value={optStatus.status.toUpperCase()}
                        color={
                          optStatus.status === "completed"
                            ? "text-emerald-400"
                            : optStatus.status === "error"
                              ? "text-rose-400"
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

                    {/* Error banner */}
                    {optStatus.status === "error" && optStatus.error && (
                      <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3">
                        <p className="text-sm font-medium text-rose-400">Optimizer Error</p>
                        <p className="mt-1 font-mono text-xs text-rose-300/80">{optStatus.error}</p>
                        {optStatus.traceback && (
                          <details className="mt-2">
                            <summary className="cursor-pointer text-xs text-rose-400/70 hover:text-rose-400">Show traceback</summary>
                            <pre className="mt-1 max-h-64 overflow-auto scrollbar-thin whitespace-pre-wrap rounded bg-black/40 p-2 font-mono text-[11px] leading-relaxed text-rose-300/70">{optStatus.traceback}</pre>
                          </details>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Karpathy progress chart */}
                <BentoCard>
                  <OptimizerProgressChart experiments={optExperiments} />
                </BentoCard>

                {/* Experiments table */}
                {optExperiments.length > 0 && (
                  <BentoCard>
                    <h3 className="mb-4 text-lg font-semibold text-slate-300">Experiment Log</h3>
                    <div className="max-h-96 overflow-y-auto scrollbar-thin">
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
                              <td className="px-3 py-2 font-mono text-xs text-slate-500">
                                {exp.status === "BASELINE" 
                                  ? `${exp.run_id.slice(0,8)}`
                                  : `${exp.parent_run_id?.slice(0,8) || "?"}-${String(i).padStart(2,"0")}`}
                              </td>
                              <td className="max-w-xs truncate px-3 py-2 text-xs text-slate-400" title={exp.param_changed}>
                                {exp.status === "BASELINE" ? (exp.param_changed || "baseline") : (exp.param_changed || exp.run_id)}
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

            {/* ═══ INSIGHTS TAB ═══ */}
            {tab === "insights" && (
              <OptimizerInsightsView
                insights={insights}
                onRefresh={async () => {
                  const data = await getOptimizerInsights().catch(() => null);
                  if (data) setInsights(data);
                }}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
