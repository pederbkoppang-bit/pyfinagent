"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { Sidebar } from "@/components/Sidebar";
import { PageSkeleton } from "@/components/Skeleton";
import {
  getPaperTradingStatus,
  getPaperPortfolio,
  getPaperTrades,
  getPaperSnapshots,
  getPaperPerformance,
  getPaperReconciliation,
  getPaperGate,
  startPaperTrading,
  stopPaperTrading,
  triggerPaperTradingCycle,
} from "@/lib/api";
import type {
  PaperTradingStatus,
  PaperPosition,
  PaperTrade,
  PaperSnapshot,
  PaperPerformance,
  PaperPortfolio,
  PaperReconciliation,
} from "@/lib/types";
import { PaperReconciliationChart } from "@/components/PaperReconciliationChart";
import { GoLiveGateWidget, type GoLiveGate } from "@/components/GoLiveGateWidget";
import { AgentRationaleDrawer } from "@/components/AgentRationaleDrawer";
import { KillSwitchPanel } from "@/components/KillSwitchPanel";
import { CycleHealthStrip } from "@/components/CycleHealthStrip";
import { MfeMaeScatter } from "@/components/MfeMaeScatter";
import { useLivePrices } from "@/lib/useLivePrices";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ── Small formatting helpers ─────────────────────────────────────

function PnlBadge({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-slate-500">—</span>;
  const isPositive = value >= 0;
  return (
    <span className={isPositive ? "text-emerald-400" : "text-rose-400"}>
      {isPositive ? "+" : ""}
      {value.toFixed(2)}%
    </span>
  );
}

function Dollar({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-slate-500">—</span>;
  return (
    <span className="text-slate-200">
      ${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
    </span>
  );
}

// ── Metric hero (globally relevant, stays above the tabs) ─────────

function MetricCard({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold">{children}</p>
    </div>
  );
}

function SummaryHero({
  status,
  perf,
}: {
  status: PaperTradingStatus | null;
  perf: PaperPerformance | null;
}) {
  const pnl = status?.portfolio.pnl_pct ?? 0;
  const bench = status?.portfolio.benchmark_return_pct ?? 0;
  return (
    <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-6">
      <MetricCard label="NAV"><Dollar value={status?.portfolio.nav} /></MetricCard>
      <MetricCard label="Cash"><Dollar value={status?.portfolio.cash} /></MetricCard>
      <MetricCard label="Total P&L"><PnlBadge value={status?.portfolio.pnl_pct} /></MetricCard>
      <MetricCard label="vs SPY"><PnlBadge value={pnl - bench} /></MetricCard>
      <MetricCard label="Sharpe">
        <span className="text-slate-200">{perf?.sharpe_ratio?.toFixed(2) ?? "—"}</span>
      </MetricCard>
      <MetricCard label="Positions">
        <span className="text-slate-200">{status?.position_count ?? 0}</span>
      </MetricCard>
    </div>
  );
}

// ── Paper-vs-Backtest card (lives inside Reality-gap tab) ─────────

function PaperVsBacktestCard({
  perf,
  snapshotsLen,
}: {
  perf: PaperPerformance | null;
  snapshotsLen: number;
}) {
  const sharpe = perf?.sharpe_ratio ?? 0;
  const maxDd = perf?.max_drawdown_pct ?? 0;
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
        Paper vs Backtest
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">Sharpe</span>
          <span>
            <span className={`font-mono ${sharpe >= 0.82 ? "text-emerald-400" : "text-rose-400"}`}>
              {perf?.sharpe_ratio?.toFixed(2) ?? "—"}
            </span>
            <span className="mx-1 text-slate-600">/</span>
            <span className="text-slate-500">1.17</span>
            {sharpe >= 0.82 ? (
              <span className="ml-2 text-xs text-emerald-400">OK</span>
            ) : sharpe > 0 ? (
              <span className="ml-2 text-xs text-rose-400">BELOW 0.7x</span>
            ) : null}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Max DD</span>
          <span>
            <span className={`font-mono ${maxDd > -15 ? "text-emerald-400" : "text-rose-400"}`}>
              {perf?.max_drawdown_pct?.toFixed(1) ?? "—"}%
            </span>
            <span className="mx-1 text-slate-600">/</span>
            <span className="text-slate-500">-12.0%</span>
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Snapshots</span>
          <span className="font-mono text-slate-300">{snapshotsLen}</span>
        </div>
      </div>
    </div>
  );
}

// ── Risk Monitor card (lives inside Positions tab) ────────────────

function RiskMonitorCard({
  perf,
  positions,
  portfolio,
}: {
  perf: PaperPerformance | null;
  positions: PaperPosition[];
  portfolio: PaperPortfolio | null;
}) {
  const maxDd = perf?.max_drawdown_pct ?? 0;
  const navDenom = portfolio?.total_nav ?? 10000;
  const concentrations = positions.map(
    (p) => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / navDenom) * 100,
  );
  const maxPos = concentrations.length > 0 ? Math.max(...concentrations) : null;
  const concentrationHigh = maxPos != null && maxPos > 20;
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
        Risk Monitor
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-slate-400">Kill switch (-15%)</span>
          <span
            className={clsx(
              "rounded px-2 py-0.5 text-xs font-medium",
              maxDd > -10
                ? "bg-emerald-500/10 text-emerald-400"
                : maxDd > -13
                  ? "bg-amber-500/10 text-amber-400"
                  : "bg-rose-500/10 text-rose-400",
            )}
          >
            {maxDd > -10 ? "SAFE" : maxDd > -13 ? "WARNING" : "DANGER"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Max position</span>
          <span className="font-mono text-slate-300">
            {maxPos != null ? `${maxPos.toFixed(1)}%` : "—"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Concentration</span>
          <span
            className={clsx(
              "rounded px-2 py-0.5 text-xs font-medium",
              concentrationHigh
                ? "bg-amber-500/10 text-amber-400"
                : "bg-emerald-500/10 text-emerald-400",
            )}
          >
            {concentrationHigh ? "HIGH" : "OK"}
          </span>
        </div>
        <div className="mt-2">
          <div className="mb-1 flex justify-between text-xs text-slate-500">
            <span>Drawdown</span>
            <span>{perf?.max_drawdown_pct?.toFixed(1) ?? "0"}% / -15%</span>
          </div>
          <div className="h-2 rounded-full bg-navy-700">
            <div
              className={clsx(
                "h-2 rounded-full",
                maxDd > -10 ? "bg-emerald-500" : maxDd > -13 ? "bg-amber-500" : "bg-rose-500",
              )}
              style={{ width: `${Math.min(100, (Math.abs(maxDd) / 15) * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Scheduler details (collapsible Tier 3) ─────────────────────────

function SchedulerDetails({
  status,
  perf,
  isActive,
}: {
  status: PaperTradingStatus | null;
  perf: PaperPerformance | null;
  isActive: boolean | undefined;
}) {
  const isRunning = !!status?.loop.running;
  const lastRunLabel = status?.loop.last_run
    ? `Last run: ${new Date(status.loop.last_run).toLocaleString()}`
    : status?.next_run
      ? `Next run: ${new Date(status.next_run).toLocaleString()}`
      : "Never run";
  return (
    <details
      open={isRunning}
      className="mb-4 rounded-xl border border-slate-700/60 bg-[#080f1e]"
    >
      <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm">
        <span
          className={clsx(
            "h-2 w-2 rounded-full",
            isRunning
              ? "animate-pulse bg-sky-400"
              : isActive
                ? "bg-emerald-500"
                : "bg-amber-500",
          )}
        />
        <span className="font-medium text-slate-200">
          {isRunning ? "Cycle running" : isActive ? "Scheduler active" : "Scheduler paused"}
        </span>
        <span className="ml-auto font-mono text-xs text-slate-500">{lastRunLabel}</span>
      </summary>
      <div className="flex items-center gap-4 border-t border-slate-700/60 px-4 py-3 text-xs text-slate-400">
        <span>
          Days active:{" "}
          <span className="font-mono text-slate-300">{perf?.days_active ?? 0}</span>
        </span>
        <span>
          Total cost:{" "}
          <span className="font-mono text-slate-300">
            ${(perf?.total_analysis_cost ?? 0).toFixed(2)}
          </span>
        </span>
      </div>
    </details>
  );
}

// ── Tab definitions ───────────────────────────────────────────────

const TABS = [
  { id: "positions", label: "Positions" },
  { id: "trades", label: "Trades" },
  { id: "chart", label: "NAV Chart" },
  { id: "reality-gap", label: "Reality gap" },
  { id: "exit-quality", label: "Exit quality" },
] as const;

type TabId = (typeof TABS)[number]["id"];

// ── Page ──────────────────────────────────────────────────────────

export default function PaperTradingPage() {
  const [status, setStatus] = useState<PaperTradingStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PaperPortfolio | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [snapshots, setSnapshots] = useState<PaperSnapshot[]>([]);
  const [perf, setPerf] = useState<PaperPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("positions");
  const [actionLoading, setActionLoading] = useState(false);
  const [reconciliation, setReconciliation] = useState<PaperReconciliation | null>(null);
  const [reconciliationLoading, setReconciliationLoading] = useState(false);
  const [gate, setGate] = useState<GoLiveGate | null>(null);
  const [gateLoading, setGateLoading] = useState(false);
  const [gateError, setGateError] = useState<string | null>(null);
  const [rationaleTradeId, setRationaleTradeId] = useState<string | null>(null);

  const positionTickers = useMemo(() => positions.map((p) => p.ticker), [positions]);
  const { prices: livePrices } = useLivePrices(
    positionTickers,
    tab === "positions" && positions.length > 0,
  );

  const runNowIntervalRef = useRef<number | null>(null);
  const runNowTimeoutRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (runNowIntervalRef.current != null) window.clearInterval(runNowIntervalRef.current);
      if (runNowTimeoutRef.current != null) window.clearTimeout(runNowTimeoutRef.current);
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [s, portfolio_data, trade_data, snap_data, p] = await Promise.all([
        getPaperTradingStatus(),
        getPaperPortfolio().catch(() => null),
        getPaperTrades(),
        getPaperSnapshots(),
        getPaperPerformance().catch(() => null),
      ]);
      setStatus(s);
      if (portfolio_data) {
        setPortfolio(portfolio_data.portfolio);
        setPositions(portfolio_data.positions);
      }
      setTrades(trade_data.trades);
      setSnapshots(snap_data.snapshots);
      if (p) setPerf(p);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load paper trading data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (tab !== "reality-gap" || reconciliation) return;
    setReconciliationLoading(true);
    getPaperReconciliation()
      .then(setReconciliation)
      .catch(() => setReconciliation(null))
      .finally(() => setReconciliationLoading(false));
  }, [tab, reconciliation]);

  const loadGate = useCallback(() => {
    setGateLoading(true);
    setGateError(null);
    getPaperGate()
      .then((g) => {
        setGate(g);
        setGateError(null);
      })
      .catch((e: unknown) => {
        setGate(null);
        setGateError(e instanceof Error ? e.message : "gate failed");
      })
      .finally(() => setGateLoading(false));
  }, []);

  useEffect(() => {
    loadGate();
  }, [loadGate]);

  const handleStart = async () => {
    setActionLoading(true);
    try {
      await startPaperTrading();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start");
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    setActionLoading(true);
    try {
      await stopPaperTrading();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to stop");
    } finally {
      setActionLoading(false);
    }
  };

  const handleRunNow = async () => {
    setActionLoading(true);
    try {
      await triggerPaperTradingCycle();
      setError(null);
      const clearTimers = () => {
        if (runNowIntervalRef.current != null) {
          window.clearInterval(runNowIntervalRef.current);
          runNowIntervalRef.current = null;
        }
        if (runNowTimeoutRef.current != null) {
          window.clearTimeout(runNowTimeoutRef.current);
          runNowTimeoutRef.current = null;
        }
      };
      clearTimers(); // discard any stale timers from a previous click
      runNowIntervalRef.current = window.setInterval(async () => {
        try {
          const s = await getPaperTradingStatus();
          setStatus(s);
          if (!s.loop.running) {
            clearTimers();
            setActionLoading(false);
            await refresh();
          }
        } catch {
          /* keep retrying up to the 5-minute ceiling */
        }
      }, 10_000);
      runNowTimeoutRef.current = window.setTimeout(() => {
        clearTimers();
        setActionLoading(false);
      }, 300_000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to trigger cycle");
      setActionLoading(false);
    }
  };

  const isInitialized = status?.status !== "not_initialized";
  const isActive = status?.scheduler_active;
  const cycleRunning = !!status?.loop.running;

  // Chart data for the NAV Chart tab: reverse snapshots (oldest first).
  const chartData = [...snapshots].reverse().map((s) => ({
    date: s.snapshot_date,
    nav: s.total_nav,
    portfolio: s.cumulative_pnl_pct,
    benchmark: s.benchmark_pnl_pct,
    alpha: s.alpha_pct,
  }));

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      <main className="flex flex-1 flex-col overflow-hidden">
        {/* ── Fixed header zone (Tier 1 + Tier 5) ── */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Paper Trading</h2>
              <p className="text-sm text-slate-500">
                Autonomous AI-managed $10K virtual fund
              </p>
            </div>
            <div className="flex gap-2">
              {!isInitialized && (
                <button
                  type="button"
                  onClick={handleStart}
                  disabled={actionLoading}
                  className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
                >
                  {actionLoading ? "Starting..." : "Initialize Fund"}
                </button>
              )}
              {isInitialized && !isActive && (
                <button
                  type="button"
                  onClick={handleStart}
                  disabled={actionLoading}
                  className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  Start scheduler
                </button>
              )}
              {isActive && (
                <button
                  type="button"
                  onClick={handleStop}
                  disabled={actionLoading || cycleRunning}
                  title={cycleRunning ? "Cannot pause while a cycle is running" : "Pause the scheduler"}
                  className="rounded-lg bg-rose-600/80 px-4 py-2 text-sm font-medium text-white hover:bg-rose-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Pause
                </button>
              )}
              {isInitialized && (
                <button
                  type="button"
                  onClick={handleRunNow}
                  disabled={actionLoading || cycleRunning}
                  className="flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 hover:bg-sky-500/20 disabled:opacity-50"
                >
                  {(actionLoading || cycleRunning) && (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                  )}
                  {cycleRunning ? "Running cycle..." : actionLoading ? "Starting..." : "Run Now"}
                </button>
              )}
            </div>
          </div>

          {/* Tier 5: Tab bar (pinned, never scrolls) */}
          {isInitialized && (
            <div className="flex gap-1 rounded-lg bg-navy-800/50 p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(t.id)}
                  className={clsx(
                    "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                    tab === t.id
                      ? "bg-sky-500/10 text-sky-400"
                      : "text-slate-400 hover:text-slate-200",
                  )}
                >
                  {t.label}
                  {t.id === "positions" && ` (${positions.length})`}
                  {t.id === "trades" && ` (${trades.length})`}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Scrollable content zone (Tiers 2 / 3 / 4 / 6) ── */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {/* Tier 2: Error banner */}
          {error && (
            <div className="mb-6 rounded-lg border border-rose-500/30 bg-rose-950/50 p-4">
              <p className="text-sm font-medium text-rose-200">{error}</p>
              {error.includes("Cannot reach") && (
                <p className="mt-1 text-xs text-rose-300/60">
                  Make sure the backend is running:{" "}
                  <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">
                    uvicorn backend.main:app --port 8000
                  </code>
                </p>
              )}
              <button
                type="button"
                onClick={() => {
                  setError(null);
                  setLoading(true);
                  refresh();
                }}
                className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60"
              >
                Retry
              </button>
            </div>
          )}

          {loading ? (
            <PageSkeleton />
          ) : !isInitialized ? (
            <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-12 text-center">
              <p className="mb-2 text-lg text-slate-300">No paper portfolio initialized</p>
              <p className="text-sm text-slate-500">
                Click &quot;Initialize Fund&quot; to start with $10,000 virtual capital.
              </p>
            </div>
          ) : (
            <>
              {/* Ops-strip bento: Go-Live Gate (tall) on the left, two
                  short widgets stacked on the right. items-start so short
                  widgets don't stretch to fill the tall column's height. */}
              <div className="mb-4 grid grid-cols-1 items-start gap-3 lg:grid-cols-2">
                <GoLiveGateWidget
                  gate={gate}
                  loading={gateLoading}
                  error={gateError}
                  onRetry={loadGate}
                />
                <div className="flex flex-col gap-3">
                  <KillSwitchPanel />
                  <CycleHealthStrip />
                  <SchedulerDetails status={status} perf={perf} isActive={isActive} />
                </div>
              </div>

              {/* Tier 4: Global KPI hero (portfolio-level metrics, always visible) */}
              <SummaryHero status={status} perf={perf} />

              {/* Tier 6: Tab content */}
              {tab === "positions" && (
                <div className="space-y-4">
                  <RiskMonitorCard perf={perf} positions={positions} portfolio={portfolio} />

                  <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-navy-700 text-xs uppercase text-slate-500">
                            <th className="px-4 py-3">Ticker</th>
                            <th className="px-4 py-3">Qty</th>
                            <th className="px-4 py-3">Entry</th>
                            <th className="px-4 py-3">Current</th>
                            <th className="px-4 py-3">Market Value</th>
                            <th className="px-4 py-3">P&amp;L</th>
                            <th className="px-4 py-3">Stop Loss</th>
                            <th className="px-4 py-3">Days Held</th>
                          </tr>
                        </thead>
                        <tbody>
                          {positions.length === 0 ? (
                            <tr>
                              <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                                No open positions
                              </td>
                            </tr>
                          ) : (
                            positions.map((pos) => {
                              const daysHeld = pos.entry_date
                                ? Math.floor(
                                    (Date.now() - new Date(pos.entry_date).getTime()) / 86_400_000,
                                  )
                                : 0;
                              const live = livePrices[pos.ticker];
                              const shown = live?.price ?? pos.current_price;
                              const ageLabel =
                                live?.age_sec != null ? `${Math.round(live.age_sec)}s` : null;
                              return (
                                <tr
                                  key={pos.position_id}
                                  className="border-b border-navy-700/50 text-slate-300 hover:bg-navy-700/30"
                                >
                                  <td className="px-4 py-3 font-mono font-semibold text-slate-100">
                                    {pos.ticker}
                                  </td>
                                  <td className="px-4 py-3">{pos.quantity.toFixed(2)}</td>
                                  <td className="px-4 py-3">${pos.avg_entry_price.toFixed(2)}</td>
                                  <td className="px-4 py-3">
                                    {shown == null ? (
                                      "—"
                                    ) : (
                                      <span>
                                        ${shown.toFixed(2)}
                                        {ageLabel && (
                                          <span
                                            className="ml-1 font-mono text-[10px] text-slate-500"
                                            title="Seconds since last yfinance fetch"
                                          >
                                            ({ageLabel})
                                          </span>
                                        )}
                                      </span>
                                    )}
                                  </td>
                                  <td className="px-4 py-3">
                                    <Dollar value={pos.market_value} />
                                  </td>
                                  <td className="px-4 py-3">
                                    <PnlBadge value={pos.unrealized_pnl_pct} />
                                  </td>
                                  <td className="px-4 py-3 text-slate-500">
                                    {pos.stop_loss_price != null
                                      ? `$${pos.stop_loss_price.toFixed(2)}`
                                      : "—"}
                                  </td>
                                  <td className="px-4 py-3 text-slate-500">{daysHeld}d</td>
                                </tr>
                              );
                            })
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {tab === "trades" && (
                <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-navy-700 text-xs uppercase text-slate-500">
                          <th className="px-4 py-3">Date</th>
                          <th className="px-4 py-3">Action</th>
                          <th className="px-4 py-3">Ticker</th>
                          <th className="px-4 py-3">Qty</th>
                          <th className="px-4 py-3">Price</th>
                          <th className="px-4 py-3">Value</th>
                          <th className="px-4 py-3">Fee</th>
                          <th className="px-4 py-3">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trades.length === 0 ? (
                          <tr>
                            <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                              No trades yet
                            </td>
                          </tr>
                        ) : (
                          trades.map((t) => (
                            <tr
                              key={t.trade_id}
                              onClick={() => setRationaleTradeId(t.trade_id)}
                              className="cursor-pointer border-b border-navy-700/50 text-slate-300 hover:bg-navy-700/30"
                              title="Click to view agent rationale"
                            >
                              <td className="px-4 py-3 text-xs text-slate-500">
                                {new Date(t.created_at).toLocaleDateString()}
                              </td>
                              <td className="px-4 py-3">
                                <span
                                  className={clsx(
                                    "rounded px-2 py-0.5 text-xs font-medium",
                                    t.action === "BUY"
                                      ? "bg-emerald-500/10 text-emerald-400"
                                      : "bg-rose-500/10 text-rose-400",
                                  )}
                                >
                                  {t.action}
                                </span>
                              </td>
                              <td className="px-4 py-3 font-mono font-semibold text-slate-100">
                                {t.ticker}
                              </td>
                              <td className="px-4 py-3">{t.quantity.toFixed(2)}</td>
                              <td className="px-4 py-3">${t.price.toFixed(2)}</td>
                              <td className="px-4 py-3">
                                <Dollar value={t.total_value} />
                              </td>
                              <td className="px-4 py-3 text-slate-500">
                                {t.transaction_cost != null
                                  ? `$${t.transaction_cost.toFixed(2)}`
                                  : "—"}
                              </td>
                              <td className="px-4 py-3">
                                <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                                  {t.reason ?? "—"}
                                </span>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {tab === "chart" && (
                <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
                  {chartData.length < 2 ? (
                    <p className="py-8 text-center text-slate-500">
                      Need at least 2 days of data for charting.
                    </p>
                  ) : (
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis
                          dataKey="date"
                          tick={{ fill: "#64748b", fontSize: 11 }}
                          tickFormatter={(d: string) => d.slice(5)}
                        />
                        <YAxis
                          tick={{ fill: "#64748b", fontSize: 11 }}
                          tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0f172a",
                            border: "1px solid #334155",
                            borderRadius: 8,
                          }}
                          labelStyle={{ color: "#94a3b8" }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="portfolio"
                          name="Portfolio"
                          stroke="#0ea5e9"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="benchmark"
                          name="SPY"
                          stroke="#64748b"
                          strokeWidth={1.5}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="alpha"
                          name="Alpha"
                          stroke="#22c55e"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              )}

              {tab === "reality-gap" && (
                <div className="space-y-4">
                  <PaperVsBacktestCard perf={perf} snapshotsLen={snapshots.length} />
                  <PaperReconciliationChart
                    reconciliation={reconciliation}
                    loading={reconciliationLoading}
                  />
                </div>
              )}

              {tab === "exit-quality" && <MfeMaeScatter />}
            </>
          )}

          <AgentRationaleDrawer
            tradeId={rationaleTradeId}
            onClose={() => setRationaleTradeId(null)}
          />
        </div>
      </main>
    </div>
  );
}
