"use client";

import { useCallback, useEffect, useState } from "react";
import { clsx } from "clsx";
import { Sidebar } from "@/components/Sidebar";
import { PageSkeleton } from "@/components/Skeleton";
import {
  getPaperTradingStatus,
  getPaperPortfolio,
  getPaperTrades,
  getPaperSnapshots,
  getPaperPerformance,
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
} from "@/lib/types";
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

export default function PaperTradingPage() {
  const [status, setStatus] = useState<PaperTradingStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PaperPortfolio | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [snapshots, setSnapshots] = useState<PaperSnapshot[]>([]);
  const [perf, setPerf] = useState<PaperPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"positions" | "trades" | "chart">("positions");
  const [actionLoading, setActionLoading] = useState(false);

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
      // Lightweight status-only poll every 10s, full refresh after loop completes
      const interval = setInterval(async () => {
        try {
          const s = await getPaperTradingStatus();
          setStatus(s);
          setActionLoading(false);
          if (!s.loop.running) {
            clearInterval(interval);
            await refresh();
          }
        } catch { /* swallow poll errors */ }
      }, 10000);
      setTimeout(() => clearInterval(interval), 300000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to trigger cycle");
      setActionLoading(false);
    }
  };

  const isInitialized = status?.status !== "not_initialized";
  const isActive = status?.scheduler_active;

  // Chart data: reverse snapshots (oldest first)
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

      <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
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
                onClick={handleStart}
                disabled={actionLoading}
                className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
              >
                {actionLoading ? "Starting..." : "Initialize Fund"}
              </button>
            )}
            {isInitialized && !isActive && (
              <button
                onClick={handleStart}
                disabled={actionLoading}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
              >
                Start Scheduler
              </button>
            )}
            {isActive && (
              <button
                onClick={handleStop}
                disabled={actionLoading}
                className="rounded-lg bg-rose-600/80 px-4 py-2 text-sm font-medium text-white hover:bg-rose-500 disabled:opacity-50"
              >
                Pause
              </button>
            )}
            {isInitialized && (
              <button
                onClick={handleRunNow}
                disabled={actionLoading || status?.loop.running}
                className="flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 hover:bg-sky-500/20 disabled:opacity-50"
              >
                {(actionLoading || status?.loop.running) && (
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                )}
                {status?.loop.running ? "Running cycle..." : actionLoading ? "Starting..." : "Run Now"}
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-rose-900 bg-rose-950/50 p-4">
            <p className="text-sm font-medium text-rose-200">{error}</p>
            {error.includes("Cannot reach") && (
              <p className="mt-1 text-xs text-rose-300/60">
                Make sure the backend is running: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
              </p>
            )}
            <button onClick={() => { setError(null); setLoading(true); refresh(); }} className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60">
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <PageSkeleton />
        ) : !isInitialized ? (
          <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-12 text-center">
            <p className="text-lg text-slate-300 mb-2">No Paper Portfolio Initialized</p>
            <p className="text-sm text-slate-500">
              Click &quot;Initialize Fund&quot; to start with $10,000 virtual capital
            </p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-6">
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">NAV</p>
                <p className="mt-1 text-lg font-semibold">
                  <Dollar value={status?.portfolio.nav} />
                </p>
              </div>
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">Cash</p>
                <p className="mt-1 text-lg font-semibold">
                  <Dollar value={status?.portfolio.cash} />
                </p>
              </div>
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">Total P&amp;L</p>
                <p className="mt-1 text-lg font-semibold">
                  <PnlBadge value={status?.portfolio.pnl_pct} />
                </p>
              </div>
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">vs SPY</p>
                <p className="mt-1 text-lg font-semibold">
                  <PnlBadge value={(status?.portfolio.pnl_pct ?? 0) - (status?.portfolio.benchmark_return_pct ?? 0)} />
                </p>
              </div>
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">Sharpe</p>
                <p className="mt-1 text-lg font-semibold text-slate-200">
                  {perf?.sharpe_ratio?.toFixed(2) ?? "—"}
                </p>
              </div>
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <p className="text-xs text-slate-500">Positions</p>
                <p className="mt-1 text-lg font-semibold text-slate-200">
                  {status?.position_count ?? 0}
                </p>
              </div>
            </div>

            {/* Backtest Comparison + Risk Monitor */}
            <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2">
              {/* Paper vs Backtest */}
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">Paper vs Backtest</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sharpe</span>
                    <span>
                      <span className={`font-mono ${(perf?.sharpe_ratio ?? 0) >= 0.82 ? "text-emerald-400" : "text-rose-400"}`}>
                        {perf?.sharpe_ratio?.toFixed(2) ?? "—"}
                      </span>
                      <span className="text-slate-600 mx-1">/</span>
                      <span className="text-slate-500">1.17</span>
                      {(perf?.sharpe_ratio ?? 0) >= 0.82 ? (
                        <span className="ml-2 text-xs text-emerald-400">OK</span>
                      ) : (perf?.sharpe_ratio ?? 0) > 0 ? (
                        <span className="ml-2 text-xs text-rose-400">BELOW 0.7x</span>
                      ) : null}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Max DD</span>
                    <span>
                      <span className={`font-mono ${(perf?.max_drawdown_pct ?? 0) > -15 ? "text-emerald-400" : "text-rose-400"}`}>
                        {perf?.max_drawdown_pct?.toFixed(1) ?? "—"}%
                      </span>
                      <span className="text-slate-600 mx-1">/</span>
                      <span className="text-slate-500">-12.0%</span>
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Snapshots</span>
                    <span className="font-mono text-slate-300">{snapshots.length}</span>
                  </div>
                </div>
              </div>

              {/* Kill Switch Status */}
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
                <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">Risk Monitor</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Kill Switch (-15%)</span>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                      (perf?.max_drawdown_pct ?? 0) > -10 ? "bg-emerald-500/10 text-emerald-400" :
                      (perf?.max_drawdown_pct ?? 0) > -13 ? "bg-amber-500/10 text-amber-400" :
                      "bg-rose-500/10 text-rose-400"
                    }`}>
                      {(perf?.max_drawdown_pct ?? 0) > -10 ? "SAFE" :
                       (perf?.max_drawdown_pct ?? 0) > -13 ? "WARNING" : "DANGER"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Max Position</span>
                    <span className="font-mono text-slate-300">
                      {positions.length > 0 && portfolio
                        ? `${Math.max(...positions.map(p => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / (portfolio.total_nav ?? 10000)) * 100)).toFixed(1)}%`
                        : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Concentration</span>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                      positions.length > 0 && portfolio &&
                      Math.max(...positions.map(p => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / (portfolio.total_nav ?? 10000)) * 100)) > 20
                        ? "bg-amber-500/10 text-amber-400" : "bg-emerald-500/10 text-emerald-400"
                    }`}>
                      {positions.length > 0 && portfolio &&
                       Math.max(...positions.map(p => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / (portfolio.total_nav ?? 10000)) * 100)) > 20
                        ? "HIGH" : "OK"}
                    </span>
                  </div>
                  <div className="mt-2">
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Drawdown</span>
                      <span>{perf?.max_drawdown_pct?.toFixed(1) ?? "0"}% / -15%</span>
                    </div>
                    <div className="h-2 rounded-full bg-navy-700">
                      <div
                        className={`h-2 rounded-full ${
                          (perf?.max_drawdown_pct ?? 0) > -10 ? "bg-emerald-500" :
                          (perf?.max_drawdown_pct ?? 0) > -13 ? "bg-amber-500" : "bg-rose-500"
                        }`}
                        style={{ width: `${Math.min(100, Math.abs(perf?.max_drawdown_pct ?? 0) / 15 * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Status Banner */}
            <div className="mb-6 flex items-center gap-3 rounded-lg border border-navy-700 bg-navy-800/50 px-4 py-3">
              <div className={clsx(
                "h-2.5 w-2.5 rounded-full",
                isActive ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              )} />
              <span className="text-sm text-slate-300">
                {isActive ? "Scheduler active" : "Scheduler paused"} —{" "}
                {status?.loop.last_run
                  ? `Last run: ${new Date(status.loop.last_run).toLocaleString()}`
                  : status?.next_run
                    ? `Next run: ${new Date(status.next_run).toLocaleString()}`
                    : "Never run"}
              </span>
              {perf && (
                <span className="ml-auto text-xs text-slate-500">
                  Total cost: ${perf.total_analysis_cost.toFixed(2)} | Days: {perf.days_active}
                </span>
              )}
            </div>

            {/* Tabs */}
            <div className="mb-4 flex gap-1 rounded-lg bg-navy-800/50 p-1">
              {["positions", "trades", "chart"].map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t as typeof tab)}
                  className={clsx(
                    "flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                    tab === t
                      ? "bg-sky-500/10 text-sky-400"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  {t === "positions" && `Positions (${positions.length})`}
                  {t === "trades" && `Trades (${trades.length})`}
                  {t === "chart" && "NAV Chart"}
                </button>
              ))}
            </div>

            {/* Positions Tab */}
            {tab === "positions" && (
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
                            ? Math.floor((Date.now() - new Date(pos.entry_date).getTime()) / 86400000)
                            : 0;
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
                                {pos.current_price != null ? `$${pos.current_price.toFixed(2)}` : "—"}
                              </td>
                              <td className="px-4 py-3">
                                <Dollar value={pos.market_value} />
                              </td>
                              <td className="px-4 py-3">
                                <PnlBadge value={pos.unrealized_pnl_pct} />
                              </td>
                              <td className="px-4 py-3 text-slate-500">
                                {pos.stop_loss_price != null ? `$${pos.stop_loss_price.toFixed(2)}` : "—"}
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
            )}

            {/* Trades Tab */}
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
                            className="border-b border-navy-700/50 text-slate-300 hover:bg-navy-700/30"
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
                                    : "bg-rose-500/10 text-rose-400"
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
                              {t.transaction_cost != null ? `$${t.transaction_cost.toFixed(2)}` : "—"}
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

            {/* NAV Chart Tab */}
            {tab === "chart" && (
              <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
                {chartData.length < 2 ? (
                  <p className="text-center text-slate-500 py-8">
                    Need at least 2 days of data for charting
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
          </>
        )}
      </main>
    </div>
  );
}
