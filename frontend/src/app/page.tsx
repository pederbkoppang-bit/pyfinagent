"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import {
  NavAnalyze, NavSignals, NavBacktest,
} from "@/lib/icons";
import { listReports, getPaperTradingStatus, getPaperPortfolio } from "@/lib/api";
import type { ReportSummary, PaperTradingStatus, PaperPosition } from "@/lib/types";

/* ── Recommendation badge color ── */
function recColor(rec: string) {
  const r = rec?.toUpperCase() ?? "";
  if (r.includes("STRONG_BUY") || r.includes("STRONG BUY")) return "bg-emerald-500/20 text-emerald-400";
  if (r.includes("BUY")) return "bg-emerald-500/15 text-emerald-400";
  if (r.includes("STRONG_SELL") || r.includes("STRONG SELL")) return "bg-rose-500/20 text-rose-400";
  if (r.includes("SELL")) return "bg-rose-500/15 text-rose-400";
  return "bg-amber-500/15 text-amber-400";
}

function fmtPct(v: number | null | undefined) {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function fmtUsd(v: number | null | undefined) {
  if (v == null) return "—";
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export default function HomePage() {
  const router = useRouter();
  const [ticker, setTicker] = useState("");
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [ptStatus, setPtStatus] = useState<PaperTradingStatus | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [reps, status, portfolio] = await Promise.allSettled([
        listReports(5),
        getPaperTradingStatus(),
        getPaperPortfolio(),
      ]);
      if (cancelled) return;
      if (reps.status === "fulfilled") setReports(reps.value);
      if (status.status === "fulfilled") setPtStatus(status.value);
      if (portfolio.status === "fulfilled") setPositions(portfolio.value.positions ?? []);

      // If ALL calls failed, show an error
      const allFailed = reps.status === "rejected" && status.status === "rejected" && portfolio.status === "rejected";
      if (allFailed) {
        const reason = reps.reason instanceof Error ? reps.reason.message : "Cannot reach backend.";
        setLoadError(reason);
      }
      setLoaded(true);
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const nav = ptStatus?.portfolio;
  const navValue = nav?.nav;
  const pnl = nav?.pnl_pct;
  const benchmark = nav?.benchmark_return_pct;
  const alpha = pnl != null && benchmark != null ? pnl - benchmark : null;

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
        {/* Page header */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-100">Home</h2>
          <p className="text-sm text-slate-500">Portfolio overview &amp; quick actions</p>
        </div>

        {loadError && (
          <div className="mb-6 rounded-lg border border-rose-900 bg-rose-950/50 p-4">
            <p className="text-sm font-medium text-rose-200">{loadError}</p>
            {loadError.includes("Cannot reach") && (
              <p className="mt-1 text-xs text-rose-300/60">
                Run: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
              </p>
            )}
          </div>
        )}

        {/* ─── Portfolio Snapshot (hero) ─── */}
        <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">NAV</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {loaded ? fmtUsd(navValue) : "—"}
            </p>
          </div>
          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">P&amp;L</p>
            <p className={`mt-1 text-2xl font-bold ${pnl != null && pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {loaded ? fmtPct(pnl) : "—"}
            </p>
          </div>
          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">vs SPY</p>
            <p className={`mt-1 text-2xl font-bold ${alpha != null && alpha >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {loaded ? fmtPct(alpha) : "—"}
            </p>
          </div>
          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Positions</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {loaded ? (positions.length || 0) : "—"}
            </p>
          </div>
        </div>

        {/* ─── Recent Reports ─── */}
        <div className="mb-8">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Recent Reports</h3>
            <Link href="/reports" className="text-xs text-sky-400 hover:text-sky-300">View all</Link>
          </div>
          <div className="overflow-hidden rounded-xl border border-navy-700">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-navy-700 bg-navy-800/80">
                <tr>
                  <th className="px-4 py-3 font-medium text-slate-400">Ticker</th>
                  <th className="px-4 py-3 font-medium text-slate-400">Date</th>
                  <th className="px-4 py-3 font-medium text-slate-400">Score</th>
                  <th className="px-4 py-3 font-medium text-slate-400">Recommendation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/50">
                {loaded && reports.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-4 py-8 text-center text-slate-600">
                      No reports yet. Run your first analysis.
                    </td>
                  </tr>
                )}
                {reports.map((r) => (
                  <tr
                    key={`${r.ticker}-${r.analysis_date}`}
                    className="cursor-pointer bg-navy-800/40 transition-colors hover:bg-navy-700/40"
                    onClick={() => router.push(`/reports?ticker=${encodeURIComponent(r.ticker)}`)}
                  >
                    <td className="px-4 py-3 font-mono font-medium text-slate-200">{r.ticker}</td>
                    <td className="px-4 py-3 text-slate-400">{r.analysis_date?.slice(0, 10)}</td>
                    <td className="px-4 py-3">
                      <span className="font-semibold text-slate-200">{r.final_score?.toFixed(1) ?? "—"}</span>
                      <span className="text-slate-500">/10</span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-block rounded-md px-2.5 py-1 text-xs font-medium ${recColor(r.recommendation)}`}>
                        {r.recommendation?.replace(/_/g, " ") ?? "—"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ─── Quick Actions ─── */}
        <div>
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">Quick Actions</h3>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {/* Run Analysis */}
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
              <div className="mb-3 flex items-center gap-2">
                <NavAnalyze size={20} weight="duotone" className="text-sky-400" />
                <span className="text-sm font-medium text-slate-200">Run Analysis</span>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="TICKER"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && ticker.trim()) {
                      router.push(`/analyze?ticker=${encodeURIComponent(ticker.trim())}`);
                    }
                  }}
                  className="w-28 rounded-lg border border-navy-700 bg-navy-900 px-3 py-1.5 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
                />
                <button
                  onClick={() => {
                    if (ticker.trim()) router.push(`/analyze?ticker=${encodeURIComponent(ticker.trim())}`);
                  }}
                  className="rounded-lg bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500"
                >
                  Go
                </button>
              </div>
            </div>

            {/* View Signals */}
            <Link
              href="/signals"
              className="group rounded-xl border border-navy-700 bg-navy-800/60 p-5 transition-colors hover:border-sky-600/30"
            >
              <div className="mb-2 flex items-center gap-2">
                <NavSignals size={20} weight="duotone" className="text-sky-400" />
                <span className="text-sm font-medium text-slate-200">View Signals</span>
              </div>
              <p className="text-xs text-slate-500">Explore 11 enrichment signals for any ticker</p>
            </Link>

            {/* Run Backtest */}
            <Link
              href="/backtest"
              className="group rounded-xl border border-navy-700 bg-navy-800/60 p-5 transition-colors hover:border-sky-600/30"
            >
              <div className="mb-2 flex items-center gap-2">
                <NavBacktest size={20} weight="duotone" className="text-sky-400" />
                <span className="text-sm font-medium text-slate-200">Run Backtest</span>
              </div>
              <p className="text-xs text-slate-500">Walk-forward backtesting with Triple Barrier labels</p>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
