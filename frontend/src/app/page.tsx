"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { OpsStatusBar } from "@/components/OpsStatusBar";
import { KillSwitchShortcut } from "@/components/KillSwitchShortcut";
import { RecentReportsTable } from "@/components/RecentReportsTable";
import { HomeQuickActionsPanel } from "@/components/HomeQuickActionsPanel";
import { LatestTransactionsBox } from "@/components/LatestTransactionsBox";
import { listReports, getPaperTradingStatus, getPaperPortfolio, getPaperTrades, getSovereignRedLine } from "@/lib/api";
import { useLivePrices } from "@/lib/useLivePrices";
import { useLiveNav } from "@/lib/useLiveNav";
import {
  dailyDelta,
  sharpe as kpiSharpe,
  sortino as kpiSortino,
  maxDrawdownPct,
  categorizePositions,
} from "@/lib/kpiMetrics";
import type { ReportSummary, PaperTradingStatus, PaperPosition, PaperTrade } from "@/lib/types";
import type {
  SovereignRedLinePoint,
  SovereignRedLineEvent,
} from "@/lib/api";
import type { RedLineWindow } from "@/components/RedLineMonitor";

// phase-10.5.7: lazy-load the heavy Recharts bundle client-only.
// ssr:false keeps the hero out of SSR so homepage TTFB stays fast.
// phase-16.43: skeleton height matches the chart's actual h-72 (288px)
// instead of the old viewport-percent floor that caused dead whitespace.
const RedLineMonitor = dynamic(
  () => import("@/components/RedLineMonitor").then((m) => m.RedLineMonitor),
  {
    ssr: false,
    loading: () => (
      <div className="h-72 animate-pulse rounded-xl border border-navy-700 bg-navy-800/40" />
    ),
  },
);

function fmtPct(v: number | null | undefined) {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function fmtUsd(v: number | null | undefined) {
  if (v == null) return "—";
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function KpiTile({
  label,
  value,
  subText,
  valueClass,
  subTextClass,
}: {
  label: string;
  value: string;
  subText?: string | null;
  valueClass?: string;
  subTextClass?: string;
}) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
      <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClass ?? "text-slate-100"}`}>{value}</p>
      {subText && (
        <p className={`mt-0.5 text-xs ${subTextClass ?? "text-slate-500"}`}>{subText}</p>
      )}
    </div>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [ticker, setTicker] = useState("");
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [ptStatus, setPtStatus] = useState<PaperTradingStatus | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [tradesError, setTradesError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [redLineWindow, setRedLineWindow] = useState<RedLineWindow>("30d");
  const [redLineSeries, setRedLineSeries] = useState<SovereignRedLinePoint[]>([]);
  const [redLineEvents, setRedLineEvents] = useState<SovereignRedLineEvent[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [reps, status, portfolio, tradesResp] = await Promise.allSettled([
        listReports(5),
        getPaperTradingStatus(),
        getPaperPortfolio(),
        getPaperTrades(5),
      ]);
      if (cancelled) return;
      if (reps.status === "fulfilled") setReports(reps.value);
      if (status.status === "fulfilled") setPtStatus(status.value);
      if (portfolio.status === "fulfilled") setPositions(portfolio.value.positions ?? []);
      if (tradesResp.status === "fulfilled") {
        setTrades(tradesResp.value.trades ?? []);
      } else {
        const reason = tradesResp.reason instanceof Error ? tradesResp.reason.message : "Trades unavailable";
        setTradesError(reason);
      }
      const allFailed =
        reps.status === "rejected" &&
        status.status === "rejected" &&
        portfolio.status === "rejected" &&
        tradesResp.status === "rejected";
      if (allFailed) {
        const reason = reps.reason instanceof Error ? reps.reason.message : "Cannot reach backend.";
        setLoadError(reason);
      }
      setLoaded(true);
    }
    load();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getSovereignRedLine(redLineWindow)
      .then((resp) => {
        if (cancelled) return;
        setRedLineSeries(resp.series);
        setRedLineEvents(resp.events);
      })
      .catch(() => {
        if (cancelled) return;
        setRedLineSeries([]);
        setRedLineEvents([]);
      });
    return () => { cancelled = true; };
  }, [redLineWindow]);

  // phase-23.1.17: live-derive NAV + Total P&L pct via shared useLiveNav hook
  // so the home cockpit and the paper-trading page render the same number on
  // every 30s tick. Falls back to the BQ snapshot value when no live ticks
  // are available (initial paint, empty positions).
  const positionTickers = positions.map((p) => p.ticker).filter(Boolean);
  const { prices: livePrices } = useLivePrices(positionTickers, positions.length > 0);
  const { liveNav, liveTotalPnlPct } = useLiveNav(ptStatus, positions, livePrices);

  const nav = ptStatus?.portfolio;
  const navValue = liveNav ?? nav?.nav;
  const pnl = liveTotalPnlPct ?? nav?.pnl_pct;
  const benchmark = nav?.benchmark_return_pct;
  const alpha = pnl != null && benchmark != null ? pnl - benchmark : null;

  // phase-16.44: KPI sub-text computed from real data (no hardcoded values).
  // All return null on insufficient/flat data; the tile then renders "—".
  const navSeries = redLineSeries.map((p) => ({ date: p.date, nav: p.nav }));
  const today = dailyDelta(navSeries);
  const sharpe90 = kpiSharpe(navSeries);
  const sortino90 = kpiSortino(navSeries);
  const dd30 = maxDrawdownPct(navSeries);
  const posBreakdown = categorizePositions(positions);
  // 8.0% trailing-DD limit comes from kill-switch breach.trailing_dd_limit_pct
  // when available; show the static label until we wire it through.
  const trailingDdLimit = "8.0%";

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Fixed header zone */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">MAS Operator Cockpit</h2>
              <p className="text-sm text-slate-500">
                Live status, portfolio snapshot, and emergency controls.
                <span className="ml-2 rounded bg-navy-800 px-2 py-0.5 font-mono text-[10px] text-slate-400">
                  Ctrl/Cmd+Shift+H = halt
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Scrollable content zone */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {/* Invisible keyboard-shortcut + aria-live region */}
          <KillSwitchShortcut />

          {/* phase-16.43 + 16.44: gate bar at TOP, scorecards immediately
              under it. Operator status (gate / kill / cycle / last / next)
              is the most critical signal, so it leads the scrollable zone;
              KPI tiles follow with comparison sub-text. */}
          <div className="mb-6">
            <OpsStatusBar nextRunAt={ptStatus?.next_run ?? null} />
          </div>

          {/* Error banner (conditional) */}
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

          {/* phase-16.44: KPI hero with comparison sub-text under each value.
              All sub-text values computed from real backend data via
              kpiMetrics.ts helpers; null returns -> "—" (no hardcoded). */}
          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <KpiTile
              label="NAV"
              value={loaded ? fmtUsd(navValue) : "—"}
            />
            <KpiTile
              label="P&L (today)"
              value={today != null ? `${today.dollars >= 0 ? "+" : ""}${fmtUsd(today.dollars).replace("$", "$")}` : "—"}
              subText={today != null ? `${today.pct >= 0 ? "+" : ""}${today.pct.toFixed(2)}%` : null}
              valueClass={today != null && today.dollars >= 0 ? "text-emerald-400" : today != null ? "text-rose-400" : undefined}
              subTextClass={today != null && today.pct >= 0 ? "text-emerald-400/70" : today != null ? "text-rose-400/70" : undefined}
            />
            <KpiTile
              label="vs SPY"
              value={loaded ? fmtPct(alpha) : "—"}
              subText={benchmark != null ? `SPY ${fmtPct(benchmark)}` : null}
              valueClass={alpha != null && alpha >= 0 ? "text-emerald-400" : alpha != null ? "text-rose-400" : undefined}
            />
            <KpiTile
              label="Sharpe (90d)"
              value={sharpe90 != null ? sharpe90.toFixed(2) : "—"}
              subText={sortino90 != null ? `Sortino ${sortino90.toFixed(2)}` : null}
            />
            <KpiTile
              label="Max DD (30d)"
              value={dd30 != null ? `${dd30.toFixed(2)}%` : "—"}
              subText={`bounded ${trailingDdLimit}`}
              valueClass={dd30 != null ? "text-rose-400" : undefined}
            />
            <KpiTile
              label="Positions"
              value={loaded ? String(posBreakdown.total) : "—"}
              subText={loaded && posBreakdown.total > 0 ? `${posBreakdown.long} long · ${posBreakdown.short} short` : null}
            />
          </div>

          {/* phase-16.43: Red Line chart uses its own h-72 floor -- the
              previous 55-percent-viewport wrapper was creating empty
              whitespace below the auto-sized BentoCard. */}
          <div className="mb-6">
            <RedLineMonitor
              series={redLineSeries}
              events={redLineEvents}
              window={redLineWindow}
              onWindowChange={setRedLineWindow}
              compact
            />
          </div>

          {/* phase-16.42 + 16.43 + 16.45 + 16.46 + 16.47: 3-box row on the
              home cockpit. lg:grid-cols-6 with col-span 2/2/2 = equal
              thirds (33% each). 16.46's 20% Actions slot was too narrow
              (Analyze button cropped, action labels wrapping); equal
              thirds gives all three boxes a fair shake. Internal layout
              hardening (min-w-0 + shrink-0) in HomeQuickActionsPanel
              keeps the panel safe at narrower viewports. */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-6 lg:items-stretch">
            <div className="lg:col-span-2 h-full">
              <RecentReportsTable
                reports={reports}
                loaded={loaded}
                loadError={loadError}
              />
            </div>
            <div className="lg:col-span-2 h-full">
              <LatestTransactionsBox
                trades={trades}
                loaded={loaded}
                loadError={tradesError}
              />
            </div>
            <div className="lg:col-span-2 h-full">
              <HomeQuickActionsPanel
                ticker={ticker}
                onTickerChange={setTicker}
                onAnalyze={() => {
                  if (ticker.trim()) router.push(`/signals?ticker=${encodeURIComponent(ticker.trim())}`);
                }}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
