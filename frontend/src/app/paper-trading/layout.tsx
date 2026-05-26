"use client";

// phase-44.2 -- cockpit shared shell + ARIA tablist.
//
// Hosts:
// - Sidebar + page-shell two-zone flex (per frontend-layout.md Section 1).
// - Page header (Tier 1) with action buttons (Initialize / Start / Pause / Run Now).
// - OpsStatusBar (Tier 4) + SummaryHero (Tier 4) above the tab bar.
// - Link-based tablist (Tier 5) implementing W3C WAI-ARIA APG tabs pattern:
//   role="tablist" container, role="tab" + aria-selected + aria-controls per
//   Link, roving tabindex + ArrowLeft/Right/Home/End keyboard nav.
// - PaperTradingDataContext provider so sub-routes consume shared state
//   without prop-drilling.
// - AgentRationaleDrawer mounted once, opens on row click from any sub-route.
//
// MANAGE_REMOVAL_DEFERRED -- Manage tab stays as the 6th tab pending
// operator_approval_44.2.md per research brief topic 5 + risk flag P-4.

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { clsx } from "clsx";
import { Sidebar } from "@/components/Sidebar";
import { OpsStatusBar } from "@/components/OpsStatusBar";
import { AgentRationaleDrawer } from "@/components/AgentRationaleDrawer";
import { PageSkeleton } from "@/components/Skeleton";
import { SummaryHero } from "@/components/paper-trading/cockpit-helpers";
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
  TabPositions,
  TabTrades,
  TabNavChart,
  TabRealityGap,
  TabExitQuality,
  Gear,
  type Icon,
} from "@/lib/icons";
import { useLivePrices } from "@/lib/useLivePrices";
import { useLiveNav } from "@/lib/useLiveNav";
import { useTickerMeta } from "@/lib/useTickerMeta";
import {
  PaperTradingDataContext,
  type PaperTradingDataValue,
} from "@/lib/paper-trading-context";

// ── Tab registry (sub-route slugs + icons + labels) ───────────────

interface TabSpec {
  slug: string;
  href: string;
  label: string;
  icon: Icon;
  // Optional dynamic badge resolver.
  badge?: (data: { positions: PaperPosition[]; trades: PaperTrade[] }) => string;
}

const TABS: TabSpec[] = [
  {
    slug: "positions",
    href: "/paper-trading/positions",
    label: "Positions",
    icon: TabPositions,
    badge: ({ positions }) => positions.length > 0 ? `(${positions.length})` : "",
  },
  {
    slug: "trades",
    href: "/paper-trading/trades",
    label: "Trades",
    icon: TabTrades,
    badge: ({ trades }) => trades.length > 0 ? `(${trades.length})` : "",
  },
  { slug: "nav", href: "/paper-trading/nav", label: "NAV Chart", icon: TabNavChart },
  { slug: "reality-gap", href: "/paper-trading/reality-gap", label: "Reality gap", icon: TabRealityGap },
  { slug: "exit-quality", href: "/paper-trading/exit-quality", label: "Exit quality", icon: TabExitQuality },
  // phase-44.2 (operator_approval_44.2.md, 2026-05-26): Manage tab REMOVED
  // from the tablist per operator approval. The /paper-trading/manage
  // sub-route is STILL REACHABLE via the Settings gear button in the
  // page header (see action-buttons block above) -- removed from tablist,
  // not from the app. /paper-trading/manage/page.tsx hosts the Top up
  // fund deposit + all 10 paper-trading knobs (lite_mode, max_positions,
  // per_sector cap, daily_cost_cap, stop_loss, screen_top_n, analyze_top_n,
  // transaction_cost, daily_loss_limit, trailing_dd_limit, min_cash_reserve).
];

function activeTabIndex(pathname: string | null): number {
  if (!pathname) return 0;
  for (let i = 0; i < TABS.length; i++) {
    if (pathname === TABS[i].href || pathname.startsWith(TABS[i].href + "/")) {
      return i;
    }
  }
  return 0;
}

// ── Layout ────────────────────────────────────────────────────────

export default function PaperTradingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const activeIdx = activeTabIndex(pathname);

  const [status, setStatus] = useState<PaperTradingStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PaperPortfolio | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [snapshots, setSnapshots] = useState<PaperSnapshot[]>([]);
  const [perf, setPerf] = useState<PaperPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [rationaleTradeId, setRationaleTradeId] = useState<string | null>(null);

  const positionTickers = useMemo(() => positions.map((p) => p.ticker), [positions]);
  const { prices: livePrices } = useLivePrices(positionTickers, positions.length > 0);
  const { liveNav, liveTotalPnlPct } = useLiveNav(status, positions, livePrices);

  const allTickersForMeta = useMemo(() => {
    const set = new Set<string>();
    positions.forEach((p) => p.ticker && set.add(p.ticker));
    trades.forEach((t) => t.ticker && set.add(t.ticker));
    return Array.from(set);
  }, [positions, trades]);
  const { meta: tickerMeta } = useTickerMeta(
    allTickersForMeta,
    allTickersForMeta.length > 0,
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

  const isInitialized = status?.status !== "not_initialized";
  const isActive = status?.scheduler_active;
  const cycleRunning = !!status?.loop.running;

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
      clearTimers();
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

  // ── Keyboard nav for the tablist (W3C APG manual activation) ────

  const tabRefs = useRef<Array<HTMLAnchorElement | null>>([]);
  const focusTab = (idx: number) => {
    const a = tabRefs.current[idx];
    if (a) a.focus();
  };
  const onTabKeyDown = (e: KeyboardEvent<HTMLAnchorElement>) => {
    const total = TABS.length;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      focusTab((activeIdx + 1) % total);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      focusTab((activeIdx - 1 + total) % total);
    } else if (e.key === "Home") {
      e.preventDefault();
      focusTab(0);
    } else if (e.key === "End") {
      e.preventDefault();
      focusTab(total - 1);
    }
  };

  const ctxValue: PaperTradingDataValue = {
    status,
    portfolio,
    positions,
    trades,
    snapshots,
    perf,
    livePrices,
    liveNav,
    liveTotalPnlPct,
    tickerMeta,
    loading,
    error,
    refresh,
    openRationale: setRationaleTradeId,
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Fixed header zone */}
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
              {/* phase-44.2 cycle 67 follow-up: Manage settings + Top up fund
                  reached via a gear button now (removed from tablist per
                  operator approval but kept reachable so settings aren't
                  orphaned). Sub-route at /paper-trading/manage was restored. */}
              {isInitialized && (
                <Link
                  href="/paper-trading/manage"
                  aria-label="Paper trading settings + deposit"
                  className="flex items-center gap-1.5 rounded-lg border border-navy-700 bg-navy-800/60 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-navy-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px]"
                >
                  <Gear size={16} weight="regular" />
                  <span>Settings</span>
                </Link>
              )}
            </div>
          </div>

          {/* Tab bar (Tier 5) -- W3C WAI APG link-based tablist */}
          {isInitialized && (
            <div
              role="tablist"
              aria-label="Paper trading sections"
              className="flex gap-1 rounded-lg bg-navy-800/50 p-1"
            >
              {TABS.map((t, i) => {
                const isActiveTab = i === activeIdx;
                const badge = t.badge ? t.badge({ positions, trades }) : "";
                return (
                  <Link
                    key={t.slug}
                    ref={(el) => { tabRefs.current[i] = el; }}
                    href={t.href}
                    role="tab"
                    id={`tab-${t.slug}`}
                    aria-selected={isActiveTab}
                    aria-controls={`panel-${t.slug}`}
                    tabIndex={isActiveTab ? 0 : -1}
                    onKeyDown={onTabKeyDown}
                    className={clsx(
                      "flex flex-1 items-center justify-center gap-2 min-h-[24px] rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40",
                      isActiveTab
                        ? "bg-sky-500/10 text-sky-400"
                        : "text-slate-400 hover:text-slate-200",
                    )}
                  >
                    <t.icon size={16} weight={isActiveTab ? "fill" : "regular"} />
                    <span>{t.label}{badge ? ` ${badge}` : ""}</span>
                  </Link>
                );
              })}
            </div>
          )}
        </div>

        {/* Scrollable content zone */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
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
            <PaperTradingDataContext.Provider value={ctxValue}>
              <OpsStatusBar nextRunAt={status?.next_run} />
              <SummaryHero
                status={status}
                perf={perf}
                liveNav={liveNav}
                liveTotalPnlPct={liveTotalPnlPct}
              />
              {children}
            </PaperTradingDataContext.Provider>
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
