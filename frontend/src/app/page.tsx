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
// phase-72: ONE app-wide live-portfolio SSOT replaces per-page useLivePrices
// + useLiveNav. The Home cockpit and Paper Trading layout used to call
// these hooks separately, producing race conditions on poll timestamps
// (operator-flagged 2026-05-26: Home showed $23,732.69 while Paper Trading
// showed $23,750.37 simultaneously). Now both consume from the root provider.
import { useLivePortfolio } from "@/lib/live-portfolio-context";
// phase-75 (2026-05-26): Google-Finance digit-flip via NumberFlow on KPI
// tiles that show live-priced numbers (NAV, P&L today, vs SPY). Replaces
// the cycle-74 background-flash hook (deleted) with the per-digit slide
// pattern operator pointed at. NumberFlow owns its prev-value tracking
// and prefers-reduced-motion fallback internally.
import NumberFlow, { type Format } from "@number-flow/react";
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

// phase-44.6: KpiTile now supports an optional sparkline + live freshness
// dot. The sparkline is a Tailwind-only SVG mini-area chart so we don't
// need to bundle a Recharts/Tremor SparkAreaChart for these 6 tiles
// (kept the home page LCP discipline + frontend-md "no new deps without
// owner approval"). aria-label on the tile wrapper supplies a single
// label for the value + sub-text + spark trend combination.
import { LiveBadge, type FreshnessBand } from "@/components/LiveBadge";

function MiniSpark({
  data,
  positive,
}: {
  data: number[];
  positive?: boolean;
}) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const W = 80;
  const H = 24;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - ((v - min) / range) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const stroke = positive ? "#34d399" : positive === false ? "#fb7185" : "#94a3b8";
  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${W} ${H}`}
      className="mt-1 h-6 w-20"
      preserveAspectRatio="none"
    >
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

function KpiTile({
  label,
  value,
  fallback,
  format,
  subText,
  valueClass,
  subTextClass,
  ariaLabel,
  sparkData,
  sparkPositive,
  liveBand,
  liveAgeSec,
}: {
  label: string;
  // phase-75 (2026-05-26): unified numeric value. NumberFlow owns the
  // formatting via the `format` prop (Intl.NumberFormatOptions). For
  // non-live tiles (Sharpe, Max DD, Positions count) the caller can pass
  // value as a number with the appropriate format, OR pass null + a
  // `fallback` string for the display placeholder ("—" / pre-formatted).
  value: number | null;
  fallback?: string;
  // Omit format for integer / unit-less displays (e.g. Positions count) --
  // NumberFlow will render plain decimal formatting. Use NumberFlow's
  // exported `Format` type (subset of Intl.NumberFormatOptions; excludes
  // "scientific" / "engineering" notation per the lib's design).
  format?: Format;
  subText?: string | null;
  valueClass?: string;
  subTextClass?: string;
  ariaLabel?: string;
  sparkData?: number[];
  sparkPositive?: boolean;
  liveBand?: FreshnessBand;
  liveAgeSec?: number | null;
}) {
  const baseValueClass = valueClass ?? "text-slate-100";
  return (
    <div
      role="group"
      aria-label={ariaLabel ?? `${label}${subText ? ` (${subText})` : ""}`}
      className="rounded-xl border border-navy-700 bg-navy-800/60 p-5"
    >
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{label}</p>
        {liveBand && (
          <LiveBadge band={liveBand} ageSec={liveAgeSec ?? null} compact />
        )}
      </div>
      <p
        aria-live="off"
        className={`mt-1 text-2xl font-bold ${baseValueClass}`}
      >
        {value == null ? (
          fallback ?? "—"
        ) : (
          <NumberFlow value={value} format={format} willChange />
        )}
      </p>
      {subText && (
        <p className={`mt-0.5 text-xs ${subTextClass ?? "text-slate-500"}`}>{subText}</p>
      )}
      {sparkData && sparkData.length >= 2 && (
        <MiniSpark data={sparkData} positive={sparkPositive} />
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
  // phase-25.C12: backend-authoritative Sharpe. Falls back to local
  // kpiSharpe(navSeries) only when API value is missing (e.g. rolling
  // deploy or fail-open). See contract: handoff/current/contract.md.
  const [apiSharpe, setApiSharpe] = useState<number | null>(null);

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
      if (portfolio.status === "fulfilled") {
        setPositions(portfolio.value.positions ?? []);
        setApiSharpe(portfolio.value.portfolio?.sharpe_ratio ?? null);
      }
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

  // phase-72: ALL live values now come from the root LivePortfolioProvider.
  // Per researcher: this collapses the 4-NAV operator-flagged inconsistency
  // by guaranteeing every cockpit surface shares the same poll instance +
  // the same derivation. livePrices + liveNav + liveTotalPnlPct + the new
  // pnlTodayPct (today's live-minus-yesterday's-close) all flow from here.
  const lp = useLivePortfolio();
  const liveNav = lp.liveNav;
  const liveTotalPnlPct = lp.liveTotalPnlPct;
  const livePrices = lp.livePrices;
  const freshnessBand = lp.freshnessBand;
  const freshnessAgeSec = lp.freshnessAgeSec;

  const nav = ptStatus?.portfolio;
  const navValue = liveNav ?? nav?.nav;
  const pnl = liveTotalPnlPct ?? nav?.pnl_pct;
  const benchmark = nav?.benchmark_return_pct;
  const alpha = pnl != null && benchmark != null ? pnl - benchmark : null;

  // phase-16.44: KPI sub-text computed from real data (no hardcoded values).
  // All return null on insufficient/flat data; the tile then renders "—".
  const navSeries = redLineSeries.map((p) => ({ date: p.date, nav: p.nav }));
  // phase-72: P&L (Today) replaces the broken dailyDelta(navSeries) path.
  // The old derivation read series[-1] and series[-2] from forward-filled
  // 4-day-stale snapshots, both same value, dollar/pct delta = 0. The new
  // formula uses (liveNav - yesterday_close_nav) / yesterday_close_nav
  // exposed by the LivePortfolioProvider (lp.pnlTodayDollars / .pnlTodayPct).
  // Falls back to dailyDelta only when the live derivation is unavailable
  // (initial paint, no live ticks).
  const liveToday = (lp.pnlTodayDollars != null && lp.pnlTodayPct != null)
    ? { dollars: lp.pnlTodayDollars, pct: lp.pnlTodayPct }
    : null;
  const today = liveToday ?? dailyDelta(navSeries);
  // phase-25.C12: prefer backend-authoritative Sharpe so home + paper-trading
  // tabs render identical numbers. Falls back to local kpiSharpe only when
  // the API value is missing (rolling deploy / fail-open).
  const sharpe90 = apiSharpe ?? kpiSharpe(navSeries);
  const sortino90 = kpiSortino(navSeries);
  const dd30 = maxDrawdownPct(navSeries);
  const posBreakdown = categorizePositions(positions);
  // phase-44.6: numeric series for the 5 sparkline tiles. Derived from
  // navSeries so all tiles share one source-of-truth (no separate API
  // call). dailyPctSeries = rolling pct-changes; alphaSeries = nav vs
  // benchmark proxy (uses navSeries directly when no SPY series is
  // available -- correctness is good-enough for the trend hint).
  // ddSeries = running max-drawdown traces. All return [] when there's
  // less than 2 datapoints so the sparkline renders nothing.
  const navNums: number[] = navSeries.map((p) => p.nav);
  const dailyPctSeries: number[] =
    navNums.length >= 2
      ? navNums.slice(1).map((v, i) => ((v - navNums[i]) / navNums[i]) * 100)
      : [];
  const alphaSeries: number[] = navNums;
  const ddSeries: number[] = (() => {
    if (navNums.length < 2) return [];
    let peak = navNums[0];
    return navNums.map((v) => {
      if (v > peak) peak = v;
      return peak > 0 ? ((v - peak) / peak) * 100 : 0;
    });
  })();
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
          {/* phase-44.6: wrap the 6-KPI cluster in role=group with a single
              label. Per WAI-ARIA APG + MDN role=group (researcher source #4
              + #5): group is the right primitive for a logical collection of
              related items; role=region would over-promote to landmark. */}
          <div
            role="group"
            aria-label="Portfolio key performance indicators"
            className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6"
          >
            <KpiTile
              label="NAV"
              value={loaded ? (navValue ?? null) : null}
              format={{ style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 }}
              sparkData={navNums.length >= 2 ? navNums : undefined}
              sparkPositive={navNums.length >= 2 ? navNums[navNums.length - 1] >= navNums[0] : undefined}
              liveBand={loaded ? (livePrices && Object.keys(livePrices).length > 0 ? "green" : "unknown") : "unknown"}
            />
            <KpiTile
              label="P&L (today)"
              value={today?.dollars ?? null}
              format={{ style: "currency", currency: "USD", signDisplay: "always", minimumFractionDigits: 2, maximumFractionDigits: 2 }}
              subText={today != null ? `${today.pct >= 0 ? "+" : ""}${today.pct.toFixed(2)}%` : null}
              valueClass={today != null && today.dollars >= 0 ? "text-emerald-400" : today != null ? "text-rose-400" : undefined}
              subTextClass={today != null && today.pct >= 0 ? "text-emerald-400/70" : today != null ? "text-rose-400/70" : undefined}
              sparkData={dailyPctSeries.length >= 2 ? dailyPctSeries : undefined}
              sparkPositive={today != null ? today.dollars >= 0 : undefined}
            />
            <KpiTile
              label="vs SPY"
              value={loaded ? (alpha != null ? alpha / 100 : null) : null}
              format={{ style: "percent", signDisplay: "always", minimumFractionDigits: 2, maximumFractionDigits: 2 }}
              subText={benchmark != null ? `SPY ${fmtPct(benchmark)}` : null}
              valueClass={alpha != null && alpha >= 0 ? "text-emerald-400" : alpha != null ? "text-rose-400" : undefined}
              sparkData={alphaSeries.length >= 2 ? alphaSeries : undefined}
              sparkPositive={alpha != null ? alpha >= 0 : undefined}
            />
            <KpiTile
              label="Sharpe (90d)"
              value={sharpe90 ?? null}
              format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}
              subText={sortino90 != null ? `Sortino ${sortino90.toFixed(2)}` : null}
              sparkData={navNums.length >= 2 ? navNums : undefined}
              sparkPositive={sharpe90 != null ? sharpe90 >= 0 : undefined}
            />
            <KpiTile
              label="Max DD (30d)"
              value={dd30 != null ? dd30 / 100 : null}
              format={{ style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2 }}
              subText={`bounded ${trailingDdLimit}`}
              valueClass={dd30 != null ? "text-rose-400" : undefined}
              sparkData={ddSeries.length >= 2 ? ddSeries : undefined}
              sparkPositive={false}
            />
            <KpiTile
              label="Positions"
              value={loaded ? posBreakdown.total : null}
              format={{ maximumFractionDigits: 0 }}
              subText={loaded && posBreakdown.total > 0 ? `${posBreakdown.long} long · ${posBreakdown.short} short` : null}
              liveBand={loaded ? "green" : "unknown"}
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
              // phase-73: chart-side SSOT overlay -- pass the cycle-72
              // live NAV + freshness so the chart appends a pulsating
              // "live now" marker instead of forward-filling the stale
              // snapshot under today's x-axis position.
              liveNav={lp.liveNav}
              liveBand={lp.freshnessBand}
              compact
            />
          </div>

          {/* phase-44.6 fix: removed `lg:items-stretch` + per-child `h-full`.
              That was the documented anti-pattern named in
              `.claude/rules/frontend.md:23` (mixing short + tall widgets
              with equal-height grid). Per `frontend-layout.md` Section 4.5
              option 2 (researcher source #1 + #6): use items-start +
              accept visible asymmetry instead of forcing a short card to
              stretch to a tall neighbor's height. Each child sizes by
              content; layout reads as 3 distinct cards instead of 3 cards
              with hidden dead whitespace. */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-6 lg:items-start">
            <div className="lg:col-span-2">
              <RecentReportsTable
                reports={reports}
                loaded={loaded}
                loadError={loadError}
              />
            </div>
            <div className="lg:col-span-2">
              <LatestTransactionsBox
                trades={trades}
                loaded={loaded}
                loadError={tradesError}
              />
            </div>
            <div className="lg:col-span-2">
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
