"use client";

// phase-72 cycle (2026-05-26) -- root-level live-portfolio SSOT.
//
// Operator-flagged 2026-05-26 that 4 different NAV values were rendering
// simultaneously across cockpit surfaces (Home tile, Paper Trading tile,
// Donut center, Slack digest, Red Line tooltip). Root cause per researcher
// `a9c94760e6f0240af` (deep-tier brief at
// research_brief_phase_ssot_nav.md):
//
//   1. Two competing sources (persisted BQ snapshot + frontend live recompute)
//      stored alongside each other with no surface labels.
//   2. Two separate `useLivePrices` instances polling independently --
//      one on the Home page, one on the Paper Trading layout. Race condition
//      on poll timestamp produces ~$18 gap between Home and Paper Trading.
//   3. P&L (Today) computed from `dailyDelta(redLineSeries)` over 4-day-stale
//      snapshots -> series[-1] == series[-2] -> $0.00.
//
// Fix (Path A from brief): lift `useLivePrices` + `useLiveNav` into THIS
// root-level Context. ONE polling instance, ONE derivation, consumed by
// every surface. Zero new deps. Mirrors the existing AuthProvider pattern.
//
// Backward-compat: the `useLivePrices` + `useLiveNav` hooks are NOT
// deleted -- this provider USES them internally. Sub-route consumers
// (e.g. `PaperTradingDataContext`) read from this provider going forward.

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { usePathname } from "next/navigation";
import {
  getPaperTradingStatus,
  getPaperPortfolio,
  getPaperSnapshots,
} from "@/lib/api";
import { useLivePrices, type LivePriceEntry } from "@/lib/useLivePrices";
import { useLiveNav } from "@/lib/useLiveNav";
import { useTickerMeta } from "@/lib/useTickerMeta";
import type {
  PaperPosition,
  PaperPortfolio,
  PaperSnapshot,
  PaperTradingStatus,
} from "@/lib/types";

export type FreshnessBand = "green" | "amber" | "red" | "unknown";

export interface LivePortfolioValue {
  // Shared upstream state
  status: PaperTradingStatus | null;
  portfolio: PaperPortfolio | null;
  positions: PaperPosition[];
  snapshots: PaperSnapshot[];
  livePrices: Record<string, LivePriceEntry>;
  tickerMeta: Record<string, { company_name?: string; sector?: string }>;
  // Derived live values (SSOT)
  liveNav: number | null;
  liveTotalPnlPct: number | null;
  // Today's P&L derived from live - yesterday's close (replaces the
  // broken dailyDelta path).
  pnlTodayPct: number | null;
  pnlTodayDollars: number | null;
  // Freshness: max age across all live-price entries. amber=stale (90-300s),
  // red>=300s. Used by surfaces to render a LiveBadge.
  freshnessBand: FreshnessBand;
  freshnessAgeSec: number | null;
  // Latest persisted snapshot date (yyyy-mm-dd), used by surfaces that
  // intentionally render the persisted-snapshot NAV (Red Line, Slack)
  // so they can label "as of YYYY-MM-DD".
  latestSnapshotDate: string | null;
  // Loading + error state surfaced for top-level error banners.
  loading: boolean;
  error: string | null;
  // Imperative refresh trigger (e.g. after a manual paper-trading action).
  refresh: () => Promise<void>;
}

const LivePortfolioContext = createContext<LivePortfolioValue | null>(null);

const POLL_INTERVAL_MS = 60_000; // 60s; matches cycle-23.1.17 + useLivePrices

function deriveFreshness(
  prices: Record<string, LivePriceEntry>,
): { band: FreshnessBand; ageSec: number | null } {
  const ages: number[] = [];
  for (const v of Object.values(prices)) {
    if (typeof v?.age_sec === "number") ages.push(v.age_sec);
  }
  if (ages.length === 0) return { band: "unknown", ageSec: null };
  const max = Math.max(...ages);
  if (max < 90) return { band: "green", ageSec: max };
  if (max < 300) return { band: "amber", ageSec: max };
  return { band: "red", ageSec: max };
}

export function LivePortfolioProvider({ children }: { children: ReactNode }) {
  // phase-75.12 (frontend-02): root-mounted LivePortfolioProvider (see
  // app/layout.tsx:36) previously polled authed endpoints unconditionally,
  // including on a logged-out /login visit -- combined with api.ts's
  // then-unguarded 401 redirect, this produced a sub-second reload loop
  // that interrupted SSO/passkey ceremonies. Gate the initial fetch, the
  // 60s interval, and the live-price/ticker-meta polls on pathname !==
  // "/login". This realizes the "future hardening pass" this comment
  // used to describe.
  const pathname = usePathname();
  const isLoginPage = pathname === "/login";

  const [status, setStatus] = useState<PaperTradingStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PaperPortfolio | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [snapshots, setSnapshots] = useState<PaperSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Stable poll for status + portfolio + snapshots. Live prices are
  // polled by the dedicated useLivePrices hook below (its own 60s loop).
  const refresh = useMemo(() => {
    return async () => {
      if (isLoginPage) return;
      try {
        const [s, port, snap] = await Promise.allSettled([
          getPaperTradingStatus(),
          getPaperPortfolio(),
          getPaperSnapshots(),
        ]);
        if (s.status === "fulfilled") setStatus(s.value);
        if (port.status === "fulfilled" && port.value) {
          setPortfolio(port.value.portfolio);
          setPositions(port.value.positions ?? []);
        }
        if (snap.status === "fulfilled") setSnapshots(snap.value.snapshots ?? []);
        // Surface error only when ALL three failed (graceful degrade).
        const allFailed = [s, port, snap].every((r) => r.status === "rejected");
        if (allFailed) {
          const first = [s, port, snap].find((r) => r.status === "rejected");
          setError(
            first && first.status === "rejected" && first.reason instanceof Error
              ? first.reason.message
              : "Failed to load portfolio state",
          );
        } else {
          setError(null);
        }
      } finally {
        setLoading(false);
      }
    };
  }, [isLoginPage]);

  useEffect(() => {
    if (isLoginPage) {
      // Nothing will populate `positions`/`status` while gated -- clear
      // the initial loading spinner rather than leaving it stuck forever
      // for any /login-rendered consumer of useLivePortfolioOptional().
      setLoading(false);
      return;
    }
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [refresh, isLoginPage]);

  // ONE useLivePrices instance for the whole app. Every surface gets
  // the same prices at the same moment (no race). Next.js App Router
  // keeps this provider mounted across client-side navigations, so
  // `positions` can still hold a stale non-empty array from a previous
  // route when the user lands on /login (e.g. after the 401 redirect
  // above) -- `!isLoginPage` guards that, not just `positions.length`.
  const positionTickers = useMemo(
    () => positions.map((p) => p.ticker),
    [positions],
  );
  const { prices: livePrices } = useLivePrices(
    positionTickers,
    !isLoginPage && positions.length > 0,
  );

  // ONE useLiveNav derivation for the whole app.
  const { liveNav, liveTotalPnlPct } = useLiveNav(
    status,
    positions,
    livePrices,
  );

  // Ticker meta (company name + sector) for cross-surface filters.
  const allTickers = useMemo(() => {
    const set = new Set<string>();
    positions.forEach((p) => p.ticker && set.add(p.ticker));
    return Array.from(set);
  }, [positions]);
  const { meta: tickerMeta } = useTickerMeta(allTickers, !isLoginPage && allTickers.length > 0);

  // Freshness band: max age across live-price entries.
  const { band: freshnessBand, ageSec: freshnessAgeSec } = useMemo(
    () => deriveFreshness(livePrices),
    [livePrices],
  );

  // Latest persisted snapshot date for "as of YYYY-MM-DD" labels.
  const latestSnapshotDate = useMemo(() => {
    if (!snapshots || snapshots.length === 0) return null;
    // snapshots come in DESC order from getPaperSnapshots; [0] is latest.
    return snapshots[0].snapshot_date ?? null;
  }, [snapshots]);

  // P&L (Today) = live NAV - latest snapshot NAV (yesterday's close).
  // Per researcher: replaces the broken `dailyDelta(redLineSeries)` path
  // that returned 0 when both endpoints of the series were the same stale row.
  const { pnlTodayDollars, pnlTodayPct } = useMemo(() => {
    if (
      liveNav == null ||
      snapshots.length === 0 ||
      snapshots[0].total_nav == null ||
      snapshots[0].total_nav <= 0
    ) {
      return { pnlTodayDollars: null, pnlTodayPct: null };
    }
    const yesterdayNav = snapshots[0].total_nav;
    const dollars = liveNav - yesterdayNav;
    const pct = (dollars / yesterdayNav) * 100;
    return { pnlTodayDollars: dollars, pnlTodayPct: pct };
  }, [liveNav, snapshots]);

  const value = useMemo<LivePortfolioValue>(
    () => ({
      status,
      portfolio,
      positions,
      snapshots,
      livePrices,
      tickerMeta,
      liveNav,
      liveTotalPnlPct,
      pnlTodayDollars,
      pnlTodayPct,
      freshnessBand,
      freshnessAgeSec,
      latestSnapshotDate,
      loading,
      error,
      refresh,
    }),
    [
      status,
      portfolio,
      positions,
      snapshots,
      livePrices,
      tickerMeta,
      liveNav,
      liveTotalPnlPct,
      pnlTodayDollars,
      pnlTodayPct,
      freshnessBand,
      freshnessAgeSec,
      latestSnapshotDate,
      loading,
      error,
      refresh,
    ],
  );

  return (
    <LivePortfolioContext.Provider value={value}>
      {children}
    </LivePortfolioContext.Provider>
  );
}

// Soft-consumer hook: returns null if no provider is mounted. Use this
// from a component that may render outside the provider (e.g. /login).
export function useLivePortfolioOptional(): LivePortfolioValue | null {
  return useContext(LivePortfolioContext);
}

// Strict consumer: throws if no provider is mounted. Use from components
// that must be inside the cockpit tree (Home, Paper Trading, Donut).
export function useLivePortfolio(): LivePortfolioValue {
  const ctx = useContext(LivePortfolioContext);
  if (ctx === null) {
    throw new Error(
      "useLivePortfolio must be used inside <LivePortfolioProvider>",
    );
  }
  return ctx;
}
