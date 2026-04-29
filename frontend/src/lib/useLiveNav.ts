"use client";

import { useMemo } from "react";

import type { LivePriceEntry } from "@/lib/useLivePrices";
import type { PaperPosition, PaperTradingStatus } from "@/lib/types";

/**
 * phase-23.1.17 — single source of truth for the live-derived NAV and
 * total-P&L percent. Both the home (`/`) and paper-trading
 * (`/paper-trading`) pages compute these the same way: NAV is the
 * post-refund cash plus the sum of (live yfinance price * quantity)
 * across positions; total-P&L pct is anchored to the operator's
 * `starting_capital` (deposit-aware via phase-23.1.9).
 *
 * Falls back to the BQ snapshot fields when no live ticks are available
 * (initial paint, empty positions, yfinance offline).
 */
export interface UseLiveNavResult {
  liveNav: number | null;
  liveTotalPnlPct: number | null;
}

export function useLiveNav(
  status: PaperTradingStatus | null,
  positions: PaperPosition[],
  livePrices: Record<string, LivePriceEntry>,
): UseLiveNavResult {
  const liveNav = useMemo(() => {
    if (positions.length === 0) return status?.portfolio.nav ?? null;
    const cash = status?.portfolio.cash ?? 0;
    const hasAnyLive = positions.some((p) => livePrices[p.ticker]?.price != null);
    if (!hasAnyLive) return status?.portfolio.nav ?? null;
    const positionsValue = positions.reduce((sum, pos) => {
      const lp =
        livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
      return sum + lp * pos.quantity;
    }, 0);
    return cash + positionsValue;
  }, [positions, livePrices, status]);

  const liveTotalPnlPct = useMemo(() => {
    const startingCapital = status?.portfolio.starting_capital ?? null;
    if (liveNav == null || startingCapital == null || startingCapital <= 0) {
      return status?.portfolio.pnl_pct ?? null;
    }
    return ((liveNav - startingCapital) / startingCapital) * 100;
  }, [liveNav, status]);

  return { liveNav, liveTotalPnlPct };
}
