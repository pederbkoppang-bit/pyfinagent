"use client";

// phase-44.2 -- shared data context for the route-split /paper-trading
// cockpit. The parent layout.tsx fetches `status`, `portfolio`, `positions`,
// `trades`, `snapshots`, `perf`, `livePrices`, `liveNav`, `tickerMeta` ONCE
// and publishes them via this Context so each sub-route consumes without
// prop-drilling. Sub-routes that need additional fetches (manage settings,
// reality-gap reconciliation) own those fetches locally.

import { createContext, useContext } from "react";
import type {
  PaperTradingStatus,
  PaperPortfolio,
  PaperPosition,
  PaperTrade,
  PaperSnapshot,
  PaperPerformance,
} from "@/lib/types";

// Re-exports the hook's entry shape so consumers stay consistent
// (the hook owns the canonical fields including `cached` / `rate_gated`).
export type { LivePriceEntry } from "@/lib/useLivePrices";
import type { LivePriceEntry as _LivePriceEntry } from "@/lib/useLivePrices";

export interface TickerMeta {
  company_name?: string;
  sector?: string;
}

export interface PaperTradingDataValue {
  status: PaperTradingStatus | null;
  portfolio: PaperPortfolio | null;
  positions: PaperPosition[];
  trades: PaperTrade[];
  snapshots: PaperSnapshot[];
  perf: PaperPerformance | null;
  livePrices: Record<string, _LivePriceEntry>;
  liveNav: number | null;
  liveTotalPnlPct: number | null;
  tickerMeta: Record<string, TickerMeta>;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  // Drawer open is owned at the layout level so any sub-route can trigger.
  openRationale: (tradeId: string | null) => void;
  // goal-multimarket-ux: global market filter. "ALL" = combined USD-base view.
  // Sub-routes filter `positions`/`trades` by this; the layout owns the state.
  activeMarket: string;
  setActiveMarket: (market: string) => void;
}

export const PaperTradingDataContext =
  createContext<PaperTradingDataValue | null>(null);

export function usePaperTradingData(): PaperTradingDataValue {
  const ctx = useContext(PaperTradingDataContext);
  if (ctx === null) {
    throw new Error(
      "usePaperTradingData must be used inside the /paper-trading layout context",
    );
  }
  return ctx;
}
