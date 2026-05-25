// phase-44.2 -- paper-trading utility helpers.
//
// Pure functions used by sub-routes to derive UI state from server data.
// Kept side-effect-free so they unit-test without a DOM.

import type { PaperTrade } from "@/lib/types";

// Finds the trade_id of the most recent BUY trade for a given ticker.
// Used by the positions sub-route to open AgentRationaleDrawer on row click
// -- `PaperPosition` lacks a `last_trade_id` field, so we look it up
// against the trades list. Returns null if no BUY trade exists for the
// ticker (defensive; positions without a recorded BUY should not exist
// in practice but we don't want to crash if data is partial).
export function latestTradeIdForTicker(
  trades: PaperTrade[],
  ticker: string,
): string | null {
  let latest: PaperTrade | null = null;
  for (const t of trades) {
    if (t.ticker !== ticker) continue;
    if (t.action !== "BUY") continue;
    if (latest === null || t.created_at > latest.created_at) {
      latest = t;
    }
  }
  return latest?.trade_id ?? null;
}

// Maps a livePrices entry's age in seconds to a freshness band that the
// LiveBadge consumes. Thresholds chosen to match `OpsStatusBar` /
// `useLivePrices`: under 90s = green (within one poll), under 300s = amber
// (one missed poll), over 300s OR unknown = red.
export type FreshnessBand = "green" | "amber" | "red" | "unknown";

export function bandFromAgeSec(ageSec: number | null | undefined): FreshnessBand {
  if (ageSec == null) return "unknown";
  if (ageSec < 90) return "green";
  if (ageSec < 300) return "amber";
  return "red";
}
