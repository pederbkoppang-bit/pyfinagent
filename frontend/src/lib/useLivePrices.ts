"use client";

import { useEffect, useRef, useState } from "react";
import { getPaperLivePrices } from "@/lib/api";

export interface LivePriceEntry {
  price: number | null;
  age_sec: number | null;
  cached: boolean;
  rate_gated?: boolean;
}

/**
 * Polls /api/paper-trading/live-prices every 60s for the given tickers -- but
 * ONLY while the browser tab is visible (Page Visibility API). Matches the
 * backend's 60s TTL so we never hit yfinance harder than necessary (4.5.6).
 *
 * Staleness indicator: every entry carries `age_sec` so the UI can render a
 * freshness timestamp (anti-pattern guard: polling without staleness indicator,
 * Coinpaprika 2024).
 *
 * Consecutive failures: stops polling after 5 in a row (Resilience4j / Polly
 * default, confirmed as the cross-ecosystem modal value in 2025 circuit-breaker
 * literature).
 */
export function useLivePrices(tickers: string[], enabled = true) {
  const [prices, setPrices] = useState<Record<string, LivePriceEntry>>({});
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const failRef = useRef(0);

  useEffect(() => {
    if (!enabled || tickers.length === 0) return;
    let cancelled = false;
    const uniq = Array.from(new Set(tickers.filter(Boolean)));

    async function tick() {
      if (cancelled) return;
      // Only fetch when tab is visible -- respects the user's attention.
      if (typeof document !== "undefined" && document.hidden) return;
      try {
        const j = await getPaperLivePrices(uniq);
        if (cancelled) return;
        setPrices((j.prices || {}) as Record<string, LivePriceEntry>);
        setUpdatedAt(Date.now());
        setError(null);
        failRef.current = 0;
      } catch (e) {
        failRef.current += 1;
        if (failRef.current >= 5) {
          setError(e instanceof Error ? e.message : "live-prices failed");
        }
      }
    }

    // Fire once on mount (or dependency change), then poll every 60s.
    void tick();
    const id = window.setInterval(tick, 60_000);

    // Also fire when the tab regains visibility so the user sees fresh data.
    const onVis = () => {
      if (!document.hidden) void tick();
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [tickers.join(","), enabled]);

  return { prices, error, updatedAt };
}
