"use client";

import { useEffect, useRef, useState } from "react";
import { getTickerMeta } from "@/lib/api";
import type { TickerMeta } from "@/lib/types";

/**
 * phase-23.1.10 — fetch {ticker -> {company_name, sector}} once per unique
 * ticker set. Re-fetches when the set changes (sorted-key dedup so
 * order-only changes don't refetch). Graceful on error: returns empty map
 * so the table renders ticker-only without crashing.
 */
export function useTickerMeta(tickers: string[], enabled = true) {
  const [meta, setMeta] = useState<Record<string, TickerMeta>>({});
  const fetchedKey = useRef<string>("");

  const uniq = Array.from(new Set(tickers.filter(Boolean))).sort();
  const key = uniq.join(",");

  useEffect(() => {
    if (!enabled || uniq.length === 0) return;
    if (fetchedKey.current === key) return;
    fetchedKey.current = key;

    let cancelled = false;
    getTickerMeta(uniq)
      .then((r) => {
        if (!cancelled) setMeta(r.meta ?? {});
      })
      .catch(() => {
        // graceful — tables show ticker-only on miss
      });
    return () => {
      cancelled = true;
    };
    // key is the sorted-joined tickers; serves as the dependency
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  return { meta };
}
