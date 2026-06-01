"use client";

// goal-multimarket-ux #6: market-session indicator. Shows each market's open/closed
// state (heuristic; holiday-blind -- the backend exchange_calendars gate is
// authoritative, this is a UI hint; see format.ts isMarketOpen). Dot = emerald (open)
// / slate (closed). NO emoji. Re-evaluates every 60s. Computes only after mount to
// avoid a server/client hydration mismatch on the OPEN/CLOSED text.

import { useEffect, useState } from "react";
import { isMarketOpen, MARKET_EXCHANGE } from "@/lib/format";

const DEFAULT_MARKETS = ["US", "EU", "KR"];

export function MarketSessionStrip({
  className,
  markets = DEFAULT_MARKETS,
}: {
  className?: string;
  markets?: string[];
}) {
  // `now` is null on the server / first client render (renders "—" to avoid a
  // hydration mismatch), then set on mount and refreshed each minute so the
  // open/closed dots stay current without a heavy poll.
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div
      className={`inline-flex flex-wrap items-center gap-3 ${className ?? ""}`}
      aria-label="Market sessions"
    >
      {markets.map((m) => {
        const open = now != null && isMarketOpen(m, now);
        const label = now != null ? (open ? "OPEN" : "CLOSED") : "—";
        return (
          <span
            key={m}
            className="inline-flex items-center gap-1.5 text-xs"
            title={MARKET_EXCHANGE[m] ?? undefined}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${open ? "bg-emerald-400" : "bg-slate-600"}`}
              aria-hidden="true"
            />
            <span className="font-mono text-slate-300">{m}</span>
            <span className={open ? "text-emerald-400" : "text-slate-500"}>{label}</span>
          </span>
        );
      })}
    </div>
  );
}
