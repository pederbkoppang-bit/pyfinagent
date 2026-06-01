"use client";

// phase-50.6: backtest-page market/currency/market-hours context strip.
//
// The walk-forward backtest pipeline is US-only / USD / SPY-benchmarked (an ML
// pipeline distinct from the live multi-market paper loop). This ADDITIVE strip
// labels that scope explicitly + shows the US session open/closed -- reusing the
// shared multimarket helpers (no change to the backtest's own cells/tables).
//
// The open/closed dot uses a mount-guarded clock (null on the server / first
// paint) to avoid an SSR/client hydration mismatch -- same two-pass pattern as
// MarketSessionStrip. No emoji; dark navy palette; JIT-safe literal classes.

import { useEffect, useState } from "react";
import { isMarketOpen, MARKET_BENCHMARK_LABEL, MARKET_EXCHANGE } from "@/lib/format";

export function BacktestScopeStrip({ className }: { className?: string }) {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  const open = now != null && isMarketOpen("US", now);
  const sessionLabel = now != null ? (open ? "OPEN" : "CLOSED") : "--";
  const bench = MARKET_BENCHMARK_LABEL["US"] ?? "vs SPY";

  return (
    <div
      className={`mt-2 inline-flex flex-wrap items-center gap-2 text-[11px] ${className ?? ""}`}
      aria-label="Backtest scope: US market, USD, benchmark SPY"
    >
      <span
        className="inline-flex items-center gap-1.5 rounded-md bg-navy-800/60 px-2 py-0.5 font-mono text-slate-300"
        title={MARKET_EXCHANGE["US"] ?? "US"}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-sky-400" aria-hidden="true" />
        US
      </span>
      <span className="rounded-md bg-navy-800/60 px-2 py-0.5 font-mono text-slate-300">USD</span>
      <span className="rounded-md bg-navy-800/60 px-2 py-0.5 font-mono text-slate-300">{bench}</span>
      <span
        className="inline-flex items-center gap-1.5 rounded-md bg-navy-800/60 px-2 py-0.5"
        title="US cash-session (heuristic; holiday-blind)"
        suppressHydrationWarning
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${open ? "bg-emerald-400" : "bg-slate-600"}`}
          aria-hidden="true"
        />
        <span className={open ? "text-emerald-400" : "text-slate-500"}>{sessionLabel}</span>
      </span>
    </div>
  );
}
