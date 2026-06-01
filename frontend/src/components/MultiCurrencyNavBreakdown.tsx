"use client";

// phase-50.6: multi-currency NAV-breakdown widget.
//
// Groups the fund's USD market values by the local currency of each held
// market (point-in-time snapshot -- no retro-FX revaluation, per the research
// brief / PortfolioPilot convention) and shows each currency's USD sub-total +
// % of NAV. Pure client-side from the existing /portfolio payload (each
// PaperPosition already carries `market` + USD `market_value`); cash is the
// fund base (USD) and is added to the USD bucket when provided.
//
// Reuses format.ts helpers (resolveMarket / MARKET_CURRENCY / formatUsd). Dark
// navy palette + JIT-safe literal dot-class map. No emoji. Graceful when the
// book is single-currency (one row) or empty.

import { useMemo } from "react";
import { resolveMarket, MARKET_CURRENCY, formatUsd } from "@/lib/format";
import type { PaperPosition } from "@/lib/types";

// JIT-safe per-currency dot colors (mirrors MARKET_DOT_CLASS hues).
const CURRENCY_DOT: Record<string, string> = {
  USD: "bg-sky-400",
  EUR: "bg-amber-400",
  KRW: "bg-violet-400",
  NOK: "bg-rose-400",
  SEK: "bg-cyan-400",
  DKK: "bg-pink-400",
  GBP: "bg-emerald-400",
  CAD: "bg-teal-400",
  JPY: "bg-lime-400",
};

export interface MultiCurrencyNavBreakdownProps {
  positions: PaperPosition[];
  totalNav: number | null;
  cashUsd?: number | null;
  title?: string;
  className?: string;
}

export function MultiCurrencyNavBreakdown({
  positions,
  totalNav,
  cashUsd,
  title = "Currency exposure",
  className,
}: MultiCurrencyNavBreakdownProps) {
  const rows = useMemo(() => {
    const acc = new Map<string, number>();
    for (const p of positions) {
      const market = resolveMarket({ market: p.market, ticker: p.ticker });
      const ccy = MARKET_CURRENCY[market] ?? p.base_currency ?? "USD";
      acc.set(ccy, (acc.get(ccy) ?? 0) + (p.market_value ?? 0));
    }
    if (cashUsd && cashUsd > 0) {
      acc.set("USD", (acc.get("USD") ?? 0) + cashUsd);
    }
    const total = Array.from(acc.values()).reduce((s, v) => s + v, 0);
    // Percent denominator: prefer the fund NAV (so it reads as % of NAV) but
    // fall back to the summed total when NAV isn't ready.
    const denom = totalNav && totalNav > 0 ? totalNav : total || 1;
    return Array.from(acc.entries())
      .filter(([, v]) => v > 0)
      .map(([ccy, usd]) => ({ ccy, usd, pct: (usd / denom) * 100 }))
      .sort((a, b) => b.usd - a.usd);
  }, [positions, cashUsd, totalNav]);

  const containerClass = `rounded-xl border border-navy-700 bg-navy-800/70 p-4 ${className ?? ""}`;

  if (rows.length === 0) {
    return (
      <div className={containerClass}>
        <h3 className="mb-2 text-sm font-medium text-slate-300">{title}</h3>
        <p className="text-sm text-slate-400">No holdings yet.</p>
      </div>
    );
  }

  return (
    <div className={containerClass} role="region" aria-label={title}>
      <div className="mb-1 flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        {rows.length === 1 && (
          <span className="text-[11px] text-slate-500">single-currency book</span>
        )}
      </div>
      <p className="mb-3 text-[11px] text-slate-400">
        Fund NAV (USD base) split by the local currency of each held market.
      </p>
      <ul className="space-y-2 text-xs">
        {rows.map((r) => {
          const dot = CURRENCY_DOT[r.ccy] ?? "bg-slate-400";
          return (
            <li key={r.ccy}>
              <div className="flex items-center gap-2">
                <span className={`inline-block h-2 w-2 rounded-full ${dot} shrink-0`} aria-hidden="true" />
                <span className="font-mono text-slate-200">{r.ccy}</span>
                <span className="ml-auto font-mono tabular-nums text-slate-100">{formatUsd(r.usd)}</span>
                <span className="w-12 text-right font-mono tabular-nums text-slate-400">
                  {r.pct.toFixed(1)}%
                </span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-navy-900">
                <div
                  className={`h-full rounded-full ${dot}`}
                  style={{ width: `${Math.min(100, Math.max(0, r.pct)).toFixed(2)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
