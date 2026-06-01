"use client";

/**
 * phase-16.45: Latest Transactions box for the home cockpit.
 *
 * Sits between Recent Reports (left) and Quick Actions (right) in the
 * 4-column home grid. Wired to the existing
 * `GET /api/paper-trading/trades?limit=5` endpoint via the `trades`
 * prop (parent fetches in the same Promise.allSettled batch as the
 * other home-page data so the loading state stays unified).
 *
 * 5 columns: TICKER | SIDE | QTY | PRICE | TIME. Drops total_value,
 * transaction_cost, reason, analysis_id, risk_judge_decision -- those
 * are full-page detail columns visible at /paper-trading.
 *
 * Strict no-hardcoded-data: every value comes from props -- no sample
 * tickers, no sample quantities, no sample prices baked in.
 */

import { useRouter } from "next/navigation";
import Link from "next/link";
import type { PaperTrade } from "@/lib/types";
import { formatRelativeTime } from "@/lib/formatRelativeTime";
import { NavPaperTrading } from "@/lib/icons";
// goal-multimarket-ux: market dot + local-currency price. Trades carry no market
// column, so market is derived from the ticker suffix.
import { MARKET_DOT_CLASS, formatCurrency, resolveCurrency, resolveMarket } from "@/lib/format";

type Props = {
  trades: PaperTrade[];
  loaded: boolean;
  loadError: string | null;
};

function sideColor(action: string): string {
  // Mirrors the existing pattern at frontend/src/app/paper-trading/page.tsx:650-659.
  // BUY = emerald, SELL = rose. Color + text label both -- WCAG accessibility.
  return action === "BUY"
    ? "bg-emerald-500/15 text-emerald-400"
    : "bg-rose-500/15 text-rose-400";
}

function fmtPrice(p: number | null | undefined, ticker?: string): string {
  if (p == null || !Number.isFinite(p)) return "—";
  const cur = resolveCurrency({ ticker });
  // USD path preserved byte-identical (browser-locale grouping); non-USD uses Intl.
  return cur === "USD"
    ? `$${p.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    : formatCurrency(p, cur);
}

function fmtQty(q: number | null | undefined): string {
  if (q == null || !Number.isFinite(q)) return "—";
  // Show fractional shares with up to 4 decimals; integer shares without.
  return Number.isInteger(q)
    ? q.toString()
    : q.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function LatestTransactionsBox({ trades, loaded, loadError }: Props) {
  const router = useRouter();
  const goto = () => router.push("/paper-trading");

  return (
    <div className="h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/40">
      <div className="flex items-center justify-between border-b border-navy-700 px-4 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Latest Transactions
        </h3>
        <Link href="/paper-trading" className="text-xs text-sky-400 hover:text-sky-300">
          View all →
        </Link>
      </div>

      <div className="flex-1 overflow-x-auto">
        <table className="w-full text-left text-sm" aria-label="Latest transactions">
          <thead className="border-b border-navy-700 bg-navy-800/60">
            <tr>
              <th className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Ticker</th>
              <th className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Side</th>
              <th className="px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Qty</th>
              <th className="px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Price</th>
              <th className="px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-700/50">
            {!loaded && [0, 1, 2, 3, 4].map((i) => (
              <tr key={`skel-${i}`} className="animate-pulse">
                <td className="px-3 py-3"><div className="h-4 w-10 rounded bg-navy-700/60" /></td>
                <td className="px-3 py-3"><div className="h-5 w-12 rounded-full bg-navy-700/60" /></td>
                <td className="px-3 py-3 text-right"><div className="ml-auto h-4 w-8 rounded bg-navy-700/60" /></td>
                <td className="px-3 py-3 text-right"><div className="ml-auto h-4 w-14 rounded bg-navy-700/60" /></td>
                <td className="px-3 py-3 text-right"><div className="ml-auto h-4 w-12 rounded bg-navy-700/60" /></td>
              </tr>
            ))}

            {loaded && loadError && trades.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-12">
                  <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-center">
                    <p className="text-sm text-rose-300">{loadError}</p>
                  </div>
                </td>
              </tr>
            )}

            {loaded && !loadError && trades.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <NavPaperTrading size={36} weight="duotone" className="text-slate-600" />
                    <p className="mt-3 text-sm text-slate-400">No trades yet</p>
                    <p className="mt-1 text-xs text-slate-600">Trades appear here after the daily cycle runs</p>
                  </div>
                </td>
              </tr>
            )}

            {loaded && trades.map((t) => (
              <tr
                key={t.trade_id}
                tabIndex={0}
                role="button"
                aria-label={`${t.action} ${t.ticker} ${t.quantity} @ ${t.price}`}
                onClick={goto}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    goto();
                  }
                }}
                className="cursor-pointer transition-colors hover:bg-navy-700/40 focus:bg-navy-700/40 focus:outline-none focus:ring-1 focus:ring-sky-500/40"
              >
                <td className="px-3 py-3 font-mono text-sm font-bold text-slate-100">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${MARKET_DOT_CLASS[resolveMarket({ ticker: t.ticker })] ?? "bg-slate-400"}`}
                      aria-hidden="true"
                    />
                    {t.ticker}
                  </span>
                </td>
                <td className="px-3 py-3">
                  <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${sideColor(t.action)}`}>
                    {t.action}
                  </span>
                </td>
                <td className="px-3 py-3 text-right font-mono text-sm text-slate-300">{fmtQty(t.quantity)}</td>
                <td className="px-3 py-3 text-right font-mono text-sm text-slate-300">{fmtPrice(t.price, t.ticker)}</td>
                <td className="px-3 py-3 text-right text-xs text-slate-500" suppressHydrationWarning>
                  {formatRelativeTime(t.created_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
