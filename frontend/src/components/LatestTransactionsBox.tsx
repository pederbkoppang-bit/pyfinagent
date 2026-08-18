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
 * `table-fixed` + explicit column widths (2026-08-14): with only 5 rows
 * this table never grew wider than its card, but fetching 20 rows for
 * the home page (see page.tsx) surfaced long values -- Korean Won prices
 * ("W2,425,000") and long tickers ("000660.KS") -- that pushed the
 * table-auto layout to 407px inside a 354px card, clipping the right
 * edge. table-fixed + truncate caps every column to the card's actual
 * width regardless of content, matching RecentReportsTable.tsx.
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
  // 2 decimals (was 4) -- this compact card's Qty column has ~40px of
  // usable width after table-fixed column sizing; 4 decimals ("4.8064")
  // regularly overflowed it. Full precision remains on /paper-trading
  // and in the title tooltip on this cell.
  return Number.isInteger(q)
    ? q.toString()
    : q.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function fmtQtyFull(q: number | null | undefined): string {
  if (q == null || !Number.isFinite(q)) return "—";
  return Number.isInteger(q)
    ? q.toString()
    : q.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function LatestTransactionsBox({ trades, loaded, loadError }: Props) {
  const router = useRouter();
  const goto = () => router.push("/paper-trading");

  return (
    <div className="flex h-[420px] flex-col rounded-xl border border-navy-700 bg-navy-800/40">
      <div className="flex items-center justify-between border-b border-navy-700 px-4 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Latest Transactions
        </h3>
        <Link href="/paper-trading" className="text-xs text-sky-400 hover:text-sky-300">
          View all →
        </Link>
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-auto scrollbar-thin">
        <table className="w-full table-fixed text-left text-sm" aria-label="Latest transactions">
          <thead className="border-b border-navy-700 bg-navy-800/60">
            <tr>
              <th className="w-[22%] truncate px-3 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Ticker</th>
              <th className="w-[18%] truncate px-3 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Side</th>
              <th className="w-[18%] truncate px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Qty</th>
              <th className="w-[28%] truncate px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Price</th>
              <th className="w-[14%] truncate px-3 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Time</th>
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
                <td className="truncate px-3 py-3 font-mono text-sm font-bold text-slate-100" title={t.ticker}>
                  <span className="inline-flex max-w-full items-center gap-1.5">
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${MARKET_DOT_CLASS[resolveMarket({ ticker: t.ticker })] ?? "bg-slate-400"}`}
                      aria-hidden="true"
                    />
                    <span className="truncate">{t.ticker}</span>
                  </span>
                </td>
                <td className="truncate px-3 py-3">
                  <span className={`inline-block max-w-full truncate rounded-full px-1.5 py-0.5 text-[10px] font-medium ${sideColor(t.action)}`}>
                    {t.action}
                  </span>
                </td>
                <td
                  className="truncate px-3 py-3 text-right font-mono text-sm text-slate-300"
                  title={fmtQtyFull(t.quantity)}
                >
                  {fmtQty(t.quantity)}
                </td>
                <td className="truncate px-3 py-3 text-right font-mono text-sm text-slate-300" title={fmtPrice(t.price, t.ticker)}>
                  {fmtPrice(t.price, t.ticker)}
                </td>
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
