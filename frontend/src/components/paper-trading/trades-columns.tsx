"use client";

// phase-44.2 -- TanStack v8 column factory for the trades DataTable.

import type { ColumnDef } from "@tanstack/react-table";
import { clsx } from "clsx";
import type { PaperTrade } from "@/lib/types";
import type { TickerMeta } from "@/lib/paper-trading-context";
import { Dollar, MarketChip } from "./cockpit-helpers";
// goal-multimarket-ux: trades carry no market column, so market is derived from the
// ticker suffix. `price` is LOCAL currency; `total_value`/`transaction_cost` are USD
// for rows written on/after the 2026-06-10 phase-56.1 fix (55.1 F-2). CAVEAT: the 7
// KR rows written 2026-06-01..2026-06-09 hold LOCAL (KRW) magnitudes in those two
// fields until the operator-gated backfill (scripts/migrations/
// backfill_56_1_kr_trade_values.py) is approved and executed.
import { formatCurrency, resolveCurrency, resolveMarket } from "@/lib/format";

export function tradesColumns(
  tickerMeta: Record<string, TickerMeta>,
): ColumnDef<PaperTrade, unknown>[] {
  return [
    {
      id: "date",
      accessorKey: "created_at",
      header: "Date",
      cell: ({ row }) => (
        <span className="text-xs text-slate-500">
          {new Date(row.original.created_at).toLocaleDateString()}
        </span>
      ),
      meta: { align: "left" },
    },
    {
      id: "action",
      accessorKey: "action",
      header: "Action",
      cell: ({ row }) => (
        <span
          className={clsx(
            "rounded px-2 py-0.5 text-xs font-medium",
            row.original.action === "BUY"
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-rose-500/10 text-rose-400",
          )}
        >
          {row.original.action}
        </span>
      ),
      meta: { align: "left" },
    },
    {
      id: "ticker",
      accessorKey: "ticker",
      header: "Ticker",
      cell: ({ row }) => (
        <span className="font-mono font-semibold text-slate-100">{row.original.ticker}</span>
      ),
      meta: { align: "left" },
    },
    {
      id: "market",
      accessorFn: (row) => resolveMarket({ market: row.market, ticker: row.ticker }),
      header: "Market",
      cell: ({ row }) => (
        <MarketChip market={row.original.market} ticker={row.original.ticker} showExchange />
      ),
      meta: { align: "left" },
    },
    {
      id: "company",
      accessorFn: (row) => tickerMeta[row.ticker]?.company_name ?? "",
      header: "Company",
      cell: ({ row }) => (
        <span className="text-xs text-slate-400">
          {tickerMeta[row.original.ticker]?.company_name ?? "—"}
        </span>
      ),
      meta: { align: "left" },
    },
    {
      id: "qty",
      accessorKey: "quantity",
      header: "Qty",
      cell: ({ row }) => (
        <span className="text-slate-100">{row.original.quantity.toFixed(2)}</span>
      ),
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "price",
      accessorKey: "price",
      header: "Price",
      cell: ({ row }) => {
        const cur = resolveCurrency({
          currency: row.original.currency,
          market: row.original.market,
          ticker: row.original.ticker,
        });
        return (
          <span className="text-slate-100">
            {cur === "USD"
              ? `$${row.original.price.toFixed(2)}`
              : formatCurrency(row.original.price, cur)}
          </span>
        );
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "value",
      accessorKey: "total_value",
      header: "Value",
      cell: ({ row }) => <Dollar value={row.original.total_value} />,
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "fee",
      accessorKey: "transaction_cost",
      header: "Fee",
      cell: ({ row }) => (
        <span className="text-slate-400">
          {row.original.transaction_cost != null
            ? `$${row.original.transaction_cost.toFixed(2)}`
            : "—"}
        </span>
      ),
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "reason",
      accessorKey: "reason",
      header: "Reason",
      cell: ({ row }) => (
        <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
          {row.original.reason ?? "—"}
        </span>
      ),
      meta: { align: "left" },
    },
  ];
}
