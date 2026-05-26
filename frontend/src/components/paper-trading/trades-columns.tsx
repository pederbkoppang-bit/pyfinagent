"use client";

// phase-44.2 -- TanStack v8 column factory for the trades DataTable.

import type { ColumnDef } from "@tanstack/react-table";
import { clsx } from "clsx";
import type { PaperTrade } from "@/lib/types";
import type { TickerMeta } from "@/lib/paper-trading-context";
import { Dollar } from "./cockpit-helpers";

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
      cell: ({ row }) => (
        <span className="text-slate-100">${row.original.price.toFixed(2)}</span>
      ),
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
