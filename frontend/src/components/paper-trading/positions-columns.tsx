"use client";

// phase-44.2 -- TanStack v8 column factory for the positions DataTable.
//
// Numeric columns right-align per Tufte/Cleveland-McGill position-encoding
// principle (frontend-layout.md Section 9). Live price + per-row freshness
// badge (LiveBadge compact) is the visible "live or stale" signal that
// criterion 7 calls for.

import type { ColumnDef } from "@tanstack/react-table";
import NumberFlow from "@number-flow/react";
import type { PaperPosition } from "@/lib/types";
import { LiveBadge } from "@/components/LiveBadge";
import type { LivePriceEntry, TickerMeta } from "@/lib/paper-trading-context";
import { bandFromAgeSec } from "@/lib/paper-trading-utils";
import { Dollar, PnlBadge } from "./cockpit-helpers";
// phase-76 (2026-05-26): trend tracker for the data-pyfa-trend host
// attribute. globals.css targets number-flow[data-pyfa-trend="up"]
// ::part(digit) for color tint on changing digits.
import { useTrend } from "@/lib/use-trend";

// phase-75 (2026-05-26): Google-Finance digit-flip via NumberFlow. Per-row
// Current cell stays its own component so React's render path is clean
// (NumberFlow's internal hooks live inside the component, no rules-of-hooks
// boundary concern). Market Value + P&L cells inherit NumberFlow via
// Dollar + PnlBadge.
function CurrentPriceCell({
  shown,
  band,
  ageSec,
}: {
  shown: number | null | undefined;
  band: ReturnType<typeof bandFromAgeSec>;
  ageSec: number | null;
}) {
  const trend = useTrend(shown);
  return (
    <span
      aria-live="off"
      className="inline-flex items-center justify-end gap-2 text-slate-100"
    >
      <LiveBadge band={band} ageSec={ageSec} compact />
      {shown == null ? (
        <span className="text-slate-500">—</span>
      ) : (
        <NumberFlow
          value={shown}
          format={{
            style: "currency",
            currency: "USD",
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          }}
          transformTiming={{ duration: 700 }}
          willChange
          data-pyfa-trend={trend}
          className="tabular-nums"
        />
      )}
    </span>
  );
}

export function positionsColumns(
  tickerMeta: Record<string, TickerMeta>,
  livePrices: Record<string, LivePriceEntry>,
): ColumnDef<PaperPosition, unknown>[] {
  return [
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
      id: "sector",
      accessorFn: (row) => tickerMeta[row.ticker]?.sector ?? "",
      header: "Sector",
      cell: ({ row }) => (
        <span className="text-xs text-slate-400">
          {tickerMeta[row.original.ticker]?.sector || "—"}
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
      id: "entry",
      accessorKey: "avg_entry_price",
      header: "Entry",
      cell: ({ row }) => (
        <span className="text-slate-100">${row.original.avg_entry_price.toFixed(2)}</span>
      ),
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "current",
      accessorFn: (row) => livePrices[row.ticker]?.price ?? row.current_price ?? 0,
      header: "Current",
      cell: ({ row }) => {
        const pos = row.original;
        const live = livePrices[pos.ticker];
        const shown = live?.price ?? pos.current_price;
        const band = bandFromAgeSec(live?.age_sec ?? null);
        return (
          <CurrentPriceCell shown={shown} band={band} ageSec={live?.age_sec ?? null} />
        );
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "market_value",
      accessorFn: (row) => {
        const live = livePrices[row.ticker];
        const livePrice = live?.price;
        return livePrice != null ? livePrice * row.quantity : (row.market_value ?? 0);
      },
      header: "Market Value",
      cell: ({ row }) => {
        const pos = row.original;
        const live = livePrices[pos.ticker];
        const livePrice = live?.price ?? null;
        const liveMarketValue =
          livePrice != null ? livePrice * pos.quantity : pos.market_value;
        return <Dollar value={liveMarketValue} />;
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "pnl",
      accessorFn: (row) => {
        const live = livePrices[row.ticker];
        const livePrice = live?.price ?? null;
        const liveCostBasis =
          row.cost_basis != null && row.cost_basis > 0
            ? row.cost_basis
            : row.avg_entry_price * row.quantity;
        if (livePrice != null && liveCostBasis > 0) {
          return ((livePrice * row.quantity - liveCostBasis) / liveCostBasis) * 100;
        }
        return row.unrealized_pnl_pct ?? 0;
      },
      header: "P&L",
      cell: ({ row }) => {
        const pos = row.original;
        const live = livePrices[pos.ticker];
        const livePrice = live?.price ?? null;
        const liveCostBasis =
          pos.cost_basis != null && pos.cost_basis > 0
            ? pos.cost_basis
            : pos.avg_entry_price * pos.quantity;
        const livePnlPct =
          livePrice != null && liveCostBasis > 0
            ? ((livePrice * pos.quantity - liveCostBasis) / liveCostBasis) * 100
            : pos.unrealized_pnl_pct;
        return <PnlBadge value={livePnlPct} />;
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "stop_loss",
      accessorKey: "stop_loss_price",
      header: "Stop Loss",
      cell: ({ row }) => (
        <span className="text-slate-300">
          {row.original.stop_loss_price != null
            ? `$${row.original.stop_loss_price.toFixed(2)}`
            : "—"}
        </span>
      ),
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "days_held",
      accessorFn: (row) =>
        row.entry_date
          ? Math.floor((Date.now() - new Date(row.entry_date).getTime()) / 86_400_000)
          : 0,
      header: "Days Held",
      cell: ({ row }) => {
        const daysHeld = row.original.entry_date
          ? Math.floor(
              (Date.now() - new Date(row.original.entry_date).getTime()) / 86_400_000,
            )
          : 0;
        return <span className="text-slate-400">{daysHeld}d</span>;
      },
      meta: { align: "right", className: "tabular-nums" },
    },
  ];
}
