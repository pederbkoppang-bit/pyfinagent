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
import { Dollar, MarketChip, PnlBadge } from "./cockpit-helpers";
// goal-multimarket-ux: per-share ENTRY/CURRENT/STOP are LOCAL currency; MARKET-VALUE
// and P&L% stay USD/backend (no client-side FX). resolveMarket==='US' is both the
// do-no-harm guard (byte-identical) AND the no-FX guard (don't mix local*qty with USD).
import {
  formatCurrency,
  numberFlowFormat,
  numberFlowLocale,
  resolveCurrency,
  resolveMarket,
} from "@/lib/format";
// phase-76 (2026-05-26): trend tracker for the data-pyfa-trend host
// attribute. globals.css targets number-flow-react[data-pyfa-trend="up"]
// ::part(digit) for color tint on changing digits. (Cycle 77 bugfix:
// the lib's React wrapper renders <number-flow-react>, not
// <number-flow> -- cycle 76 had the wrong element name in the CSS.)
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
  currency = "USD",
}: {
  shown: number | null | undefined;
  band: ReturnType<typeof bandFromAgeSec>;
  ageSec: number | null;
  currency?: string;
}) {
  const trend = useTrend(shown);
  const cur = (currency || "USD").toUpperCase();
  const isUsd = cur === "USD";
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
          format={
            isUsd
              ? {
                  style: "currency",
                  currency: "USD",
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                }
              : numberFlowFormat(cur)
          }
          locales={isUsd ? undefined : numberFlowLocale(cur)}
          transformTiming={{ duration: 900 }}
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
      cell: ({ row }) => {
        const cur = resolveCurrency({
          baseCurrency: row.original.base_currency,
          market: row.original.market,
          ticker: row.original.ticker,
        });
        return (
          <span className="text-slate-100">
            {cur === "USD"
              ? `$${row.original.avg_entry_price.toFixed(2)}`
              : formatCurrency(row.original.avg_entry_price, cur)}
          </span>
        );
      },
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
        // Live price + stored current_price are LOCAL currency (phase-50.2).
        const cur = resolveCurrency({
          baseCurrency: pos.base_currency,
          market: pos.market,
          ticker: pos.ticker,
        });
        return (
          <CurrentPriceCell
            shown={shown}
            band={band}
            ageSec={live?.age_sec ?? null}
            currency={cur}
          />
        );
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "market_value",
      // Market value is USD. The live recompute `livePrice * quantity` is LOCAL
      // notional, so it is ONLY valid for US (local==USD). For non-US fall back to
      // the backend's USD `market_value` -- never multiply local price by qty and
      // label it USD (no client-side FX; do-no-harm for US stays exact).
      accessorFn: (row) => {
        const isUs = resolveMarket({ market: row.market, ticker: row.ticker }) === "US";
        const live = livePrices[row.ticker];
        const livePrice = live?.price;
        return isUs && livePrice != null
          ? livePrice * row.quantity
          : (row.market_value ?? 0);
      },
      header: "Market Value",
      cell: ({ row }) => {
        const pos = row.original;
        const isUs = resolveMarket({ market: pos.market, ticker: pos.ticker }) === "US";
        const live = livePrices[pos.ticker];
        const livePrice = live?.price ?? null;
        const liveMarketValue =
          isUs && livePrice != null ? livePrice * pos.quantity : pos.market_value;
        return <Dollar value={liveMarketValue} />;
      },
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "pnl",
      // P&L% mixes a price (local) against cost_basis (USD); the live recompute is
      // only currency-consistent for US. Non-US uses the backend's USD-consistent
      // unrealized_pnl_pct (no client-side FX).
      accessorFn: (row) => {
        const isUs = resolveMarket({ market: row.market, ticker: row.ticker }) === "US";
        const live = livePrices[row.ticker];
        const livePrice = live?.price ?? null;
        const liveCostBasis =
          row.cost_basis != null && row.cost_basis > 0
            ? row.cost_basis
            : row.avg_entry_price * row.quantity;
        if (isUs && livePrice != null && liveCostBasis > 0) {
          return ((livePrice * row.quantity - liveCostBasis) / liveCostBasis) * 100;
        }
        return row.unrealized_pnl_pct ?? 0;
      },
      header: "P&L",
      cell: ({ row }) => {
        const pos = row.original;
        const isUs = resolveMarket({ market: pos.market, ticker: pos.ticker }) === "US";
        const live = livePrices[pos.ticker];
        const livePrice = live?.price ?? null;
        const liveCostBasis =
          pos.cost_basis != null && pos.cost_basis > 0
            ? pos.cost_basis
            : pos.avg_entry_price * pos.quantity;
        const livePnlPct =
          isUs && livePrice != null && liveCostBasis > 0
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
      cell: ({ row }) => {
        const sl = row.original.stop_loss_price;
        if (sl == null) return <span className="text-slate-300">—</span>;
        const cur = resolveCurrency({
          baseCurrency: row.original.base_currency,
          market: row.original.market,
          ticker: row.original.ticker,
        });
        return (
          <span className="text-slate-300">
            {cur === "USD" ? `$${sl.toFixed(2)}` : formatCurrency(sl, cur)}
          </span>
        );
      },
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
