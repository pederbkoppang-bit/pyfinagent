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
import { clsx } from "clsx";
import {
  ageSecFromIso,
  bandFromAgeSec,
  bandFromMarkAgeSec,
  type FreshnessBand,
} from "@/lib/paper-trading-utils";
import { Dollar, MarketChip, PnlBadge } from "./cockpit-helpers";
// goal-multimarket-ux: per-share ENTRY/CURRENT/STOP are LOCAL currency; MARKET-VALUE
// and P&L% stay USD/backend (no client-side FX). resolveMarket==='US' is both the
// do-no-harm guard (byte-identical) AND the no-FX guard (don't mix local*qty with USD).
import {
  formatCurrency,
  numberFlowFormat,
  numberFlowLocale,
  resolveLocalCurrency,
  resolveMarket,
} from "@/lib/format";
// phase-76 (2026-05-26): trend tracker for the data-pyfa-trend host
// attribute. globals.css targets number-flow-react[data-pyfa-trend="up"]
// ::part(digit) for color tint on changing digits. (Cycle 77 bugfix:
// the lib's React wrapper renders <number-flow-react>, not
// <number-flow> -- cycle 76 had the wrong element name in the CSS.)
import { useTrend } from "@/lib/use-trend";

// phase-61.3: the as-of chip on a non-live P&L. Static class strings (Tailwind JIT
// cannot see interpolated names), same palette as LiveBadge so the two freshness
// signals read as one system.
const MARK_BAND_CLASS: Record<FreshnessBand, string> = {
  green: "bg-slate-800 text-slate-400",
  amber: "bg-amber-950 text-amber-400",
  red: "bg-rose-950 text-rose-400",
  unknown: "bg-slate-800 text-slate-500",
};

// Compact age for the chip: "2h", "3d". Marks are cycle-scale, so minutes are noise
// and anything under an hour reads as "just marked".
export function formatMarkAge(ageSec: number | null): string {
  if (ageSec == null) return "as of ?";
  const hours = ageSec / 3600;
  if (hours < 1) return "as of now";
  if (hours < 48) return `as of ${Math.floor(hours)}h`;
  return `as of ${Math.floor(hours / 24)}d`;
}

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
  // phase-61.3: `locales` must ALWAYS be explicit. NumberFlow's own docs: "When
  // omitted, the component will use the browser's default locale" -- so the USD
  // branch used to render in the operator's nb-NO locale ("70 000,00 USD") right
  // beside an en-US "$70000.00" from the sibling cells. numberFlowLocale("USD")
  // returns "en-US", so this is the same output on an en-US browser and a FIX on
  // any other. The USD format object keeps minimumFractionDigits:2 (a USD-only
  // convention -- generalising it would render "₩1,234,567.00").
  const locales = numberFlowLocale(cur);
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
          locales={locales}
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
        // phase-61.3: avg_entry_price is LOCAL, so the currency comes from the
        // market -- never from base_currency, which describes the USD columns.
        const cur = resolveLocalCurrency({
          market: row.original.market,
          ticker: row.original.ticker,
        });
        return (
          <span className="text-slate-100">
            {formatCurrency(row.original.avg_entry_price, cur)}
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
        // Live price + stored current_price are LOCAL currency (phase-50.2), so
        // phase-61.3 resolves them market-first rather than from base_currency.
        const cur = resolveLocalCurrency({
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
        const isLive = isUs && livePrice != null && liveCostBasis > 0;
        const livePnlPct = isLive
          ? ((livePrice! * pos.quantity - liveCostBasis) / liveCostBasis) * 100
          : pos.unrealized_pnl_pct;
        // phase-61.3: when the number is NOT live-recomputed it is the stored mark,
        // which is as old as the last mark_to_market run -- label it with that time
        // instead of letting it sit next to a live price implying it is current.
        const markAgeSec = ageSecFromIso(pos.marked_at);
        return (
          <span className="inline-flex items-center justify-end gap-2">
            <PnlBadge value={livePnlPct} />
            {!isLive && pos.marked_at ? (
              <span
                title={`Marked ${new Date(pos.marked_at).toLocaleString("en-US", {
                  timeZone: "UTC",
                  timeZoneName: "short",
                })} — P&L is as of that mark, not live`}
                className={clsx(
                  "rounded px-1 text-[10px] font-medium uppercase tracking-wide",
                  MARK_BAND_CLASS[bandFromMarkAgeSec(markAgeSec)],
                )}
              >
                {formatMarkAge(markAgeSec)}
              </span>
            ) : null}
          </span>
        );
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
        // phase-61.3: the stop is compared against a LOCAL price by
        // paper_trader.check_stop_losses, so it must render in the LOCAL currency.
        const cur = resolveLocalCurrency({
          market: row.original.market,
          ticker: row.original.ticker,
        });
        return (
          <span className="text-slate-300">{formatCurrency(sl, cur)}</span>
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
