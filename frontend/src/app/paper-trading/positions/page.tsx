"use client";

// phase-44.2 -- positions sub-route.
//
// Closes UX-DoD criteria 4 (positions_table_uses_DataTable + sort + filter),
// 6 (AgentRationaleDrawer opens from positions row), 7 (LiveBadge per row),
// and 8 (SectorBarList right column).

import { useCallback, useMemo } from "react";
import type { FilterFn } from "@tanstack/react-table";
import { DataTable } from "@/components/DataTable";
import { SectorBarList } from "@/components/SectorBarList";
import { PortfolioAllocationDonut } from "@/components/PortfolioAllocationDonut";
import { RiskMonitorCard } from "@/components/paper-trading/cockpit-helpers";
import { positionsColumns } from "@/components/paper-trading/positions-columns";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import { useLivePortfolio } from "@/lib/live-portfolio-context";
import { latestTradeIdForTicker } from "@/lib/paper-trading-utils";
import type { PaperPosition } from "@/lib/types";

// Hard-coded default cap; the operator-tunable setting is
// paper_max_per_sector_nav_pct (lives at /paper-trading/manage; reachable
// via the Settings gear button in the layout header).
// We don't fetch FullSettings at the layout level to avoid an extra
// round-trip on every cockpit page load; the cap rendered here will
// update when /manage refetches and the operator returns.
const DEFAULT_SECTOR_CAP_PCT = 30;

export default function PositionsPage() {
  const {
    positions,
    trades,
    perf,
    portfolio,
    tickerMeta,
    livePrices,
    openRationale,
  } = usePaperTradingData();
  // phase-72: pull live NAV + freshness from the root LivePortfolioProvider
  // so the donut center label matches every other NAV display.
  const lp = useLivePortfolio();

  const columns = useMemo(
    () => positionsColumns(tickerMeta, livePrices),
    [tickerMeta, livePrices],
  );

  // phase-44.2 cycle-68: custom global filter that matches ticker OR
  // company_name OR sector. Default TanStack auto-filter only inspects
  // primitive accessor values; the operator wants to search by company
  // (which lives in tickerMeta, NOT on the row). Closes over tickerMeta
  // so updates to ticker meta data re-create the filter.
  const positionsFilterFn = useCallback<FilterFn<PaperPosition>>(
    (row, _columnId, value) => {
      const q = String(value ?? "").trim().toLowerCase();
      if (!q) return true;
      const ticker = (row.original.ticker ?? "").toLowerCase();
      const meta = tickerMeta[row.original.ticker];
      const company = (meta?.company_name ?? "").toLowerCase();
      const sector = (meta?.sector ?? "").toLowerCase();
      return ticker.includes(q) || company.includes(q) || sector.includes(q);
    },
    [tickerMeta],
  );

  // Sector concentration items: aggregate live market value by sector,
  // normalized to NAV. Uses live prices when available, falls back to
  // stored cost basis so a fresh portfolio still renders.
  const sectorItems = useMemo(() => {
    const navDenom = portfolio?.total_nav ?? 10000;
    const acc = new Map<string, number>();
    for (const pos of positions) {
      const sector = tickerMeta[pos.ticker]?.sector || "Unknown";
      const livePrice = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
      const mv = livePrice * pos.quantity;
      acc.set(sector, (acc.get(sector) ?? 0) + mv);
    }
    return Array.from(acc.entries())
      .map(([name, mv]) => ({ name, value: navDenom > 0 ? (mv / navDenom) * 100 : 0 }))
      .filter((s) => s.value > 0);
  }, [positions, tickerMeta, livePrices, portfolio]);

  // phase-44.2 cycle-68: portfolio allocation slices for the donut.
  // Each held sector contributes its summed market value; cash is
  // surfaced as its own slice. Total = sum of all = NAV when consistent.
  const allocationSlices = useMemo(() => {
    const acc = new Map<string, number>();
    for (const pos of positions) {
      const sector = tickerMeta[pos.ticker]?.sector || "Unknown";
      const livePrice = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
      const mv = livePrice * pos.quantity;
      acc.set(sector, (acc.get(sector) ?? 0) + mv);
    }
    const slices = Array.from(acc.entries()).map(([name, value]) => ({ name, value }));
    const cash = portfolio?.current_cash ?? 0;
    if (cash > 0) slices.push({ name: "Cash", value: cash });
    return slices;
  }, [positions, tickerMeta, livePrices, portfolio]);

  return (
    <div
      role="tabpanel"
      id="panel-positions"
      aria-labelledby="tab-positions"
      tabIndex={0}
      className="space-y-4"
    >
      {/* phase-44.2 cycle-69 UX-audit fix: 3-col row -- Risk Monitor +
          Sector concentration + Portfolio allocation donut. items-stretch
          equalizes heights for visual alignment (operator-flagged 2026-05-26).
          Each card internally uses flex-col + content-grow so the contents
          fill the stretched height without dead whitespace. Collapses to
          1-col on small screens. */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3 items-stretch">
        <RiskMonitorCard
          perf={perf}
          positions={positions}
          portfolio={portfolio}
          tickerMeta={tickerMeta}
        />
        <SectorBarList
          items={sectorItems}
          capPct={DEFAULT_SECTOR_CAP_PCT}
          title="Sector concentration"
          emptyState="No positions yet."
        />
        <PortfolioAllocationDonut
          slices={allocationSlices}
          // phase-72: prefer live NAV (root SSOT) for the center label so the
          // donut matches the Home + Paper Trading NAV tiles. Falls back to
          // the persisted snapshot if the live derivation isn't ready
          // (initial paint, no live ticks).
          totalNav={lp.liveNav ?? portfolio?.total_nav ?? null}
          liveBand={lp.freshnessBand}
          liveAgeSec={lp.freshnessAgeSec}
          title="Allocation"
        />
      </div>
      <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg p-4">
        <DataTable
          data={positions}
          columns={columns}
          globalFilterPlaceholder="Filter ticker, company, or sector..."
          globalFilterFn={positionsFilterFn}
          ariaLabel="Positions"
          onRowClick={(pos) => {
            const tid = latestTradeIdForTicker(trades, pos.ticker);
            if (tid) openRationale(tid);
          }}
          emptyState="No open positions"
        />
      </div>
    </div>
  );
}
