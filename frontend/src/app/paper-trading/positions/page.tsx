"use client";

// phase-44.2 -- positions sub-route.
//
// Closes UX-DoD criteria 4 (positions_table_uses_DataTable + sort + filter),
// 6 (AgentRationaleDrawer opens from positions row), 7 (LiveBadge per row),
// and 8 (SectorBarList right column).

import { useMemo } from "react";
import { DataTable } from "@/components/DataTable";
import { SectorBarList } from "@/components/SectorBarList";
import { RiskMonitorCard } from "@/components/paper-trading/cockpit-helpers";
import { positionsColumns } from "@/components/paper-trading/positions-columns";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import { latestTradeIdForTicker } from "@/lib/paper-trading-utils";

// Hard-coded default cap; the operator-tunable setting is
// paper_max_per_sector_nav_pct (lives in /settings post-44.2 Manage removal).
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

  const columns = useMemo(
    () => positionsColumns(tickerMeta, livePrices),
    [tickerMeta, livePrices],
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

  return (
    <div
      role="tabpanel"
      id="panel-positions"
      aria-labelledby="tab-positions"
      tabIndex={0}
      className="space-y-4"
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <RiskMonitorCard
            perf={perf}
            positions={positions}
            portfolio={portfolio}
            tickerMeta={tickerMeta}
          />
          <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg p-4">
            <DataTable
              data={positions}
              columns={columns}
              globalFilterPlaceholder="Filter tickers..."
              ariaLabel="Positions"
              onRowClick={(pos) => {
                const tid = latestTradeIdForTicker(trades, pos.ticker);
                if (tid) openRationale(tid);
              }}
              emptyState="No open positions"
            />
          </div>
        </div>
        <div className="lg:col-span-1">
          <SectorBarList
            items={sectorItems}
            capPct={DEFAULT_SECTOR_CAP_PCT}
            title="Sector concentration"
            emptyState="No positions yet."
            className="bg-navy-800/70 border-navy-700"
          />
        </div>
      </div>
    </div>
  );
}
