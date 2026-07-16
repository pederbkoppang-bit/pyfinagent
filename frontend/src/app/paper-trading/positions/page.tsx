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
import { MultiCurrencyNavBreakdown } from "@/components/MultiCurrencyNavBreakdown";
import { RiskMonitorCard } from "@/components/paper-trading/cockpit-helpers";
import { positionsColumns } from "@/components/paper-trading/positions-columns";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import { useLivePortfolio } from "@/lib/live-portfolio-context";
import { latestTradeIdForTicker } from "@/lib/paper-trading-utils";
import { resolveMarket } from "@/lib/format";
import type { PaperPosition } from "@/lib/types";

// phase-70.1: DISPLAY-ONLY fallback, not the live cap. The operator-tunable
// setting is paper_max_per_sector_nav_pct, which is a risk_overrides
// ALLOWED_KEY edited via the "Risk limits (live overrides)" panel on
// /paper-trading/manage (NOT a .env settings-form field -- it is not in
// SettingsUpdate). This constant is only a static default used for rendering;
// it does NOT reflect an active runtime override. For the true enforced value,
// read effective_value from getRiskLimits() (GET /api/paper-trading/risk-limits).
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
    activeMarket,
  } = usePaperTradingData();
  // phase-72: pull live NAV + freshness from the root LivePortfolioProvider
  // so the donut center label matches every other NAV display.
  const lp = useLivePortfolio();

  const columns = useMemo(
    () => positionsColumns(tickerMeta, livePrices),
    [tickerMeta, livePrices],
  );

  // goal-multimarket-ux: scope the table + donut + sector bar to the active market.
  const isAllMarkets = !activeMarket || activeMarket === "ALL";
  const visiblePositions = useMemo(
    () =>
      isAllMarkets
        ? positions
        : positions.filter(
            (p) => resolveMarket({ market: p.market, ticker: p.ticker }) === activeMarket,
          ),
    [positions, activeMarket, isAllMarkets],
  );

  // Per-position USD market value. US keeps the exact legacy live formula
  // (livePrice ?? current_price ?? entry) x qty; non-US uses the backend USD
  // `market_value` (no client-side FX -- livePrice x qty would be LOCAL notional).
  const mvUsd = useCallback(
    (pos: PaperPosition): number => {
      const isUs = resolveMarket({ market: pos.market, ticker: pos.ticker }) === "US";
      if (isUs) {
        const px = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
        return px * pos.quantity;
      }
      return pos.market_value ?? 0;
    },
    [livePrices],
  );

  const filteredNavUsd = useMemo(
    () => visiblePositions.reduce((sum, p) => sum + mvUsd(p), 0),
    [visiblePositions, mvUsd],
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

  // Sector concentration items: aggregate USD market value by sector, normalized to
  // NAV. Scoped to the active market; denominator is the fund NAV for "All" or the
  // filtered market's USD holdings when a single market is selected (so its sectors
  // sum to ~100% within the market rather than reading as a tiny slice of the fund).
  const sectorItems = useMemo(() => {
    const navDenom = isAllMarkets
      ? (portfolio?.total_nav ?? 10000)
      : (filteredNavUsd || 1);
    const acc = new Map<string, number>();
    for (const pos of visiblePositions) {
      const sector = tickerMeta[pos.ticker]?.sector || "Unknown";
      acc.set(sector, (acc.get(sector) ?? 0) + mvUsd(pos));
    }
    return Array.from(acc.entries())
      .map(([name, mv]) => ({ name, value: navDenom > 0 ? (mv / navDenom) * 100 : 0 }))
      .filter((s) => s.value > 0);
  }, [visiblePositions, tickerMeta, mvUsd, isAllMarkets, portfolio, filteredNavUsd]);

  // phase-44.2 cycle-68: portfolio allocation slices for the donut. Each held sector
  // contributes its summed USD market value. Cash is a fund-level slice, so it is only
  // shown for the combined "All" view (a single-market view shows that market's sector
  // mix without the fund's cash).
  const allocationSlices = useMemo(() => {
    const acc = new Map<string, number>();
    for (const pos of visiblePositions) {
      const sector = tickerMeta[pos.ticker]?.sector || "Unknown";
      acc.set(sector, (acc.get(sector) ?? 0) + mvUsd(pos));
    }
    const slices = Array.from(acc.entries()).map(([name, value]) => ({ name, value }));
    const cash = portfolio?.current_cash ?? 0;
    if (isAllMarkets && cash > 0) slices.push({ name: "Cash", value: cash });
    return slices;
  }, [visiblePositions, tickerMeta, mvUsd, isAllMarkets, portfolio]);

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
          // (initial paint, no live ticks). goal-multimarket-ux: when a single
          // market is selected, the center shows that market's USD holdings.
          totalNav={isAllMarkets ? (lp.liveNav ?? portfolio?.total_nav ?? null) : filteredNavUsd}
          liveBand={lp.freshnessBand}
          liveAgeSec={lp.freshnessAgeSec}
          title={isAllMarkets ? "Allocation" : `Allocation — ${activeMarket}`}
        />
      </div>
      {/* phase-50.6: multi-currency NAV breakdown -- fund NAV (USD base) split by
          the local currency of each held market. Client-side from /portfolio;
          scoped to the active market filter. Cash (USD base) only on the All view. */}
      <MultiCurrencyNavBreakdown
        positions={visiblePositions}
        totalNav={isAllMarkets ? (lp.liveNav ?? portfolio?.total_nav ?? null) : filteredNavUsd}
        cashUsd={isAllMarkets ? (portfolio?.current_cash ?? null) : null}
        title={isAllMarkets ? "Currency exposure" : `Currency exposure — ${activeMarket}`}
      />
      <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg p-4">
        <DataTable
          data={visiblePositions}
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
