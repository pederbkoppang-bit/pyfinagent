"use client";

// phase-44.2 -- trades sub-route.
//
// Closes UX-DoD criterion 5 (trades_table_uses_DataTable) and second half
// of criterion 6 (AgentRationaleDrawer opens from trades row click).

import { useCallback, useMemo } from "react";
import type { FilterFn } from "@tanstack/react-table";
import { DataTable } from "@/components/DataTable";
import { tradesColumns } from "@/components/paper-trading/trades-columns";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import type { PaperTrade } from "@/lib/types";

export default function TradesPage() {
  const { trades, tickerMeta, openRationale } = usePaperTradingData();

  const columns = useMemo(() => tradesColumns(tickerMeta), [tickerMeta]);

  // phase-44.2 cycle-69: same multi-field filter pattern as the positions
  // table -- match ticker OR company_name (from tickerMeta) OR sector
  // OR action OR reason. Operator-flagged 2026-05-26 that the original
  // single-field "Filter tickers..." was too restrictive.
  const tradesFilterFn = useCallback<FilterFn<PaperTrade>>(
    (row, _columnId, value) => {
      const q = String(value ?? "").trim().toLowerCase();
      if (!q) return true;
      const t = row.original;
      const ticker = (t.ticker ?? "").toLowerCase();
      const meta = tickerMeta[t.ticker];
      const company = (meta?.company_name ?? "").toLowerCase();
      const sector = (meta?.sector ?? "").toLowerCase();
      const action = (t.action ?? "").toLowerCase();
      const reason = (t.reason ?? "").toLowerCase();
      return (
        ticker.includes(q) ||
        company.includes(q) ||
        sector.includes(q) ||
        action.includes(q) ||
        reason.includes(q)
      );
    },
    [tickerMeta],
  );

  return (
    <div
      role="tabpanel"
      id="panel-trades"
      aria-labelledby="tab-trades"
      tabIndex={0}
    >
      <div className="rounded-xl border border-navy-700 bg-navy-800/70 backdrop-blur-lg p-4">
        <DataTable
          data={trades}
          columns={columns}
          globalFilterPlaceholder="Filter ticker, company, sector, action, or reason..."
          globalFilterFn={tradesFilterFn}
          ariaLabel="Trades"
          onRowClick={(t) => openRationale(t.trade_id)}
          emptyState="No trades yet"
        />
      </div>
    </div>
  );
}
