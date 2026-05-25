"use client";

// phase-44.2 -- trades sub-route.
//
// Closes UX-DoD criterion 5 (trades_table_uses_DataTable) and second half
// of criterion 6 (AgentRationaleDrawer opens from trades row click).

import { useMemo } from "react";
import { DataTable } from "@/components/DataTable";
import { tradesColumns } from "@/components/paper-trading/trades-columns";
import { usePaperTradingData } from "@/lib/paper-trading-context";

export default function TradesPage() {
  const { trades, tickerMeta, openRationale } = usePaperTradingData();

  const columns = useMemo(() => tradesColumns(tickerMeta), [tickerMeta]);

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
          globalFilterPlaceholder="Filter tickers..."
          ariaLabel="Trades"
          onRowClick={(t) => openRationale(t.trade_id)}
          emptyState="No trades yet"
        />
      </div>
    </div>
  );
}
