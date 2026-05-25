"use client";

// phase-44.2 -- reality-gap sub-route.
// Owns the reconciliation fetch locally (only relevant to this tab).

import { useEffect, useState } from "react";
import { PaperReconciliationChart } from "@/components/PaperReconciliationChart";
import { PaperVsBacktestCard } from "@/components/paper-trading/cockpit-helpers";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import { getPaperReconciliation } from "@/lib/api";
import type { PaperReconciliation } from "@/lib/types";

export default function RealityGapPage() {
  const { perf, snapshots } = usePaperTradingData();
  const [reconciliation, setReconciliation] = useState<PaperReconciliation | null>(null);
  const [reconciliationLoading, setReconciliationLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setReconciliationLoading(true);
    getPaperReconciliation()
      .then((r) => {
        if (!cancelled) setReconciliation(r);
      })
      .catch(() => {
        if (!cancelled) setReconciliation(null);
      })
      .finally(() => {
        if (!cancelled) setReconciliationLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div
      role="tabpanel"
      id="panel-reality-gap"
      aria-labelledby="tab-reality-gap"
      tabIndex={0}
      className="space-y-4"
    >
      <PaperVsBacktestCard perf={perf} snapshotsLen={snapshots.length} />
      <PaperReconciliationChart
        reconciliation={reconciliation}
        loading={reconciliationLoading}
      />
    </div>
  );
}
