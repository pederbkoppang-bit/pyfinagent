/**
 * phase-10.5.2 /sovereign route shell.
 *
 * Two-hero layout placeholder. The three tile components arrive in
 * later phase-10.5 sub-steps:
 *   - RedLineMonitor       (10.5.3)
 *   - ComputeCostBreakdown (10.5.4)
 *   - AlphaLeaderboard     (10.5.5)
 *
 * Backend data already shipping (phase-10.5.0 + 10.5.1):
 *   GET /api/sovereign/red-line / leaderboard / compute-cost
 *
 * This page is intentionally a thin shell so the sidebar entry +
 * route-anchor land cleanly. Empty-state cards label what's coming.
 */
"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { RedLineMonitor } from "@/components/RedLineMonitor";
import type { RedLineWindow } from "@/components/RedLineMonitor";
import { ComputeCostBreakdown } from "@/components/ComputeCostBreakdown";
import type { ProviderCostPoint } from "@/components/ComputeCostBreakdown";
import { AlphaLeaderboard } from "@/components/AlphaLeaderboard";
import {
  getSovereignRedLine,
  getSovereignComputeCost,
  getSovereignLeaderboard,
} from "@/lib/api";
import type { SovereignLeaderboardEntry } from "@/lib/api";
import {
  Crown,
} from "@phosphor-icons/react";


export default function SovereignPage() {
  // phase-10.5.3 RedLine state owned by the page; component is props-driven.
  const [redLineWindow, setRedLineWindow] = useState<RedLineWindow>("30d");
  const [redLineSeries, setRedLineSeries] = useState<
    { date: string; nav: number; source: string }[]
  >([]);
  const [redLineEvents, setRedLineEvents] = useState<
    { date: string; label: string; detail?: string | null }[]
  >([]);

  // phase-10.5.4 ComputeCostBreakdown state (30d hardcoded; no selector required).
  const [costData, setCostData] = useState<ProviderCostPoint[]>([]);
  const [costGrandTotal, setCostGrandTotal] = useState<number>(0);

  // phase-10.5.5 AlphaLeaderboard state.
  const [leaderboard, setLeaderboard] = useState<SovereignLeaderboardEntry[]>([]);
  const [leaderboardError, setLeaderboardError] = useState<string | null>(null);
  const [leaderboardLoading, setLeaderboardLoading] = useState<boolean>(true);

  useEffect(() => {
    let ignore = false;
    getSovereignRedLine(redLineWindow)
      .then((d) => {
        if (ignore) return;
        setRedLineSeries(d.series ?? []);
        setRedLineEvents(d.events ?? []);
      })
      .catch(() => {
        if (ignore) return;
        setRedLineSeries([]);
        setRedLineEvents([]);
      });
    return () => {
      ignore = true;
    };
  }, [redLineWindow]);

  useEffect(() => {
    let ignore = false;
    getSovereignComputeCost("30d")
      .then((d) => {
        if (ignore) return;
        setCostData(d.daily_breakdown ?? []);
        setCostGrandTotal(d.grand_total_usd ?? 0);
      })
      .catch(() => {
        if (ignore) return;
        setCostData([]);
        setCostGrandTotal(0);
      });
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    let ignore = false;
    setLeaderboardLoading(true);
    getSovereignLeaderboard()
      .then((d) => {
        if (ignore) return;
        setLeaderboard(d.entries ?? []);
        setLeaderboardError(null);
      })
      .catch((e: Error) => {
        if (ignore) return;
        setLeaderboard([]);
        setLeaderboardError(e.message || "Failed to load leaderboard");
      })
      .finally(() => {
        if (!ignore) setLeaderboardLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* ── Fixed header zone ── */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="flex items-center gap-2 text-2xl font-bold text-slate-100">
                <Crown size={24} className="text-amber-400" weight="fill" />
                Sovereign
              </h2>
              <p className="text-sm text-slate-500">
                Live trading control plane: red-line monitor, alpha
                leaderboard, and compute-cost breakdown.
              </p>
            </div>
          </div>
        </div>

        {/* ── Scrollable content zone ── */}
        <div
          data-sovereign-content
          className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8"
        >
          {/* Two-hero row: RedLine (3/5) + Leaderboard (2/5) */}
          <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-5">
            <div className="lg:col-span-3">
              <RedLineMonitor
                series={redLineSeries}
                events={redLineEvents}
                window={redLineWindow}
                onWindowChange={setRedLineWindow}
              />
            </div>
            <div className="lg:col-span-2">
              <AlphaLeaderboard
                entries={leaderboard}
                loading={leaderboardLoading}
                error={leaderboardError}
              />
            </div>
          </div>

          {/* Full-width compute-cost row (phase-10.5.4) */}
          <ComputeCostBreakdown
            data={costData}
            grandTotal={costGrandTotal}
            window="30d"
          />
        </div>
      </main>
    </div>
  );
}
