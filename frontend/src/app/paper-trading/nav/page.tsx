"use client";

// phase-44.2 -- NAV chart sub-route (verbatim port of monolith lines 969-1027).
// phase-73 (2026-05-26): chart-side SSOT overlay -- append live "today"
// row pct = (liveNav - starting_capital) / starting_capital * 100 so the
// chart ends at the live value instead of forward-filling the stale
// snapshot. Mirrors RedLineMonitor cycle-73 logic; SPY + Alpha lines
// stay at the last actual snapshot (they are historical by definition).

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { usePaperTradingData } from "@/lib/paper-trading-context";
import { useLivePortfolio } from "@/lib/live-portfolio-context";

export default function NavChartPage() {
  const { snapshots } = usePaperTradingData();
  const lp = useLivePortfolio();

  const chartData = useMemo(() => {
    const base = [...snapshots].reverse().map((s) => ({
      date: s.snapshot_date,
      nav: s.total_nav,
      portfolio: s.cumulative_pnl_pct,
      benchmark: s.benchmark_pnl_pct,
      alpha: s.alpha_pct,
      source: "snapshot" as string,
    }));
    // phase-73 overlay: append live "today" row when liveNav available AND
    // today > last snapshot date. Portfolio pct is (liveNav - start) / start;
    // SPY + Alpha carry the last snapshot (historical-only series).
    const todayIso = new Date().toISOString().slice(0, 10);
    const last = base.length > 0 ? base[base.length - 1] : null;
    const startingCap = lp.status?.portfolio?.starting_capital ?? null;
    if (
      lp.liveNav != null &&
      startingCap != null &&
      startingCap > 0 &&
      last != null &&
      last.date != null &&
      String(last.date) < todayIso
    ) {
      const livePct = ((lp.liveNav - startingCap) / startingCap) * 100;
      base.push({
        date: todayIso,
        nav: lp.liveNav,
        portfolio: livePct,
        benchmark: last.benchmark,
        alpha: livePct - (last.benchmark ?? 0),
        source: "live_now",
      });
    }
    return base;
  }, [snapshots, lp.liveNav, lp.status]);

  return (
    <div
      role="tabpanel"
      id="panel-nav"
      aria-labelledby="tab-nav"
      tabIndex={0}
      className="rounded-xl border border-navy-700 bg-navy-800/70 p-6"
    >
      {chartData.length < 2 ? (
        <p className="py-8 text-center text-slate-500">
          Need at least 2 days of data for charting.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(5)}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickFormatter={(v: number) => `${v.toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                border: "1px solid #334155",
                borderRadius: 8,
              }}
              labelStyle={{ color: "#94a3b8" }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="portfolio"
              name="Portfolio"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              name="SPY"
              stroke="#64748b"
              strokeWidth={1.5}
              strokeDasharray="5 5"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="alpha"
              name="Alpha"
              stroke="#22c55e"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
