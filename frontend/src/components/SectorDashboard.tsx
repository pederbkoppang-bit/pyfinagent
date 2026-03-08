"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { SectorData } from "@/lib/types";

const PERIOD_LABELS: Record<string, string> = {
  "1mo": "1M",
  "3mo": "3M",
  "6mo": "6M",
  "1y": "1Y",
};

export function SectorDashboard({ data }: { data: SectorData }) {
  // Build sector rotation bar chart
  const sectorBars = Object.entries(data.sector_performance || {})
    .map(([name, ret]) => ({ name, return: ret }))
    .sort((a, b) => b.return - a.return);

  // Build relative strength table
  const periods = ["1mo", "3mo", "6mo", "1y"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-slate-100">
              {data.company_name || data.ticker}
            </h3>
            <p className="text-sm text-slate-500">
              {data.sector} · {data.industry}
              {data.sector_etf && ` · ETF: ${data.sector_etf}`}
            </p>
          </div>
          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold ${
              data.signal === "DOUBLE_TAILWIND"
                ? "bg-emerald-500/10 text-emerald-400"
                : data.signal === "LAGGING"
                  ? "bg-rose-500/10 text-rose-400"
                  : "bg-amber-500/10 text-amber-400"
            }`}
          >
            {data.signal}
          </span>
        </div>
      </div>

      {/* Relative Strength Table */}
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
        <h4 className="mb-4 text-sm font-medium text-slate-300">
          Relative Performance
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-700 text-left text-xs text-slate-500">
                <th className="pb-2">Period</th>
                <th className="pb-2">Stock</th>
                <th className="pb-2">Sector</th>
                <th className="pb-2">S&P 500</th>
                <th className="pb-2">vs Sector</th>
                <th className="pb-2">vs Market</th>
              </tr>
            </thead>
            <tbody>
              {periods.map((p) => {
                const stock = data.stock_returns?.[p];
                const sector = data.sector_returns?.[p];
                const spy = data.spy_returns?.[p];
                const vsSector = data.relative_vs_sector?.[p];
                const vsMarket = data.relative_vs_market?.[p];
                return (
                  <tr key={p} className="border-b border-navy-700/50">
                    <td className="py-2 font-mono text-slate-400">
                      {PERIOD_LABELS[p]}
                    </td>
                    <td className={numColor(stock)}>{fmt(stock)}</td>
                    <td className={numColor(sector)}>{fmt(sector)}</td>
                    <td className={numColor(spy)}>{fmt(spy)}</td>
                    <td className={numColor(vsSector)}>{fmt(vsSector)}</td>
                    <td className={numColor(vsMarket)}>{fmt(vsMarket)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sector Rotation Chart */}
      {sectorBars.length > 0 && (
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <h4 className="mb-4 text-sm font-medium text-slate-300">
            Sector Rotation (3M Returns)
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={sectorBars}
              margin={{ top: 5, right: 10, left: 10, bottom: 60 }}
            >
              <XAxis
                dataKey="name"
                tick={{ fill: "#94a3b8", fontSize: 10 }}
                angle={-45}
                textAnchor="end"
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickFormatter={(v: number) => `${v}%`}
              />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 8,
                }}
                labelStyle={{ color: "#e2e8f0" }}
                formatter={(v: number) => [`${v.toFixed(1)}%`, "Return"]}
              />
              <Bar dataKey="return" radius={[4, 4, 0, 0]}>
                {sectorBars.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      entry.name.toLowerCase().includes(
                        (data.sector || "").toLowerCase().slice(0, 4)
                      )
                        ? "#38bdf8"
                        : entry.return >= 0
                          ? "#10b981"
                          : "#f43f5e"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Peer Comparison */}
      {data.peers && data.peers.length > 0 && (
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <h4 className="mb-4 text-sm font-medium text-slate-300">
            Peer Comparison
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-navy-700 text-left text-xs text-slate-500">
                  <th className="pb-2">Ticker</th>
                  <th className="pb-2">Name</th>
                  <th className="pb-2">P/E</th>
                  <th className="pb-2">Rev Growth</th>
                  <th className="pb-2">Margin</th>
                  <th className="pb-2">ROE</th>
                </tr>
              </thead>
              <tbody>
                {data.peers.map((peer) => (
                  <tr
                    key={peer.ticker}
                    className="border-b border-navy-700/50"
                  >
                    <td className="py-2 font-mono text-sky-400">
                      {peer.ticker}
                    </td>
                    <td className="py-2 text-slate-300">{peer.name}</td>
                    <td className="py-2 text-slate-400">
                      {peer.pe_ratio?.toFixed(1) ?? "—"}
                    </td>
                    <td className={numColor(peer.revenue_growth)}>
                      {fmt(peer.revenue_growth)}
                    </td>
                    <td className={numColor(peer.profit_margin)}>
                      {fmt(peer.profit_margin)}
                    </td>
                    <td className={numColor(peer.roe)}>{fmt(peer.roe)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function fmt(v: number | undefined | null): string {
  if (v === undefined || v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function numColor(v: number | undefined | null): string {
  if (v === undefined || v === null) return "py-2 text-slate-500";
  return v >= 0 ? "py-2 text-emerald-400" : "py-2 text-rose-400";
}
