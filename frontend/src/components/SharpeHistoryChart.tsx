"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  SharpeHistoryEntry,
  SharpeHistoryResponse,
  SharpeHistorySummary,
} from "@/lib/types";
import { getSharpeHistory } from "@/lib/api";
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
  TooltipProps,
} from "recharts";

// ── Types ───────────────────────────────────────────────────────

interface ChartPoint {
  timestamp: string;
  dateMs: number;
  sharpe: number;
  dsr: number | null;
  totalReturn: number;
  maxDrawdown: number;
  nTrades: number;
  runId: string;
  status: string;
  isBaseline: boolean;
  bestSoFar: number;
}

// ── Status colors ───────────────────────────────────────────────

function normalizeStatus(status: string): string {
  const s = status.toLowerCase();
  if (s === "baseline") return "baseline";
  if (s === "kept" || s === "keep") return "kept";
  if (s === "discarded" || s === "discard") return "discarded";
  if (s === "crash" || s === "crashed") return "discarded"; // crashes show as discarded
  if (s === "seed_test") return "seed_test";
  return s;
}

function statusColor(status: string): string {
  switch (normalizeStatus(status)) {
    case "kept":
      return "#22c55e";
    case "baseline":
      return "#38bdf8";
    case "discarded":
      return "#ef4444";
    case "seed_test":
      return "#a78bfa";
    default:
      return "#64748b";
  }
}

function statusLabel(status: string): string {
  const n = normalizeStatus(status);
  switch (n) {
    case "kept":
      return "KEPT";
    case "baseline":
      return "BASELINE";
    case "discarded":
      return status.toLowerCase() === "crash" ? "CRASHED" : "DISCARDED";
    case "seed_test":
      return "SEED TEST";
    default:
      return status.toUpperCase();
  }
}

// ── Helpers ─────────────────────────────────────────────────────

function formatDate(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return ts;
  }
}

function formatDateShort(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Custom Tooltip ──────────────────────────────────────────────

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const pt = payload[0]?.payload as ChartPoint;
  if (!pt) return null;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 shadow-xl">
      <p className="mb-1 text-xs text-slate-400">{formatDate(pt.timestamp)}</p>
      <p className="mb-1 text-xs font-semibold text-slate-200">
        {pt.runId}
        <span
          className="ml-2 inline-block rounded px-1.5 py-0.5 text-[10px] font-medium"
          style={{
            color: statusColor(pt.status),
            backgroundColor: `${statusColor(pt.status)}20`,
          }}
        >
          {statusLabel(pt.status)}
        </span>
      </p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
        <span className="text-slate-500">
          Sharpe:{" "}
          <span className="font-mono text-slate-300">{pt.sharpe.toFixed(4)}</span>
        </span>
        <span className="text-slate-500">
          DSR:{" "}
          <span className="font-mono text-slate-300">
            {pt.dsr !== null ? pt.dsr.toFixed(4) : "—"}
          </span>
        </span>
        <span className="text-slate-500">
          Return:{" "}
          <span className="font-mono text-slate-300">
            {pt.totalReturn.toFixed(1)}%
          </span>
        </span>
        <span className="text-slate-500">
          MaxDD:{" "}
          <span className="font-mono text-slate-300">
            {pt.maxDrawdown.toFixed(1)}%
          </span>
        </span>
        <span className="text-slate-500">
          Trades:{" "}
          <span className="font-mono text-slate-300">{pt.nTrades}</span>
        </span>
        <span className="text-slate-500">
          Best:{" "}
          <span className="font-mono text-emerald-400">
            {pt.bestSoFar.toFixed(4)}
          </span>
        </span>
      </div>
    </div>
  );
}

// ── Summary Cards ───────────────────────────────────────────────

function SummaryCards({ summary }: { summary: SharpeHistorySummary }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Initial
        </p>
        <p className="font-mono text-lg font-semibold text-slate-300">
          {summary.initial_sharpe.toFixed(4)}
        </p>
      </div>
      <div className="rounded-lg border border-emerald-800/50 bg-emerald-900/20 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-emerald-500">
          Best
        </p>
        <p className="font-mono text-lg font-semibold text-emerald-400">
          {summary.current_best_sharpe.toFixed(4)}
        </p>
      </div>
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Improvement
        </p>
        <p className="font-mono text-lg font-semibold text-sky-400">
          +{summary.improvement_pct.toFixed(1)}%
        </p>
      </div>
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Experiments
        </p>
        <p className="font-mono text-lg font-semibold text-slate-300">
          {summary.total_experiments}
        </p>
      </div>
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Kept
        </p>
        <p className="font-mono text-lg font-semibold text-emerald-400">
          {summary.kept_count}
        </p>
      </div>
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Discarded
        </p>
        <p className="font-mono text-lg font-semibold text-red-400">
          {summary.discarded_count}
        </p>
      </div>
    </div>
  );
}

// ── Main Chart Component ────────────────────────────────────────

export function SharpeHistoryChart() {
  const [data, setData] = useState<SharpeHistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSharpeHistory()
      .then(setData)
      .catch((e) => setError(e.message || "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  const { chartData, yMin, yMax } = useMemo(() => {
    if (!data?.timeline?.length) return { chartData: [], yMin: 0, yMax: 1 };

    const points: ChartPoint[] = data.timeline.map((entry) => ({
      timestamp: entry.timestamp,
      dateMs: new Date(entry.timestamp).getTime(),
      sharpe: entry.sharpe,
      dsr: entry.dsr,
      totalReturn: entry.total_return_pct,
      maxDrawdown: entry.max_drawdown_pct,
      nTrades: entry.n_trades,
      runId: entry.run_id,
      status: entry.status,
      isBaseline: entry.is_baseline,
      bestSoFar: entry.best_sharpe_so_far,
    }));

    // Y-axis: zoom into the interesting region
    const keptSharpes = points
      .filter((p) => normalizeStatus(p.status) === "kept" || normalizeStatus(p.status) === "baseline")
      .map((p) => p.sharpe);
    const allSharpes = points.map((p) => p.sharpe).filter((s) => s > 0);

    const minRef = keptSharpes.length
      ? Math.min(...keptSharpes)
      : Math.min(...allSharpes);
    const maxRef = keptSharpes.length
      ? Math.max(...keptSharpes)
      : Math.max(...allSharpes);
    const range = maxRef - minRef || 0.5;

    return {
      chartData: points,
      yMin: Math.max(0, Math.floor((minRef - range * 0.4) * 100) / 100),
      yMax: Math.ceil((maxRef + range * 0.25) * 100) / 100,
    };
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed border-slate-700 py-16">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <svg
            className="h-4 w-4 animate-spin"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="3"
              className="opacity-25"
            />
            <path
              d="M4 12a8 8 0 018-8"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
              className="opacity-75"
            />
          </svg>
          Loading Sharpe history…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-400">
        Failed to load Sharpe history: {error}
      </div>
    );
  }

  if (!data?.timeline?.length) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 py-12 text-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="32"
          height="32"
          viewBox="0 0 256 256"
          className="mb-3 text-slate-500"
        >
          <path
            fill="currentColor"
            d="M232 208a8 8 0 0 1-8 8H32a8 8 0 0 1-8-8V48a8 8 0 0 1 16 0v108.69l50.34-50.35a8 8 0 0 1 11.32 0L128 132.69 180.69 80H160a8 8 0 0 1 0-16h40a8 8 0 0 1 8 8v40a8 8 0 0 1-16 0V91.31l-58.34 58.35a8 8 0 0 1-11.32 0L96 123.31l-56 56V200h184a8 8 0 0 1 0 8Z"
          />
        </svg>
        <p className="text-sm font-medium text-slate-400">
          Sharpe Ratio History
        </p>
        <p className="mt-1 text-xs text-slate-500">
          No experiment results found yet
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} />

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 20, right: 20, bottom: 28, left: 16 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e293b"
              vertical={false}
            />
            <XAxis
              dataKey="dateMs"
              type="number"
              domain={["dataMin", "dataMax"]}
              scale="time"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={(v: number) => formatDateShort(v)}
              label={{
                value: "Date",
                position: "insideBottom",
                offset: -16,
                fill: "#64748b",
                fontSize: 11,
              }}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{
                value: "Sharpe Ratio",
                angle: -90,
                position: "insideLeft",
                offset: -4,
                fill: "#64748b",
                fontSize: 11,
              }}
              domain={[yMin, yMax]}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ stroke: "#334155", strokeWidth: 1 }}
            />

            {/* Best-so-far envelope line */}
            <Line
              dataKey="bestSoFar"
              type="stepAfter"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              activeDot={false}
              connectNulls
              isAnimationActive={false}
            />

            {/* All experiment dots */}
            <Scatter
              dataKey="sharpe"
              isAnimationActive={false}
              shape={(props: any) => {
                const pt = props.payload as ChartPoint;
                const color = statusColor(pt.status);
                const isKept =
                  normalizeStatus(pt.status) === "kept" || normalizeStatus(pt.status) === "baseline";
                const r = isKept ? 5 : 3;
                const opacity = isKept ? 1 : 0.4;

                return (
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={r}
                    fill={color}
                    fillOpacity={opacity}
                    stroke={isKept ? "#fff" : "none"}
                    strokeWidth={isKept ? 1.5 : 0}
                  />
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1.5 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-white/80 bg-emerald-500" />
          Kept
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-white/80 bg-sky-400" />
          Baseline
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-red-500 opacity-50" />
          Discarded
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-white/80 bg-violet-400" />
          Seed Test
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-slate-500 opacity-35" />
          Unknown
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4 bg-emerald-500" />
          Best Sharpe
        </span>
      </div>
    </div>
  );
}
