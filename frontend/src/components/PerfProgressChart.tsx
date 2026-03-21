"use client";

import { useState, useMemo } from "react";
import type { PerfExperiment } from "@/lib/types";
import {
  ScatterChart,
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
  index: number;
  p95: number;
  status: "kept" | "discarded";
  endpoint: string;
  ttl_before: string;
  ttl_after: string;
  p95_before: string;
  p95_after: string;
  hit_rate: string;
  timestamp: string;
  runningBest: number | null;
  label: string;
}

interface Props {
  experiments: PerfExperiment[];
}

// ── Chart component ─────────────────────────────────────────────

export function PerfProgressChart({ experiments }: Props) {
  const [selectedExp, setSelectedExp] = useState<ChartPoint | null>(null);

  const { data, keptCount } = useMemo(() => {
    let best = Infinity;
    const points: ChartPoint[] = experiments.map((exp, i) => {
      const p95 = parseFloat(exp.p95_after) || 0;
      const isKept = exp.status.toLowerCase() === "kept";
      if (isKept && p95 < best) best = p95;

      const ttlBefore = exp.ttl_before;
      const ttlAfter = exp.ttl_after;
      const label = isKept
        ? `${exp.endpoint} TTL ${ttlBefore}→${ttlAfter}`
        : "";

      return {
        index: i,
        p95,
        status: isKept ? "kept" : "discarded",
        endpoint: exp.endpoint,
        ttl_before: ttlBefore,
        ttl_after: ttlAfter,
        p95_before: exp.p95_before,
        p95_after: exp.p95_after,
        hit_rate: exp.hit_rate,
        timestamp: exp.timestamp,
        runningBest: isKept ? best : null,
        label,
      };
    });

    // Fill running best line between kept points
    let lastBest: number | null = null;
    for (const pt of points) {
      if (pt.status === "kept") lastBest = pt.runningBest;
      else pt.runningBest = lastBest;
    }

    return {
      data: points,
      keptCount: points.filter((p) => p.status === "kept").length,
    };
  }, [experiments]);

  if (experiments.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        No experiments yet — start the TTL Optimizer to generate data.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {/* Title */}
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-semibold text-slate-300">
          Autoresearch Progress: {experiments.length} Experiments, {keptCount}{" "}
          Kept Improvements
        </h4>
      </div>

      {/* Chart */}
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 12, right: 20, bottom: 24, left: 16 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e293b"
              vertical={false}
            />
            <XAxis
              dataKey="index"
              type="number"
              domain={[0, "dataMax"]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{
                value: "Experiment #",
                position: "insideBottom",
                offset: -12,
                fill: "#64748b",
                fontSize: 11,
              }}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              label={{
                value: "p95 Latency (ms)",
                angle: -90,
                position: "insideLeft",
                offset: -4,
                fill: "#64748b",
                fontSize: 11,
              }}
              domain={["auto", "auto"]}
            />
            <Tooltip content={<CustomTooltip onSelect={setSelectedExp} />} />

            {/* Running best line */}
            <Line
              dataKey="runningBest"
              type="stepAfter"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />

            {/* Discarded dots */}
            <Scatter
              dataKey="p95"
              data={data.filter((d) => d.status === "discarded")}
              fill="#475569"
              fillOpacity={0.5}
              r={3}
              isAnimationActive={false}
            />

            {/* Kept dots */}
            <Scatter
              dataKey="p95"
              data={data.filter((d) => d.status === "kept")}
              fill="#22c55e"
              r={5}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-5 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-slate-500 opacity-50" />
          Discarded
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
          Kept
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 bg-emerald-500" />
          Running best
        </span>
      </div>

      {/* Detail panel on click */}
      {selectedExp && (
        <div className="rounded-lg border border-slate-700 bg-slate-800/80 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h5 className="text-sm font-semibold text-slate-200">
              Experiment #{selectedExp.index}
              <span
                className={`ml-2 inline-block rounded px-1.5 py-0.5 text-xs font-medium ${
                  selectedExp.status === "kept"
                    ? "bg-emerald-900/40 text-emerald-300"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                {selectedExp.status}
              </span>
            </h5>
            <button
              onClick={() => setSelectedExp(null)}
              className="text-xs text-slate-500 transition-colors hover:text-slate-300"
            >
              Close
            </button>
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <div>
              <span className="text-slate-500">Endpoint</span>
              <p className="font-mono text-xs text-slate-300">
                {selectedExp.endpoint}
              </p>
            </div>
            <div>
              <span className="text-slate-500">Timestamp</span>
              <p className="text-xs text-slate-300">{selectedExp.timestamp}</p>
            </div>
            <div>
              <span className="text-slate-500">TTL Change</span>
              <p className="text-xs text-slate-300">
                {selectedExp.ttl_before}s → {selectedExp.ttl_after}s
              </p>
            </div>
            <div>
              <span className="text-slate-500">p95 Latency</span>
              <p className="text-xs text-slate-300">
                {selectedExp.p95_before}ms →{" "}
                <span
                  className={
                    parseFloat(selectedExp.p95_after) <
                    parseFloat(selectedExp.p95_before)
                      ? "text-emerald-400"
                      : "text-rose-400"
                  }
                >
                  {selectedExp.p95_after}ms
                </span>
              </p>
            </div>
            <div>
              <span className="text-slate-500">Hit Rate</span>
              <p className="text-xs text-slate-300">{selectedExp.hit_rate}%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Custom tooltip ──────────────────────────────────────────────

function CustomTooltip({
  active,
  payload,
  onSelect,
}: TooltipProps<number, string> & {
  onSelect: (pt: ChartPoint) => void;
}) {
  if (!active || !payload?.length) return null;

  // Recharts can deliver payload from multiple series — find the one
  // with a full ChartPoint payload (from a Scatter).
  const entry = payload.find((p) => p.payload?.endpoint);
  if (!entry) return null;

  const pt = entry.payload as ChartPoint;

  return (
    <div
      className="cursor-pointer rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-lg backdrop-blur"
      onClick={() => onSelect(pt)}
    >
      <div className="mb-1 flex items-center gap-2">
        <span className="font-semibold text-slate-200">
          #{pt.index}
        </span>
        <span
          className={`rounded px-1 py-0.5 text-[10px] font-medium ${
            pt.status === "kept"
              ? "bg-emerald-900/40 text-emerald-300"
              : "bg-slate-700 text-slate-400"
          }`}
        >
          {pt.status}
        </span>
      </div>
      <p className="font-mono text-slate-400">{pt.endpoint}</p>
      <p className="text-slate-400">
        TTL {pt.ttl_before}s → {pt.ttl_after}s
      </p>
      <p className="text-slate-400">
        p95 {pt.p95_before}ms → {pt.p95_after}ms
      </p>
      <p className="mt-1 text-[10px] text-slate-600">Click for full details</p>
    </div>
  );
}
