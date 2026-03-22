"use client";

import { useState, useMemo } from "react";
import type { PerfExperiment } from "@/lib/types";
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

// ── Endpoint color palette (10 hues for dark backgrounds) ───────

const ENDPOINT_COLORS = [
  "#22d3ee", // cyan
  "#a78bfa", // violet
  "#f472b6", // pink
  "#fb923c", // orange
  "#34d399", // emerald
  "#facc15", // yellow
  "#60a5fa", // blue
  "#f87171", // red
  "#818cf8", // indigo
  "#2dd4bf", // teal
];

function endpointColor(endpoint: string, endpoints: string[]): string {
  const idx = endpoints.indexOf(endpoint);
  return ENDPOINT_COLORS[idx % ENDPOINT_COLORS.length];
}

// ── Types ───────────────────────────────────────────────────────

interface ChartPoint {
  index: number;
  p95: number;
  p95Display: number; // clamped for display so outliers don't stretch axis
  status: "kept" | "discarded";
  clamped: boolean; // true if p95 was clamped to ceiling
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

  const { data, keptCount, yMax, uniqueEndpoints } = useMemo(() => {
    // Collect unique endpoints in order of first appearance
    const endpointSet: string[] = [];
    for (const exp of experiments) {
      if (!endpointSet.includes(exp.endpoint)) endpointSet.push(exp.endpoint);
    }

    let best = Infinity;
    const raw: ChartPoint[] = experiments.map((exp, i) => {
      const p95 = parseFloat(exp.p95_after) || 0;
      const isKept = exp.status.toLowerCase() === "keep";
      if (isKept && p95 < best) best = p95;

      const ttlBefore = exp.ttl_before;
      const ttlAfter = exp.ttl_after;
      const lbl = isKept ? `${exp.endpoint} ${ttlBefore}→${ttlAfter}` : "";
      const label = lbl.length > 35 ? lbl.slice(0, 32) + "…" : lbl;

      return {
        index: i,
        p95,
        p95Display: p95, // will clamp below
        status: isKept ? ("kept" as const) : ("discarded" as const),
        clamped: false,
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
    for (const pt of raw) {
      if (pt.status === "kept") lastBest = pt.runningBest;
      else pt.runningBest = lastBest;
    }

    // Compute smart Y ceiling: zoom into the "interesting region" (karpathy style)
    // Use the max of: baseline (first experiment), or largest kept p95, with 50% margin
    const keptPts = raw.filter((p) => p.status === "kept");
    const firstP95 = raw.length > 0 ? raw[0].p95 : 100;
    const maxKept = keptPts.length > 0 ? Math.max(...keptPts.map((p) => p.p95)) : firstP95;
    const ceiling = Math.max(maxKept, firstP95) * 1.8 + 50; // 80% margin + 50ms padding

    // Clamp outlier discarded dots to ceiling so they appear at the top edge
    for (const pt of raw) {
      if (pt.p95 > ceiling) {
        pt.p95Display = ceiling;
        pt.clamped = true;
      }
    }

    return {
      data: raw,
      keptCount: keptPts.length,
      yMax: Math.ceil(ceiling / 50) * 50, // round up to nearest 50
      uniqueEndpoints: endpointSet,
    };
  }, [experiments]);

  if (experiments.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        No experiments yet — start the TTL Optimizer to generate data.
      </p>
    );
  }

  const keptData = data.filter((d) => d.status === "kept");
  const discardedData = data.filter((d) => d.status === "discarded");

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
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 40, right: 20, bottom: 24, left: 16 }}
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
              domain={[0, yMax]}
            />
            <Tooltip
              content={<CustomTooltip onSelect={setSelectedExp} endpoints={uniqueEndpoints} />}
              cursor={{ stroke: "#334155", strokeWidth: 1 }}
            />

            {/* Running best line */}
            <Line
              dataKey="runningBest"
              type="stepAfter"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              activeDot={false}
              connectNulls
              isAnimationActive={false}
            />

            {/* Discarded dots — colored by endpoint, clamped ones as triangles */}
            <Scatter
              dataKey="p95Display"
              data={discardedData}
              fill="#475569"
              fillOpacity={0.5}
              isAnimationActive={false}
              shape={(props: any) => {
                const pt = props.payload as ChartPoint;
                const color = endpointColor(pt.endpoint, uniqueEndpoints);
                if (pt.clamped) {
                  return (
                    <polygon
                      points={`${props.cx},${props.cy - 4} ${props.cx - 3},${props.cy + 2} ${props.cx + 3},${props.cy + 2}`}
                      fill={color}
                      fillOpacity={0.35}
                    />
                  );
                }
                return <circle cx={props.cx} cy={props.cy} r={3} fill={color} fillOpacity={0.35} />;
              }}
            />

            {/* Kept dots — colored by endpoint with bright ring + staggered labels */}
            <Scatter
              dataKey="p95Display"
              data={keptData}
              fill="#22c55e"
              r={5}
              isAnimationActive={false}
              shape={(props: any) => {
                const pt = props.payload as ChartPoint;
                const color = endpointColor(pt.endpoint, uniqueEndpoints);
                // Find this dot's rank among kept dots to stagger labels
                const keptIdx = keptData.findIndex((k) => k.index === pt.index);
                const goUp = keptIdx % 2 === 0;
                const tier = Math.floor(keptIdx / 2);
                const baseOffset = 14;
                const tierSpacing = 12;
                const yOffset = goUp
                  ? -(baseOffset + tier * tierSpacing)
                  : baseOffset + 6 + tier * tierSpacing;
                return (
                  <g>
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={5}
                      fill={color}
                      stroke="#fff"
                      strokeWidth={1.5}
                    />
                    {pt.label && (
                      <>
                        {/* Leader line from dot to label */}
                        <line
                          x1={props.cx}
                          y1={props.cy}
                          x2={props.cx + 6}
                          y2={props.cy + yOffset + (goUp ? 4 : -4)}
                          stroke={color}
                          strokeWidth={0.5}
                          opacity={0.4}
                        />
                        <text
                          x={props.cx + 8}
                          y={props.cy + yOffset}
                          fill={color}
                          fontSize={8}
                          opacity={0.85}
                          textAnchor="start"
                          dominantBaseline="auto"
                        >
                          {pt.label}
                        </text>
                      </>
                    )}
                  </g>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend — endpoint colors + status indicators */}
      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1.5 text-xs text-slate-500">
        {/* Endpoint colors */}
        {uniqueEndpoints.map((ep) => {
          const color = endpointColor(ep, uniqueEndpoints);
          // Shorten long endpoint paths: "/api/paper-trading/snapshots" → "snapshots"
          const short = ep.split("/").filter(Boolean).pop() || ep;
          return (
            <span key={ep} className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="font-mono" title={ep}>{short}</span>
            </span>
          );
        })}
        {/* Separator */}
        <span className="text-slate-700">|</span>
        {/* Status indicators */}
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-white/80 bg-slate-600" />
          Kept (ring)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-slate-500 opacity-35" />
          Discarded
        </span>
        <span className="flex items-center gap-1">
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
              <p className="flex items-center gap-1.5 font-mono text-xs text-slate-300">
                <span
                  className="inline-block h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: endpointColor(selectedExp.endpoint, uniqueEndpoints) }}
                />
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
  endpoints,
}: TooltipProps<number, string> & {
  onSelect: (pt: ChartPoint) => void;
  endpoints: string[];
}) {
  if (!active || !payload?.length) return null;

  // Recharts can deliver payload from multiple series — find the one
  // with a full ChartPoint payload (from a Scatter).
  const entry = payload.find((p) => p.payload?.endpoint);
  if (!entry) return null;

  const pt = entry.payload as ChartPoint;
  const color = endpointColor(pt.endpoint, endpoints);

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
      <p className="flex items-center gap-1.5 font-mono text-slate-400">
        <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
        {pt.endpoint}
      </p>
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
