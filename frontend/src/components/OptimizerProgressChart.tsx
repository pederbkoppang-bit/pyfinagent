"use client";

import { useMemo } from "react";
import type { OptimizerExperiment } from "@/lib/types";
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
  index: number;
  sharpe: number;
  sharpeDisplay: number;
  dsr: number;
  status: "keep" | "baseline" | "discard" | "dsr_reject" | "crash";
  clamped: boolean;
  modification: string;
  metricBefore: number;
  metricAfter: number;
  runningBest: number | null;
  label: string;
}

interface Props {
  experiments: OptimizerExperiment[];
}

// ── Status colors ───────────────────────────────────────────────

function statusColor(status: string): string {
  switch (status) {
    case "keep":
      return "#22c55e"; // emerald
    case "BASELINE":
    case "baseline":
      return "#38bdf8"; // sky
    case "dsr_reject":
      return "#f59e0b"; // amber
    default:
      return "#64748b"; // slate
  }
}

// ── Custom Tooltip ──────────────────────────────────────────────

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const pt = payload[0]?.payload as ChartPoint;
  if (!pt) return null;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 shadow-xl">
      <p className="mb-1 text-xs font-semibold text-slate-200">
        Experiment #{pt.index}
        <span
          className="ml-2 inline-block rounded px-1.5 py-0.5 text-[10px] font-medium"
          style={{ color: statusColor(pt.status), backgroundColor: `${statusColor(pt.status)}20` }}
        >
          {pt.status.toUpperCase()}
        </span>
      </p>
      {pt.modification && pt.modification !== "—" && (
        <p className="mb-1 max-w-[240px] truncate text-xs text-slate-400">{pt.modification}</p>
      )}
      <div className="flex gap-4 text-xs">
        <span className="text-slate-500">
          Sharpe: <span className="font-mono text-slate-300">{pt.sharpe.toFixed(3)}</span>
        </span>
        <span className="text-slate-500">
          DSR: <span className="font-mono text-slate-300">{pt.dsr.toFixed(3)}</span>
        </span>
      </div>
      {pt.status === "keep" && pt.metricBefore > 0 && (
        <p className="mt-1 text-[10px] text-emerald-400">
          Δ +{(pt.metricAfter - pt.metricBefore).toFixed(4)} Sharpe
        </p>
      )}
    </div>
  );
}

// ── Chart Component ─────────────────────────────────────────────

export function OptimizerProgressChart({ experiments }: Props) {
  const { data, keptCount, yMin, yMax } = useMemo(() => {
    let best = -Infinity;
    const raw: ChartPoint[] = experiments.map((exp, i) => {
      const sharpe = parseFloat(exp.metric_after) || 0;
      const dsr = parseFloat(exp.dsr) || 0;
      const status = exp.status.toLowerCase();
      const isKept = status === "keep" || status === "baseline";
      if (isKept && sharpe > best) best = sharpe;

      const mod = exp.param_changed || "—";
      const label = isKept && status !== "baseline" ? `${mod.split(":")[0]}` : "";

      return {
        index: i,
        sharpe,
        sharpeDisplay: sharpe,
        dsr,
        status: status as ChartPoint["status"],
        clamped: false,
        modification: mod,
        metricBefore: parseFloat(exp.metric_before) || 0,
        metricAfter: sharpe,
        runningBest: isKept ? best : null,
        label: label.length > 25 ? label.slice(0, 22) + "…" : label,
      };
    });

    // Fill running best line between kept points
    let lastBest: number | null = null;
    for (const pt of raw) {
      if (pt.status === "keep" || pt.status === "baseline") lastBest = pt.runningBest;
      else pt.runningBest = lastBest;
    }

    // Smart Y-axis: zoom into the interesting region (Karpathy style)
    const allSharpes = raw.map((p) => p.sharpe).filter((s) => s > 0);
    const keptPts = raw.filter((p) => p.status === "keep" || p.status === "baseline");
    const minKept = keptPts.length > 0 ? Math.min(...keptPts.map((p) => p.sharpe)) : 0;
    const maxKept = keptPts.length > 0 ? Math.max(...keptPts.map((p) => p.sharpe)) : 1;

    // Floor and ceiling with margins
    const range = maxKept - minKept || 1;
    const floor = Math.max(0, minKept - range * 0.5);
    const ceiling = maxKept + range * 0.3;

    // Clamp extreme outlier discards
    for (const pt of raw) {
      if (pt.sharpe < floor * 0.5 && pt.status !== "keep" && pt.status !== "baseline") {
        pt.sharpeDisplay = floor;
        pt.clamped = true;
      } else if (pt.sharpe > ceiling * 1.5 && pt.status !== "keep" && pt.status !== "baseline") {
        pt.sharpeDisplay = ceiling;
        pt.clamped = true;
      }
    }

    return {
      data: raw,
      keptCount: keptPts.length,
      yMin: Math.floor(floor * 100) / 100,
      yMax: Math.ceil(ceiling * 100) / 100,
    };
  }, [experiments]);

  if (experiments.length < 2) return null;

  const keptData = data.filter((d) => d.status === "keep" || d.status === "baseline");
  const discardedData = data.filter((d) => d.status !== "keep" && d.status !== "baseline");

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-semibold text-slate-300">
          Optimization Progress: {experiments.length} Experiments, {keptCount} Improvements
        </h4>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 30, right: 20, bottom: 24, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
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

            {/* Discarded dots */}
            <Scatter
              dataKey="sharpeDisplay"
              data={discardedData}
              isAnimationActive={false}
              shape={(props: any) => {
                const pt = props.payload as ChartPoint;
                const color = statusColor(pt.status);
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

            {/* Kept dots with labels */}
            <Scatter
              dataKey="sharpeDisplay"
              data={keptData}
              isAnimationActive={false}
              shape={(props: any) => {
                const pt = props.payload as ChartPoint;
                const color = statusColor(pt.status);
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
          <span className="inline-block h-2 w-2 rounded-full bg-amber-500 opacity-50" />
          DSR Rejected
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
    </div>
  );
}
