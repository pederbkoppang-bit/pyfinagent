/**
 * phase-15.8 Transformer forecast panel.
 *
 * Shows TimesFM + Chronos zero-shot forecasts (currently empty arrays
 * while phase-8.4 REJECT stands) with an amber shadow-mode banner
 * that quotes the truthful rejection reason. Renders a Recharts line
 * overlay when the arrays are non-empty; empty-state placeholder
 * otherwise.
 */
"use client";

import { BentoCard } from "@/components/BentoCard";
import type { TransformerForecastResponse } from "@/lib/types";
import { Warning, LineSegments } from "@phosphor-icons/react";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Legend,
} from "recharts";

export interface TransformerForecastPanelProps {
  data: TransformerForecastResponse | null;
}

function buildSeries(d: TransformerForecastResponse): Array<{
  step: number;
  timesfm: number | null;
  chronos: number | null;
}> {
  const n = Math.max(d.timesfm.length, d.chronos.length, d.horizon);
  const rows: Array<{ step: number; timesfm: number | null; chronos: number | null }> = [];
  for (let i = 0; i < n; i++) {
    rows.push({
      step: i + 1,
      timesfm: d.timesfm[i] != null ? d.timesfm[i] : null,
      chronos: d.chronos[i] != null ? d.chronos[i] : null,
    });
  }
  return rows;
}

export function TransformerForecastPanel({ data }: TransformerForecastPanelProps) {
  if (!data) return null;
  const hasForecasts = data.timesfm.length > 0 || data.chronos.length > 0;
  const weights = data.ensemble_weights;

  return (
    <BentoCard>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <LineSegments size={18} className="text-amber-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">
          Transformer Forecast ({data.ticker})
        </h3>
        <span
          data-forecast-status={data.status}
          className="ml-auto inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs font-medium text-amber-400"
        >
          <Warning size={12} weight="fill" />
          {data.status.toUpperCase()}
        </span>
      </div>

      {/* Shadow-mode banner */}
      <div
        data-shadow-banner="true"
        className="mb-4 rounded-lg border border-amber-700/50 bg-amber-950/40 p-3"
      >
        <p className="text-xs font-semibold text-amber-300">
          SHADOW MODE -- Not for trading decisions
        </p>
        <p className="mt-1 text-xs text-amber-200/80">
          {data.phase8_reject_reason}
        </p>
      </div>

      {/* Ensemble weights */}
      <div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span>
          Ensemble weights:{" "}
          <span className="font-mono text-slate-300">MDA {weights.mda.toFixed(2)}</span>
          {" · "}
          <span className="font-mono text-slate-300">
            TimesFM {weights.timesfm.toFixed(2)}
          </span>
          {" · "}
          <span className="font-mono text-slate-300">
            Chronos {weights.chronos.toFixed(2)}
          </span>
        </span>
        <span className="ml-auto font-mono">
          {data.model_timesfm} · {data.model_chronos}
        </span>
      </div>

      {/* Chart or empty state */}
      {hasForecasts ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={buildSeries(data)} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="step"
                tick={{ fill: "#64748b", fontSize: 11 }}
                label={{
                  value: `horizon (days)`,
                  position: "insideBottom",
                  offset: -4,
                  fill: "#64748b",
                  fontSize: 11,
                }}
              />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #1e293b",
                  borderRadius: 8,
                  color: "#e2e8f0",
                }}
              />
              <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="timesfm"
                name="TimesFM"
                stroke="#38bdf8"
                strokeWidth={1.5}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="chronos"
                name="Chronos"
                stroke="#a78bfa"
                strokeWidth={1.5}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div
          data-forecast-state="empty"
          className="flex flex-col items-center justify-center py-10 text-center"
        >
          <LineSegments
            size={32}
            weight="duotone"
            className="text-slate-600"
            aria-hidden="true"
          />
          <p className="mt-3 text-sm text-slate-400">
            No forecast series to render
          </p>
          <p className="mt-1 text-xs text-slate-600">
            Clients return empty arrays in the Python 3.14 runtime; see banner
            above for the gating criteria.
          </p>
        </div>
      )}
    </BentoCard>
  );
}
