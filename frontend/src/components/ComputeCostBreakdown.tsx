/**
 * phase-10.5.4 ComputeCostBreakdown.
 *
 * Recharts stacked-bar chart for daily compute cost split by the 5
 * canonical providers (anthropic / vertex / openai / bigquery /
 * altdata). Backend at `/api/sovereign/compute-cost` always emits the
 * full provider key set so this component can rely on a deterministic
 * stack order.
 *
 * Custom CostTooltip renders each provider as `$X.XXXX (Y.Y%)` of the
 * day's total.
 *
 * Color palette: 5-of-8 Okabe-Ito (yellow excluded for dark-bg
 * contrast). Source: ConceptViz Okabe-Ito Reference (2008 definition).
 */
"use client";

import { BentoCard } from "@/components/BentoCard";
import { CurrencyDollar } from "@phosphor-icons/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type Provider =
  | "anthropic"
  | "vertex"
  | "openai"
  | "bigquery"
  | "altdata";

export const PROVIDERS: readonly Provider[] = [
  "anthropic",
  "vertex",
  "openai",
  "bigquery",
  "altdata",
] as const;

// Okabe-Ito 5-of-8, yellow dropped for dark-bg contrast.
export const PROVIDER_COLORS: Record<Provider, string> = {
  anthropic: "#56B4E9", // Sky Blue
  vertex: "#E69F00",    // Orange
  openai: "#009E73",    // Bluish Green
  bigquery: "#D55E00",  // Vermillion
  altdata: "#CC79A7",   // Reddish Purple
};

export interface ProviderCostPoint {
  date: string;
  anthropic: number;
  vertex: number;
  openai: number;
  bigquery: number;
  altdata: number;
}

export interface ComputeCostBreakdownProps {
  data: ProviderCostPoint[];
  grandTotal: number;
  window: "7d" | "30d" | "90d";
}

interface TooltipPayloadEntry {
  name?: string;
  dataKey?: string;
  value?: number;
  payload?: ProviderCostPoint;
  color?: string;
}

interface CostTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function rowTotal(p: ProviderCostPoint): number {
  return PROVIDERS.reduce((s, k) => s + (Number(p[k]) || 0), 0);
}

function fmtUsd(v: number): string {
  if (!Number.isFinite(v)) return "$0.00";
  if (v >= 1) return `$${v.toFixed(2)}`;
  return `$${v.toFixed(4)}`;
}

/** Exported so the test can call it directly without Recharts'
 * jsdom-hostile hover model. */
export function CostTooltip({ active, payload, label }: CostTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const total = rowTotal(row);
  return (
    <div
      data-testid="cost-tooltip"
      className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-xl"
    >
      <p className="mb-1 font-mono font-semibold text-slate-200">{label}</p>
      {PROVIDERS.map((k) => {
        const v = Number(row[k]) || 0;
        const pct = total > 0 ? (v / total) * 100 : 0;
        return (
          <p key={k} className="text-slate-400">
            <span
              className="mr-2 inline-block h-2 w-2 rounded-sm align-middle"
              style={{ backgroundColor: PROVIDER_COLORS[k] }}
            />
            <span className="text-slate-300">{k}</span>:{" "}
            <span className="font-mono text-slate-200">
              {fmtUsd(v)} ({pct.toFixed(1)}%)
            </span>
          </p>
        );
      })}
      <p className="mt-1 border-t border-slate-700 pt-1 text-slate-500">
        day total <span className="font-mono text-slate-300">{fmtUsd(total)}</span>
      </p>
    </div>
  );
}

export function ComputeCostBreakdown({
  data,
  grandTotal,
  window,
}: ComputeCostBreakdownProps) {
  const isEmpty = !data || data.length === 0;

  return (
    <BentoCard>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <CurrencyDollar size={18} className="text-emerald-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">
          Compute Cost Breakdown ({window})
        </h3>
        <span className="ml-auto font-mono text-xs text-slate-400">
          window total {fmtUsd(grandTotal)}
        </span>
      </div>

      {isEmpty ? (
        <div
          data-testid="compute-cost-empty"
          className="flex flex-col items-center justify-center py-10 text-center"
        >
          <CurrencyDollar
            size={32}
            weight="duotone"
            className="text-slate-600"
            aria-hidden="true"
          />
          <p className="mt-3 text-sm text-slate-400">
            No cost data for the {window} window
          </p>
          <p className="mt-1 text-xs text-slate-600">
            BQ INFORMATION_SCHEMA.JOBS reports zero billed jobs in range.
          </p>
        </div>
      ) : (
        <div
          data-testid="compute-cost-chart"
          role="img"
          aria-label={`Compute cost stacked bar, ${window} window, ${data.length} days, ${PROVIDERS.length} providers`}
          className="h-64"
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickFormatter={(v: number) => fmtUsd(v)}
              />
              <Tooltip content={<CostTooltip />} />
              <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} />
              {PROVIDERS.map((k) => (
                <Bar
                  key={k}
                  dataKey={k}
                  stackId="cost"
                  fill={PROVIDER_COLORS[k]}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </BentoCard>
  );
}
