"use client";

import { clsx } from "clsx";
import type { Icon } from "@phosphor-icons/react";
import {
  MacroFedFunds, MacroCpi, MacroUnemployment, MacroGdp,
  MacroYieldSpread, MacroConsumer, MacroTreasury, MacroDefault,
  IconWarning,
} from "@/lib/icons";

interface FredIndicator {
  name: string;
  value: number;
  unit: string;
  date: string;
}

interface FredData {
  signal: string;
  summary: string;
  indicators: Record<
    string,
    { current: number; previous: number; change: number; series_id: string }
  >;
  warnings: string[];
  [key: string]: unknown;
}

const INDICATOR_META: Record<string, { label: string; icon: Icon; unit: string }> = {
  fed_funds_rate: { label: "Fed Funds Rate", icon: MacroFedFunds, unit: "%" },
  cpi_yoy: { label: "CPI (YoY)", icon: MacroCpi, unit: "%" },
  unemployment: { label: "Unemployment", icon: MacroUnemployment, unit: "%" },
  gdp_growth: { label: "GDP Growth", icon: MacroGdp, unit: "%" },
  yield_spread_10y2y: { label: "10Y-2Y Spread", icon: MacroYieldSpread, unit: "%" },
  consumer_sentiment: { label: "Consumer Sentiment", icon: MacroConsumer, unit: "" },
  treasury_10y: { label: "10Y Treasury", icon: MacroTreasury, unit: "%" },
};

export function MacroDashboard({ data }: { data: FredData }) {
  const signal = data.signal?.toUpperCase() || "N/A";
  const indicators = data.indicators || {};

  return (
    <div className="space-y-4">
      {/* Signal Header */}
      <div className="flex items-center justify-between rounded-xl border border-navy-700 bg-navy-800/60 p-4">
        <div>
          <h3 className="text-sm font-medium text-slate-300">
            Macro Environment
          </h3>
          <p className="mt-1 text-xs text-slate-500">{data.summary}</p>
        </div>
        <span
          className={clsx(
            "rounded-full px-3 py-1 text-xs font-semibold",
            signal.includes("DEFENSIVE")
              ? "bg-rose-500/10 text-rose-400"
              : signal.includes("EASING")
                ? "bg-emerald-500/10 text-emerald-400"
                : "bg-amber-500/10 text-amber-400"
          )}
        >
          {signal}
        </span>
      </div>

      {/* Indicator Grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {Object.entries(indicators).map(([key, ind]) => {
          const meta = INDICATOR_META[key] || {
            label: key,
            icon: MacroDefault,
            unit: "",
          };
          const changeColor =
            ind.change > 0 ? "text-emerald-400" : ind.change < 0 ? "text-rose-400" : "text-slate-500";

          return (
            <div
              key={key}
              className="rounded-lg border border-navy-700 bg-navy-800/40 p-3"
            >
              <div className="mb-1 flex items-center gap-1.5 text-xs text-slate-500">
                <meta.icon size={14} className="text-slate-500" />
                {meta.label}
              </div>
              <div className="text-lg font-bold text-slate-200">
                {ind.current?.toFixed(2)}
                {meta.unit}
              </div>
              <div className={clsx("text-xs", changeColor)}>
                {ind.change > 0 ? "+" : ""}
                {ind.change?.toFixed(2)} from prev
              </div>
            </div>
          );
        })}
      </div>

      {/* Warnings */}
      {data.warnings && data.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-900/50 bg-amber-950/30 p-3">
          <h4 className="mb-1 flex items-center gap-1 text-xs font-medium text-amber-400">
            <IconWarning size={14} weight="fill" />
            Macro Warnings
          </h4>
          <ul className="space-y-0.5 text-xs text-amber-200/80">
            {data.warnings.map((w, i) => (
              <li key={i}>• {w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
