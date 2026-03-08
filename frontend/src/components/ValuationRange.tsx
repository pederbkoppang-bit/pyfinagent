"use client";

import { BentoCard } from "./BentoCard";

interface ValuationMetric {
  label: string;
  value: number | null;
  benchmark: number | null;
  unit?: string;
  invert?: boolean; // lower is better (e.g., P/E for value stocks)
}

interface ValuationRangeProps {
  valuation?: Record<string, number | null>;
  health?: Record<string, number | null>;
}

function MetricBar({
  label,
  value,
  benchmark,
  unit = "",
  invert = false,
}: ValuationMetric) {
  if (value == null) {
    return (
      <div className="flex items-center gap-4">
        <span className="w-36 text-xs text-slate-400">{label}</span>
        <span className="text-xs text-slate-600">N/A</span>
      </div>
    );
  }

  // Determine where on a 0-100 scale the value sits
  const maxRange = benchmark ? benchmark * 2.5 : value * 2;
  const pct = Math.min(Math.max((value / maxRange) * 100, 2), 98);

  // Color: green = good, amber = neutral, red = caution
  let color = "bg-sky-400";
  if (benchmark != null) {
    const ratio = value / benchmark;
    if (invert) {
      if (ratio < 0.8) color = "bg-emerald-400";
      else if (ratio > 1.3) color = "bg-rose-400";
      else color = "bg-amber-400";
    } else {
      if (ratio > 1.2) color = "bg-emerald-400";
      else if (ratio < 0.7) color = "bg-rose-400";
      else color = "bg-amber-400";
    }
  }

  const formatted =
    Math.abs(value) >= 1e9
      ? `$${(value / 1e9).toFixed(1)}B`
      : Math.abs(value) >= 1e6
      ? `$${(value / 1e6).toFixed(1)}M`
      : typeof value === "number" && unit === "%"
      ? `${value.toFixed(1)}%`
      : typeof value === "number"
      ? value.toFixed(2)
      : String(value);

  return (
    <div className="group flex items-center gap-4">
      <span className="w-36 shrink-0 text-xs text-slate-400">{label}</span>
      <div className="relative h-3 flex-1 rounded-full bg-slate-700">
        <div
          className={`absolute top-0 h-3 rounded-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
        />
        {benchmark != null && (
          <div
            className="absolute top-0 h-3 w-0.5 bg-white/60"
            style={{ left: `${Math.min((benchmark / maxRange) * 100, 98)}%` }}
            title={`Benchmark: ${benchmark}`}
          />
        )}
      </div>
      <span className="w-20 text-right font-mono text-xs text-slate-300">
        {formatted}
        {unit}
      </span>
    </div>
  );
}

export function ValuationRange({ valuation, health }: ValuationRangeProps) {
  if (!valuation && !health) {
    return null;
  }

  const metrics: ValuationMetric[] = [];
  if (valuation) {
    metrics.push(
      { label: "P/E Ratio", value: valuation["P/E Ratio"] ?? null, benchmark: 25, invert: true },
      { label: "Forward P/E", value: valuation["Forward P/E"] ?? null, benchmark: 20, invert: true },
      { label: "PEG Ratio", value: valuation["PEG Ratio"] ?? null, benchmark: 1.5, invert: true },
      { label: "Price/Book", value: valuation["Price/Book"] ?? null, benchmark: 3, invert: true },
      { label: "Dividend Yield", value: valuation["Dividend Yield"] ?? null, benchmark: 2, unit: "%" },
    );
  }
  if (health) {
    metrics.push(
      { label: "Debt/Equity", value: health["Debt/Equity Ratio"] ?? null, benchmark: 100, invert: true },
      { label: "Current Ratio", value: health["Current Ratio"] ?? null, benchmark: 1.5 },
      { label: "Free Cash Flow", value: health["Free Cash Flow"] ?? null, benchmark: null },
    );
  }

  return (
    <BentoCard>
      <h3 className="mb-5 flex items-center gap-2 text-lg font-semibold text-slate-400">
        <span>🏈</span> Valuation Football Field
      </h3>
      <div className="space-y-3">
        {metrics.map((m) => (
          <MetricBar key={m.label} {...m} />
        ))}
      </div>
      <p className="mt-4 text-[10px] text-slate-600">
        White marker = sector benchmark. Green = favorable, Red = caution relative to benchmark.
      </p>
    </BentoCard>
  );
}
