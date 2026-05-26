"use client";

// phase-44.2 cycle 68 -- portfolio allocation donut.
//
// Shows the operator the share of NAV held in each sector + the cash
// pocket. Uses Tremor DonutChart (already a dep). Center label shows
// total NAV; the legend below shows sector + percent. Per researcher
// brief Section A.3 -- DonutChart is the right primitive (already
// rendered the per-sector amber/red bars on the SectorBarList; this
// surfaces the SAME data in a more glance-able shape).

import { DonutChart } from "@tremor/react";
import { useMemo } from "react";

export interface AllocationSlice {
  name: string;       // sector or "Cash"
  value: number;      // dollar value (will be normalized to %)
}

export interface PortfolioAllocationDonutProps {
  slices: AllocationSlice[];
  totalNav: number | null;
  title?: string;
  className?: string;
}

// Stable per-sector color palette. Cash is the neutral slate. Tremor's
// `colors` prop expects entries in the same order as the data slices.
//
// IMPORTANT (cycle 68 Q/A finding): the legend dot's bg class is a
// LITERAL utility string per token -- Tailwind's JIT scanner does NOT
// pick up dynamically-built `bg-${color}-500` concatenations, which
// would silently break the rendered dot color for any color the
// scanner hadn't seen elsewhere. The DOT_BG_CLASS map below gives JIT
// a static reference for every color we ever emit.
const SECTOR_COLOR_MAP: Record<string, string> = {
  Technology: "blue",
  Industrials: "amber",
  Financials: "indigo",
  Healthcare: "emerald",
  "Consumer Discretionary": "fuchsia",
  "Consumer Staples": "lime",
  Energy: "orange",
  Materials: "yellow",
  "Communication Services": "cyan",
  Utilities: "violet",
  "Real Estate": "rose",
  Unknown: "slate",
  Cash: "slate",
};

const DOT_BG_CLASS: Record<string, string> = {
  blue: "bg-blue-500",
  amber: "bg-amber-500",
  indigo: "bg-indigo-500",
  emerald: "bg-emerald-500",
  fuchsia: "bg-fuchsia-500",
  lime: "bg-lime-500",
  orange: "bg-orange-500",
  yellow: "bg-yellow-500",
  cyan: "bg-cyan-500",
  violet: "bg-violet-500",
  rose: "bg-rose-500",
  slate: "bg-slate-500",
  pink: "bg-pink-500",
  teal: "bg-teal-500",
  sky: "bg-sky-500",
  purple: "bg-purple-500",
};

function colorFor(name: string, fallbackIdx: number): string {
  if (SECTOR_COLOR_MAP[name]) return SECTOR_COLOR_MAP[name];
  // Deterministic fallback for unmapped sectors
  const palette = ["pink", "teal", "sky", "purple"];
  return palette[fallbackIdx % palette.length];
}

export function PortfolioAllocationDonut({
  slices,
  totalNav,
  title = "Allocation",
  className,
}: PortfolioAllocationDonutProps) {
  const { data, colors, totalValue } = useMemo(() => {
    const filtered = slices.filter((s) => s.value > 0);
    const sorted = [...filtered].sort((a, b) => b.value - a.value);
    const total = sorted.reduce((sum, s) => sum + s.value, 0);
    const cols = sorted.map((s, i) => colorFor(s.name, i));
    return { data: sorted, colors: cols, totalValue: total };
  }, [slices]);

  const navForCenter = totalNav ?? totalValue;

  const containerClass = `rounded-xl border border-navy-700 bg-navy-800/70 p-4 ${className ?? ""}`;

  if (data.length === 0 || totalValue <= 0) {
    return (
      <div className={containerClass}>
        <h3 className="text-sm font-medium text-slate-300 mb-2">{title}</h3>
        <p className="text-sm text-slate-400">No allocation data yet.</p>
      </div>
    );
  }

  return (
    <div className={containerClass} role="region" aria-label={title}>
      <h3 className="text-sm font-medium text-slate-300 mb-1">{title}</h3>
      <p className="text-[11px] text-slate-400 mb-3">
        NAV split by sector + cash (% of total).
      </p>
      <div className="flex items-center gap-4">
        <DonutChart
          data={data}
          category="value"
          index="name"
          colors={colors}
          valueFormatter={(n: number) =>
            navForCenter > 0
              ? `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })} (${(
                  (n / navForCenter) *
                  100
                ).toFixed(1)}%)`
              : `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          }
          className="h-32 w-32"
          showAnimation={false}
          variant="donut"
          label={
            navForCenter > 0
              ? `$${navForCenter.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
              : ""
          }
        />
        <ul className="flex-1 min-w-0 space-y-1 text-xs">
          {data.map((s, i) => {
            const pct = totalValue > 0 ? (s.value / totalValue) * 100 : 0;
            // Legend dot color matches the DonutChart slice color. Static
            // lookup (not template-string interpolation) so Tailwind JIT
            // picks up every utility in DOT_BG_CLASS.
            const dotBg = DOT_BG_CLASS[colors[i]] ?? "bg-slate-500";
            const dotClass = `inline-block w-2 h-2 rounded-full ${dotBg} shrink-0`;
            return (
              <li
                key={s.name}
                className="flex items-center gap-2 text-slate-300"
              >
                <span className={dotClass} aria-hidden="true" />
                <span className="flex-1 truncate">{s.name}</span>
                <span className="font-mono tabular-nums text-slate-100">
                  {pct.toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
