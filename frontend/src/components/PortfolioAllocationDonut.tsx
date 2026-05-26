"use client";

// phase-44.2 cycle 70 -- inline-SVG portfolio allocation donut.
//
// Replaces the cycle-69 Tremor DonutChart. Tremor's chart-color
// resolution did not light up the SVG <path> fills even after the
// node_modules content-path fix (operator-flagged 2026-05-26: donut
// rendered uniform dark, tooltip escaped the card with white-on-dark
// styling). This rewrite follows the cycle-63 SectorBarList Option B
// precedent: take ownership of the SVG so we control every pixel.
//
// Math: each <circle> has r = 100/(2*PI) so circumference = 100, making
// stroke-dasharray = "percent 100-percent" + stroke-dashoffset = -running
// total a direct, easy-to-reason-about mapping (see Medium / Mark Caron
// "Scratch-made SVG Donut & Pie Charts" + research_brief_phase_44_2_donut.md).
//
// Accessibility: outer SVG carries role="img" + aria-label summarizing
// the chart for screen readers; the legend list below is the canonical
// SR data path. Tooltip uses role="tooltip" + ESC dismissibility per
// WCAG SC 1.4.13.

import { useCallback, useMemo, useState, type KeyboardEvent } from "react";
import { LiveBadge, type FreshnessBand } from "@/components/LiveBadge";

export interface AllocationSlice {
  name: string;       // sector or "Cash"
  value: number;      // dollar value (will be normalized to %)
}

export interface PortfolioAllocationDonutProps {
  slices: AllocationSlice[];
  totalNav: number | null;
  title?: string;
  className?: string;
  // phase-72: optional live-freshness badge in the card header so the operator
  // can immediately tell whether the center label is live or stale-snapshot.
  liveBand?: FreshnessBand;
  liveAgeSec?: number | null;
}

// Per-sector color tokens. Cash is neutral slate. Maintained in parallel
// with DOT_BG_CLASS (legend dots) + SLICE_STROKE_CLASS (donut slices) so
// Tailwind JIT picks up every utility literal.
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

// JIT-safe lookup maps -- one for the legend dot (bg-) and one for the
// SVG slice (stroke-). Tailwind's content scanner picks these up as
// literal class strings; template-string concatenation would NOT work
// (cycle-68 lesson, codified in .claude/rules/frontend.md cycle-69 section).
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

const SLICE_STROKE_CLASS: Record<string, string> = {
  blue: "stroke-blue-500",
  amber: "stroke-amber-500",
  indigo: "stroke-indigo-500",
  emerald: "stroke-emerald-500",
  fuchsia: "stroke-fuchsia-500",
  lime: "stroke-lime-500",
  orange: "stroke-orange-500",
  yellow: "stroke-yellow-500",
  cyan: "stroke-cyan-500",
  violet: "stroke-violet-500",
  rose: "stroke-rose-500",
  slate: "stroke-slate-500",
  pink: "stroke-pink-500",
  teal: "stroke-teal-500",
  sky: "stroke-sky-500",
  purple: "stroke-purple-500",
};

function colorFor(name: string, fallbackIdx: number): string {
  if (SECTOR_COLOR_MAP[name]) return SECTOR_COLOR_MAP[name];
  const palette = ["pink", "teal", "sky", "purple"];
  return palette[fallbackIdx % palette.length];
}

function fmtDollars(n: number): string {
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

// Constants: r = 100/(2*PI) so the circle's circumference is exactly 100,
// making stroke-dasharray = "<pct> <100-pct>" a direct mapping. cx/cy
// chosen so the viewBox is 42x42 (radius ~15.92, plus stroke padding).
const RADIUS = 100 / (2 * Math.PI);
const CX = 21;
const CY = 21;
const STROKE_WIDTH = 4;

export function PortfolioAllocationDonut({
  slices,
  totalNav,
  title = "Allocation",
  className,
  liveBand,
  liveAgeSec,
}: PortfolioAllocationDonutProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { data, colors, totalValue } = useMemo(() => {
    const filtered = slices.filter((s) => s.value > 0);
    const sorted = [...filtered].sort((a, b) => b.value - a.value);
    const total = sorted.reduce((sum, s) => sum + s.value, 0);
    const cols = sorted.map((s, i) => colorFor(s.name, i));
    return { data: sorted, colors: cols, totalValue: total };
  }, [slices]);

  const navForCenter = totalNav ?? totalValue;

  // Pre-compute per-slice percentage + accumulated offset for stroke
  // positioning.
  const arcs = useMemo(() => {
    let acc = 0;
    return data.map((s, i) => {
      const pct = totalValue > 0 ? (s.value / totalValue) * 100 : 0;
      const offset = -acc; // negative because SVG strokes go counter-clockwise by default
      acc += pct;
      return {
        ...s,
        pct,
        offset,
        color: colors[i],
      };
    });
  }, [data, colors, totalValue]);

  const handleEsc = useCallback((e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape" && hoverIdx !== null) {
      setHoverIdx(null);
    }
  }, [hoverIdx]);

  const containerClass = `h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/70 p-4 ${className ?? ""}`;

  if (data.length === 0 || totalValue <= 0) {
    return (
      <div className={containerClass}>
        <h3 className="text-sm font-medium text-slate-300 mb-2">{title}</h3>
        <p className="text-sm text-slate-400">No allocation data yet.</p>
      </div>
    );
  }

  // Build the screen-reader summary string for the outer SVG.
  const ariaSummary = arcs
    .map((a) => `${a.name} ${a.pct.toFixed(1)} percent`)
    .join(", ");

  const hovered = hoverIdx !== null ? arcs[hoverIdx] : null;

  return (
    <div
      className={containerClass}
      role="region"
      aria-label={title}
      onKeyDown={handleEsc}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        {liveBand && (
          <LiveBadge band={liveBand} ageSec={liveAgeSec ?? null} compact />
        )}
      </div>
      <p className="text-[11px] text-slate-400 mb-3">
        NAV split by sector + cash (% of total).
      </p>
      <div className="flex items-center gap-4 flex-1">
        <div className="relative flex-shrink-0">
          <svg
            viewBox="0 0 42 42"
            className="h-32 w-32"
            role="img"
            aria-label={`Portfolio allocation: ${ariaSummary}`}
          >
            {/* Background track so any gap reads as the card bg, not white */}
            <circle
              cx={CX}
              cy={CY}
              r={RADIUS}
              fill="transparent"
              className="stroke-navy-900"
              strokeWidth={STROKE_WIDTH}
            />
            {arcs.map((a, i) => {
              const stroke = SLICE_STROKE_CLASS[a.color] ?? "stroke-slate-500";
              const isHover = hoverIdx === i;
              const isOtherHover = hoverIdx !== null && hoverIdx !== i;
              return (
                <circle
                  key={`${a.name}-${i}`}
                  cx={CX}
                  cy={CY}
                  r={RADIUS}
                  fill="transparent"
                  strokeWidth={isHover ? STROKE_WIDTH + 1.2 : STROKE_WIDTH}
                  // stroke-dasharray = "<this-slice-pct> <rest>" makes the
                  // dash show exactly this slice's segment.
                  strokeDasharray={`${a.pct.toFixed(4)} ${(100 - a.pct).toFixed(4)}`}
                  strokeDashoffset={a.offset.toFixed(4)}
                  // Rotate so first slice starts at 12 o'clock (rather than
                  // 3 o'clock which is the SVG default).
                  transform={`rotate(-90 ${CX} ${CY})`}
                  className={`${stroke} transition-opacity duration-150 ${
                    isOtherHover ? "opacity-40" : "opacity-100"
                  } cursor-pointer`}
                  onMouseEnter={() => setHoverIdx(i)}
                  onMouseLeave={() => setHoverIdx(null)}
                  onFocus={() => setHoverIdx(i)}
                  onBlur={() => setHoverIdx(null)}
                  tabIndex={0}
                  role="graphics-symbol"
                  aria-label={`${a.name} ${a.pct.toFixed(1)} percent`}
                >
                  <title>{`${a.name}: ${fmtDollars(a.value)} (${a.pct.toFixed(1)}%)`}</title>
                </circle>
              );
            })}
            {/* Center label */}
            <text
              x={CX}
              y={CY - 0.5}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-100"
              style={{ fontSize: "4.4px", fontWeight: 600 }}
            >
              {hovered
                ? `${hovered.pct.toFixed(0)}%`
                : fmtDollars(navForCenter)}
            </text>
            <text
              x={CX}
              y={CY + 4}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-400"
              style={{ fontSize: "2.8px" }}
            >
              {hovered ? hovered.name : "NAV"}
            </text>
          </svg>
        </div>
        <ul className="flex-1 min-w-0 space-y-1 text-xs">
          {data.map((s, i) => {
            const pct = totalValue > 0 ? (s.value / totalValue) * 100 : 0;
            const dotBg = DOT_BG_CLASS[colors[i]] ?? "bg-slate-500";
            const isHover = hoverIdx === i;
            return (
              <li
                key={s.name}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(null)}
                className={`flex items-center gap-2 text-slate-300 rounded px-1 ${
                  isHover ? "bg-navy-700/40" : ""
                }`}
              >
                <span
                  className={`inline-block w-2 h-2 rounded-full ${dotBg} shrink-0`}
                  aria-hidden="true"
                />
                <span className="flex-1 truncate">{s.name}</span>
                <span className="font-mono tabular-nums text-slate-100">
                  {pct.toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ul>
      </div>
      {/* Tooltip lives under the chart -- inline so it stays inside the
          card border (cycle-69 operator-flagged: Tremor's portaled tooltip
          escaped the card with white-on-dark styling). WCAG SC 1.4.13:
          hoverable via mouseenter; dismissible via Escape (handleEsc on
          the container); persistent until mouseleave. */}
      {hovered && (
        <div
          role="tooltip"
          className="mt-3 rounded-md border border-navy-700 bg-navy-900 px-3 py-2 text-xs"
        >
          <div className="flex items-center justify-between gap-3">
            <span className="font-medium text-slate-200">{hovered.name}</span>
            <span className="font-mono tabular-nums text-slate-100">
              {fmtDollars(hovered.value)}
            </span>
          </div>
          <div className="mt-1 text-[11px] text-slate-400">
            {hovered.pct.toFixed(1)}% of {fmtDollars(totalValue)} total
          </div>
        </div>
      )}
    </div>
  );
}
