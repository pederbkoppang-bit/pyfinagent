"use client";

// phase-44.0 foundation -- sector concentration bar list.
//
// phase-44.2 Option B rewrite: replaced the Tremor BarList internals with a
// Tailwind-only horizontal-bar grid. Tremor BarList does NOT support
// per-item color (verified against
// github.com/tremorlabs/tremor/blob/main/src/components/BarList/BarList.tsx
// on 2026-05-25; the Bar<T> type lacks a `color` field and all bars
// hard-code `bg-blue-200 dark:bg-blue-900`). The amber-at-5pp-below-cap /
// red-at-or-over signal is the criticality marker UX-DoD criterion 8
// requires -- without per-item color it is a uniform-blue chart that
// undermines the cap visualization.
//
// Public API (props) unchanged so existing consumers + tests survive.

import { useMemo } from "react";

export interface SectorBarItem {
  name: string;          // GICS sector name (e.g., "Technology")
  value: number;         // NAV percentage 0..100
  href?: string;         // optional click-through to a sector detail page
}

export interface SectorBarListProps {
  items: SectorBarItem[];
  capPct: number;        // settings.paper_max_per_sector_nav_pct (e.g., 30)
  amberZonePct?: number; // amber starts at capPct - this; default 5
  title?: string;
  emptyState?: React.ReactNode;
  className?: string;
}

type Band = "emerald" | "amber" | "rose";

function bandFor(valuePct: number, capPct: number, amberZonePct: number): Band {
  if (valuePct >= capPct) return "rose";
  if (valuePct >= capPct - amberZonePct) return "amber";
  return "emerald";
}

// Per-band Tailwind class triples (bar fill / value text). Kept as
// explicit string literals so Tailwind's content scan keeps them.
const BAR_CLASS: Record<Band, string> = {
  emerald: "bg-emerald-500/80",
  amber: "bg-amber-500/80",
  rose: "bg-rose-500/85",
};

const VALUE_TEXT_CLASS: Record<Band, string> = {
  emerald: "text-emerald-400",
  amber: "text-amber-400",
  rose: "text-rose-400",
};

export function SectorBarList({
  items,
  capPct,
  amberZonePct = 5,
  title = "Sector concentration",
  emptyState,
  className,
}: SectorBarListProps) {
  const sortedDecorated = useMemo(() => {
    const sorted = [...items].sort((a, b) => b.value - a.value);
    const maxValue = Math.max(capPct, ...sorted.map((s) => s.value), 1);
    return sorted.map((item) => ({
      ...item,
      band: bandFor(item.value, capPct, amberZonePct),
      widthPct: Math.min(100, (item.value / maxValue) * 100),
    }));
  }, [items, capPct, amberZonePct]);

  // phase-44.2 cycle-67 UX-audit fix: project is dark-mode-only; drop
  // bg-white + zinc fallbacks that were conflicting with consumer
  // className. The consumer-passed className still extends the
  // container so callers can tweak (border-x, padding, etc.).
  const containerClass = `rounded-xl border border-navy-700 bg-navy-800/70 p-4 ${className ?? ""}`;

  if (items.length === 0) {
    return (
      <div className={containerClass}>
        <h3 className="text-sm font-medium text-slate-300 mb-2">{title}</h3>
        <p className="text-sm text-slate-400">
          {emptyState ?? "No positions yet."}
        </p>
      </div>
    );
  }

  return (
    <div
      className={containerClass}
      role="region"
      aria-label={title}
    >
      <h3 className="text-sm font-medium text-slate-300 mb-1">
        {title}
      </h3>
      <p className="text-[11px] text-slate-400 mb-3">
        Cap: {capPct.toFixed(0)}% per sector (amber within {amberZonePct}pp; red at/over).
      </p>
      <ul className="space-y-2" aria-label="Sector concentration bar list">
        {sortedDecorated.map((item) => {
          const Row = (
            <div className="flex items-center gap-3">
              <span className="w-28 truncate text-xs text-slate-300" title={item.name}>
                {item.name}
              </span>
              <div
                className="relative flex-1 h-5 rounded bg-navy-900 overflow-hidden"
                role="progressbar"
                aria-valuenow={item.value}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${item.name}: ${item.value.toFixed(1)}% of NAV`}
                data-band={item.band}
              >
                <div
                  className={`h-full ${BAR_CLASS[item.band]}`}
                  style={{ width: `${item.widthPct}%` }}
                />
              </div>
              <span className={`w-14 text-right font-mono text-xs ${VALUE_TEXT_CLASS[item.band]}`}>
                {`${item.value.toFixed(1)}%`}
              </span>
            </div>
          );
          return (
            <li key={item.name} className="block">
              {item.href ? (
                <a
                  href={item.href}
                  className="block rounded hover:bg-navy-700/50 px-1 py-0.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40"
                >
                  {Row}
                </a>
              ) : (
                <div className="px-1 py-0.5">{Row}</div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
