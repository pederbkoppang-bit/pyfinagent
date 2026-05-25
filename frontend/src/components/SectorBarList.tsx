"use client";

// phase-44.0 foundation -- Tremor BarList wrapper for sector concentration.
//
// Used by phase-44.2 cockpit (right column) to show which GICS sectors
// are at or near the per-sector NAV-pct cap. Color codes amber when a
// sector is within 5pp of paper_max_per_sector_nav_pct; red when over.
//
// Single source of truth for the sector-cap visualization so phase-44.4
// reports can reuse the same chart shape.

import { BarList, Card } from "@tremor/react";

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

function colorFor(valuePct: number, capPct: number, amberZonePct: number): "emerald" | "amber" | "rose" {
  if (valuePct >= capPct) return "rose";
  if (valuePct >= capPct - amberZonePct) return "amber";
  return "emerald";
}

export function SectorBarList({
  items,
  capPct,
  amberZonePct = 5,
  title = "Sector concentration",
  emptyState,
  className,
}: SectorBarListProps) {
  if (items.length === 0) {
    return (
      <Card className={className}>
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">{title}</h3>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          {emptyState ?? "No positions yet."}
        </p>
      </Card>
    );
  }

  // Tremor BarList accepts color via per-item `color` field.
  const sorted = [...items].sort((a, b) => b.value - a.value);
  const decorated = sorted.map((item) => ({
    ...item,
    color: colorFor(item.value, capPct, amberZonePct),
    // BarList shows the raw value -- pre-format as `%` string for clarity
    // when the consumer doesn't supply a valueFormatter.
  }));

  return (
    <Card className={className}>
      <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
        {title}
      </h3>
      <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mb-3">
        Cap: {capPct.toFixed(0)}% per sector (amber within {amberZonePct}pp; red at/over).
      </p>
      <BarList
        data={decorated as unknown as Array<{ name: string; value: number; color?: string; href?: string }>}
        valueFormatter={(n: number) => `${n.toFixed(1)}%`}
        aria-label="Sector concentration bar list"
      />
    </Card>
  );
}
