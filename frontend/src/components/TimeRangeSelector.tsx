"use client";

// phase-44.4 -- segmented TimeRangeSelector.
//
// Per researcher topic 5: segmented control wins decisively over dropdown
// at 4 options (1 click vs 2 clicks + scroll). role=radiogroup semantics
// because the selector filters data (not switches panels). WCAG 2.2 24px
// target-size via min-h-[32px].

import { clsx } from "clsx";
import type { KeyboardEvent } from "react";

export type TimeRange = "7d" | "30d" | "90d" | "all";

const ORDERED: TimeRange[] = ["7d", "30d", "90d", "all"];

const LABELS: Record<TimeRange, string> = {
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
  all: "All",
};

const SHORT: Record<TimeRange, string> = {
  "7d": "7d",
  "30d": "30d",
  "90d": "90d",
  all: "All",
};

export interface TimeRangeSelectorProps {
  value: TimeRange;
  onChange: (next: TimeRange) => void;
  label?: string;
  className?: string;
}

export function TimeRangeSelector({
  value,
  onChange,
  label = "Time range",
  className,
}: TimeRangeSelectorProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      onChange(ORDERED[(idx + 1) % ORDERED.length]);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      onChange(ORDERED[(idx - 1 + ORDERED.length) % ORDERED.length]);
    } else if (e.key === "Home") {
      e.preventDefault();
      onChange(ORDERED[0]);
    } else if (e.key === "End") {
      e.preventDefault();
      onChange(ORDERED[ORDERED.length - 1]);
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={clsx(
        "inline-flex items-center gap-1 rounded-lg bg-navy-800/60 p-1",
        className,
      )}
    >
      {ORDERED.map((r, i) => {
        const checked = r === value;
        return (
          <button
            key={r}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={LABELS[r]}
            tabIndex={checked ? 0 : -1}
            onClick={() => onChange(r)}
            onKeyDown={(e) => handleKeyDown(e, i)}
            className={clsx(
              "min-h-[32px] rounded-md px-3 py-1 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40",
              checked
                ? "bg-sky-500/10 text-sky-400"
                : "text-slate-400 hover:text-slate-200",
            )}
          >
            {SHORT[r]}
          </button>
        );
      })}
    </div>
  );
}

// Helper: filter an array of date-stamped items by a TimeRange. dateKey
// = the property name carrying an ISO date or yyyy-mm-dd string.
export function filterByTimeRange<T>(
  items: T[],
  range: TimeRange,
  dateKey: keyof T,
): T[] {
  if (range === "all") return items;
  const days = range === "7d" ? 7 : range === "30d" ? 30 : 90;
  const cutoff = Date.now() - days * 86_400_000;
  return items.filter((item) => {
    const raw = (item as Record<string, unknown>)[dateKey as string];
    if (typeof raw !== "string") return false;
    const t = Date.parse(raw);
    if (Number.isNaN(t)) return false;
    return t >= cutoff;
  });
}
