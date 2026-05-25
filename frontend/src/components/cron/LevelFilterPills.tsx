"use client";

// phase-44.7 -- ERROR / WARN / INFO level filter pills for /cron logs.
//
// Per researcher topic 1 (Grafana Explore Logs queryless filtering) +
// MDN ARIA role=group. Each pill is a toggle; clicking flips inclusion.
// role="group" wraps the 3 pills; each button uses aria-pressed (NOT
// aria-checked -- these are independent toggles, not a radio set).
// WCAG 2.2 24px target-size via min-h-[24px].

import { clsx } from "clsx";
import type { LogLevel } from "./density-helpers";

export interface LevelFilterPillsProps {
  active: Set<LogLevel>;
  onToggle: (level: LogLevel) => void;
  className?: string;
}

interface PillSpec {
  level: NonNullable<LogLevel>;
  label: string;
  activeClass: string;
  inactiveClass: string;
}

const PILLS: PillSpec[] = [
  {
    level: "ERROR",
    label: "ERROR",
    activeClass: "bg-rose-500/15 text-rose-300 border-rose-500/40",
    inactiveClass: "bg-slate-800 text-slate-400 border-slate-700 hover:text-rose-300",
  },
  {
    level: "WARN",
    label: "WARN",
    activeClass: "bg-amber-500/15 text-amber-300 border-amber-500/40",
    inactiveClass: "bg-slate-800 text-slate-400 border-slate-700 hover:text-amber-300",
  },
  {
    level: "INFO",
    label: "INFO",
    activeClass: "bg-sky-500/15 text-sky-300 border-sky-500/40",
    inactiveClass: "bg-slate-800 text-slate-400 border-slate-700 hover:text-sky-300",
  },
];

export function LevelFilterPills({
  active,
  onToggle,
  className,
}: LevelFilterPillsProps) {
  return (
    <div
      role="group"
      aria-label="Log level filter"
      className={clsx("inline-flex items-center gap-2", className)}
    >
      {PILLS.map((p) => {
        const isActive = active.has(p.level);
        return (
          <button
            key={p.level}
            type="button"
            aria-pressed={isActive}
            aria-label={`Toggle ${p.label} filter`}
            onClick={() => onToggle(p.level)}
            className={clsx(
              "rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px]",
              isActive ? p.activeClass : p.inactiveClass,
            )}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}
