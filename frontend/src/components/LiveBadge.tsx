"use client";

// phase-44.0 foundation -- LiveBadge component
//
// Used by phase-44.2 cockpit on each position row to show
// `live` or `stale` per the freshness band from
// /api/paper-trading/freshness. Color-coded per
// .claude/rules/frontend.md: green=live (band !== red),
// amber=stale (band === amber), red=stale (band === red),
// gray=unknown.
//
// Single source of truth so phase-44.4 reports + phase-44.5
// trading + phase-44.6 analyze can all reuse without forking.

export type FreshnessBand = "green" | "amber" | "red" | "unknown";

export interface LiveBadgeProps {
  band: FreshnessBand;
  // Optional age in seconds for the tooltip.
  ageSec?: number | null;
  // Optional override for the label.
  label?: string;
  // Compact mode renders just the dot (no label) -- for dense tables.
  compact?: boolean;
}

const BAND_STYLES: Record<FreshnessBand, { dot: string; chip: string; ariaLabel: string }> = {
  green: {
    dot: "bg-emerald-500",
    chip: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
    ariaLabel: "live",
  },
  amber: {
    dot: "bg-amber-500",
    chip: "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30",
    ariaLabel: "stale (amber band)",
  },
  red: {
    dot: "bg-rose-500",
    chip: "bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/30",
    ariaLabel: "stale (red band)",
  },
  unknown: {
    dot: "bg-zinc-400",
    chip: "bg-zinc-500/10 text-zinc-600 dark:text-zinc-400 border-zinc-500/30",
    ariaLabel: "unknown",
  },
};

function formatAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "n/a";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86_400).toFixed(1)}d`;
}

export function LiveBadge({ band, ageSec, label, compact = false }: LiveBadgeProps) {
  const style = BAND_STYLES[band];
  const text = label ?? (band === "green" ? "live" : band === "unknown" ? "unknown" : "stale");
  const ageStr = formatAge(ageSec);
  const title = ageSec !== undefined && ageSec !== null
    ? `Last update: ${ageStr} ago (band: ${band})`
    : `Band: ${band}`;

  if (compact) {
    return (
      <span
        role="status"
        aria-label={style.ariaLabel}
        title={title}
        className={`inline-block w-2 h-2 rounded-full ${style.dot}`}
      />
    );
  }

  return (
    <span
      role="status"
      aria-label={style.ariaLabel}
      title={title}
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-xs font-medium ${style.chip}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} aria-hidden="true" />
      <span>{text}</span>
      {ageSec !== undefined && ageSec !== null && (
        <span className="text-[10px] opacity-70 tabular-nums">{ageStr}</span>
      )}
    </span>
  );
}
