/**
 * phase-44.1: <StaleDataState/> -- inline badge for data that's older than
 * its SLA threshold but still usable.
 *
 * Per closure_roadmap §2.2 + closure_roadmap §3.15 (/observability per-source
 * band coloring). Standardizes the "live (3s)" / "stale (60s)" pill pattern
 * referenced in frontend_ux_master_design Section 2.7 (LiveBadge precursor).
 */
"use client";

import { clsx } from "clsx";
import { IconTimer } from "@/lib/icons";

interface StaleDataStateProps {
  /** Seconds since data was last refreshed. */
  ageSeconds: number | null;
  /** SLA threshold in seconds; data older than this is "stale". */
  slaSeconds?: number;
  /** Compact "3s" / "5m" / "2h" timestamp formatter. */
  formatAge?: (seconds: number) => string;
  className?: string;
}

const defaultFormatAge = (s: number): string => {
  if (s < 1) return "now";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
};

export function StaleDataState({ ageSeconds, slaSeconds = 60, formatAge = defaultFormatAge, className }: StaleDataStateProps) {
  const isStale = ageSeconds !== null && ageSeconds > slaSeconds;
  const isUnknown = ageSeconds === null;

  const palette = isUnknown
    ? "text-zinc-400 bg-zinc-900/40 border-zinc-800"
    : isStale
      ? "text-amber-300 bg-amber-950/40 border-amber-900"
      : "text-emerald-300 bg-emerald-950/40 border-emerald-900";

  const label = isUnknown ? "no data" : isStale ? `stale (${formatAge(ageSeconds!)})` : `live (${formatAge(ageSeconds!)})`;
  const ariaLabel = isUnknown
    ? "No data freshness available"
    : isStale
      ? `Stale data: refreshed ${formatAge(ageSeconds!)} ago, beyond ${formatAge(slaSeconds)} SLA`
      : `Live data: refreshed ${formatAge(ageSeconds!)} ago`;

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full",
        "text-[10px] font-medium border min-h-[24px]",
        palette,
        className,
      )}
      role="status"
      aria-label={ariaLabel}
    >
      <IconTimer size={10} weight="bold" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}
