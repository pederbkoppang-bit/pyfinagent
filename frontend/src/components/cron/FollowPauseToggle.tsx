"use client";

// phase-44.7 -- explicit Follow / Pause toggle for log auto-scroll.
//
// Per researcher topic 1 (Grafana pattern, NOT CloudWatch click-anywhere):
// an explicit <button> with discoverable label + aria-pressed semantics
// + keyboard reachability. Default = follow. WCAG 2.2 24px target-size.

import { clsx } from "clsx";
import { Play, Pause } from "@/lib/icons";

export interface FollowPauseToggleProps {
  following: boolean;
  onToggle: () => void;
  className?: string;
}

export function FollowPauseToggle({
  following,
  onToggle,
  className,
}: FollowPauseToggleProps) {
  return (
    <button
      type="button"
      aria-pressed={following}
      aria-label={following ? "Pause auto-scroll" : "Resume auto-scroll"}
      onClick={onToggle}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md border px-3 py-1 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px]",
        following
          ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/20"
          : "bg-amber-500/10 text-amber-300 border-amber-500/30 hover:bg-amber-500/20",
        className,
      )}
    >
      {following ? (
        <>
          <Pause size={12} weight="fill" />
          <span>Following</span>
        </>
      ) : (
        <>
          <Play size={12} weight="fill" />
          <span>Paused</span>
        </>
      )}
    </button>
  );
}
