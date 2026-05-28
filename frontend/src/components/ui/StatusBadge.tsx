/**
 * phase-47.5: canonical status pill. Consumes tokens.status so GoLiveGate
 * checks, KillSwitch state, CycleStatus, and JobStatus share ONE vocabulary
 * (ux_roadmap.md W2). Pre-attentive color coding (frontend.md): green=ok,
 * amber=warn, rose=error, slate=neutral, sky=info.
 */
"use client";

import { clsx } from "clsx";
import type { ReactNode } from "react";

import { tokens, type StatusVariant } from "@/lib/design-tokens";

interface StatusBadgeProps {
  variant: StatusVariant;
  children: ReactNode;
  className?: string;
}

export function StatusBadge({ variant, children, className }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium",
        tokens.status[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
