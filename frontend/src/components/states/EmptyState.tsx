/**
 * phase-44.1: <EmptyState/> -- consistent empty-result affordance.
 *
 * Replaces ~10 inline empty blocks identified in closure_roadmap §3.18
 * (e.g. /reports "No reports found yet.", /performance per-section empties,
 * /signals "Enter a ticker..." etc.).
 *
 * Pattern: centered Phosphor icon + 1-line heading + optional helper text +
 * optional call-to-action. Per frontend.md color rules: gray for unavailable
 * state, sky for actionable CTA.
 */
"use client";

import { clsx } from "clsx";
import { MagnifyingGlass } from "@/lib/icons";
import type { Icon } from "@/lib/icons";

interface EmptyStateProps {
  icon?: Icon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({ icon: IconCmp = MagnifyingGlass, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={clsx(
        "flex flex-col items-center justify-center text-center px-6 py-12",
        "text-slate-400",
        className,
      )}
      role="status"
    >
      <IconCmp size={40} weight="light" aria-hidden="true" />
      <h3 className="mt-3 text-sm font-medium text-slate-300">{title}</h3>
      {description ? (
        <p className="mt-1 text-xs text-slate-500 max-w-md">{description}</p>
      ) : null}
      {action ? (
        <button
          type="button"
          onClick={action.onClick}
          className={clsx(
            "mt-4 inline-flex items-center gap-1 px-3 py-1.5 rounded-lg",
            "text-xs font-medium text-sky-300 bg-sky-950/40 hover:bg-sky-950/60",
            "border border-sky-900/40",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400",
            "min-h-[24px] min-w-[24px]",
          )}
        >
          {action.label}
        </button>
      ) : null}
    </div>
  );
}
