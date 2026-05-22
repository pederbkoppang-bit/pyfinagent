/**
 * phase-44.1: <LoadingState/> -- consistent loading affordance.
 *
 * Replaces the ~30+ inline "Loading..." patterns found across pages per
 * closure_roadmap §3.18 + frontend_ux_master_design Section 2.2.
 *
 * Per frontend.md: use Skeleton.tsx pattern for content placeholders; this
 * component provides the IN-PAGE spinner + label for transient operations
 * where a skeleton doesn't fit.
 */
"use client";

import { clsx } from "clsx";

interface LoadingStateProps {
  label?: string;
  /** "inline" (default) = small dot+text, fits a row. "card" = full BentoCard frame. */
  variant?: "inline" | "card";
  className?: string;
}

export function LoadingState({ label = "Loading...", variant = "inline", className }: LoadingStateProps) {
  if (variant === "card") {
    return (
      <div
        className={clsx(
          "bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-6",
          "flex items-center gap-3 text-sm text-zinc-500 dark:text-zinc-400",
          className,
        )}
        role="status"
        aria-live="polite"
      >
        <span
          className="inline-block w-2 h-2 rounded-full bg-sky-500 animate-pulse"
          aria-hidden="true"
        />
        <span>{label}</span>
      </div>
    );
  }
  return (
    <div
      className={clsx(
        "inline-flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400",
        className,
      )}
      role="status"
      aria-live="polite"
    >
      <span
        className="inline-block w-2 h-2 rounded-full bg-sky-500 animate-pulse"
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  );
}
