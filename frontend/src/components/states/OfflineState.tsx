/**
 * phase-44.1: <OfflineState/> -- shown when backend health-check has been
 * failing for >30s OR navigator.onLine === false.
 *
 * Per closure_roadmap §2.2 + frontend_ux_master_design Section 2.2.
 */
"use client";

import { clsx } from "clsx";
import { CloudArrowDown, ArrowClockwise } from "@/lib/icons";

interface OfflineStateProps {
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export function OfflineState({ message, onRetry, className }: OfflineStateProps) {
  const text = message ?? "Backend unreachable. Check that uvicorn is running on port 8000.";
  return (
    <div
      role="alert"
      aria-live="polite"
      className={clsx(
        "rounded-2xl border border-amber-900 bg-amber-950/50",
        "px-4 py-3 text-amber-100",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <CloudArrowDown size={18} weight="bold" className="text-amber-400 mt-0.5 shrink-0" aria-hidden="true" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">Offline</p>
          <p className="mt-0.5 text-xs text-amber-200">{text}</p>
        </div>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className={clsx(
              "inline-flex items-center gap-1 px-3 py-1.5 rounded-lg",
              "text-xs font-medium text-amber-100 bg-amber-900/40 hover:bg-amber-900/60",
              "border border-amber-900",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400",
              "min-h-[24px] min-w-[24px]",
              "shrink-0",
            )}
          >
            <ArrowClockwise size={14} weight="bold" aria-hidden="true" />
            <span>Retry</span>
          </button>
        ) : null}
      </div>
    </div>
  );
}
