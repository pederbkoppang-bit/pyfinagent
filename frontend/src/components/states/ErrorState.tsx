/**
 * phase-44.1: <ErrorState/> -- consistent error banner.
 *
 * Replaces ~12 inline rose banners per closure_roadmap §3.18. Codifies the
 * phase-25.B12 retry + curl-hint pattern referenced in frontend.md "Error
 * states" rule.
 *
 * Renders a rose-bordered banner with: title, error detail (monospace),
 * optional curl-hint snippet, optional Retry button. role="alert" for
 * screen readers.
 */
"use client";

import { clsx } from "clsx";
import { XCircle, ArrowClockwise } from "@/lib/icons";

interface ErrorStateProps {
  title: string;
  /** Error message body. Rendered as preformatted text. */
  error?: string | null;
  /** Optional curl command to help operator reproduce / debug. */
  curlHint?: string;
  /** Optional retry handler. When provided, shows a "Retry" button. */
  onRetry?: () => void;
  retryLabel?: string;
  className?: string;
}

export function ErrorState({ title, error, curlHint, onRetry, retryLabel = "Retry", className }: ErrorStateProps) {
  return (
    <div
      role="alert"
      aria-live="assertive"
      className={clsx(
        "rounded-2xl border border-rose-900 bg-rose-950/50",
        "px-4 py-3",
        "text-rose-100",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <XCircle size={18} weight="bold" className="text-rose-400 mt-0.5 shrink-0" aria-hidden="true" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">{title}</p>
          {error ? (
            <pre className="mt-1 text-xs text-rose-200 whitespace-pre-wrap break-words font-mono">
              {error}
            </pre>
          ) : null}
          {curlHint ? (
            <details className="mt-2">
              <summary className="text-xs text-rose-300 cursor-pointer hover:text-rose-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 rounded">
                Reproduce with curl
              </summary>
              <pre className="mt-1 text-xs text-rose-200 bg-rose-950/70 rounded px-2 py-1 whitespace-pre-wrap break-words font-mono">
                {curlHint}
              </pre>
            </details>
          ) : null}
        </div>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className={clsx(
              "inline-flex items-center gap-1 px-3 py-1.5 rounded-lg",
              "text-xs font-medium text-rose-100 bg-rose-900/40 hover:bg-rose-900/60",
              "border border-rose-900",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400",
              "min-h-[24px] min-w-[24px]",
              "shrink-0",
            )}
          >
            <ArrowClockwise size={14} weight="bold" aria-hidden="true" />
            <span>{retryLabel}</span>
          </button>
        ) : null}
      </div>
    </div>
  );
}
