"use client";

// phase-44.4 -- ReportCompareDrawer.
//
// Per master_design Section 3.8: the compare wizard belongs in a drawer
// (overlay), not as a tab equal to History. The drawer mirrors the
// AgentRationaleDrawer pattern: aria-modal=true + role=dialog +
// close-on-Escape + backdrop. The actual comparison rendering (price
// chart + radar + score bar) stays in the parent because it consumes
// heavy Recharts components -- the drawer just owns the selection UX.

import { useEffect } from "react";
import { IconX, IconCheck } from "@/lib/icons";
import type { ReportSummary } from "@/lib/types";

function scoreColor(r: string | null | undefined): string {
  if (!r) return "text-slate-400";
  if (r === "STRONG BUY" || r === "BUY") return "text-emerald-400";
  if (r === "STRONG SELL" || r === "SELL") return "text-rose-400";
  if (r === "HOLD") return "text-amber-400";
  return "text-slate-400";
}

export interface ReportCompareDrawerProps {
  open: boolean;
  onClose: () => void;
  reports: ReportSummary[];
  selected: Set<string>;
  onToggle: (key: string) => void;
  onStartCompare: () => void;
  comparing: boolean;
}

export function ReportCompareDrawer({
  open,
  onClose,
  reports,
  selected,
  onToggle,
  onStartCompare,
  comparing,
}: ReportCompareDrawerProps) {
  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="compare-drawer-title"
      className="fixed inset-0 z-50 flex items-stretch justify-end"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Panel */}
      <div className="relative flex h-full w-full max-w-2xl flex-col bg-navy-900 shadow-2xl border-l border-navy-700">
        <div className="flex items-center justify-between border-b border-navy-700 px-6 py-4">
          <div>
            <h2
              id="compare-drawer-title"
              className="text-lg font-semibold text-slate-100"
            >
              Compare reports
            </h2>
            <p className="text-xs text-slate-500">
              Select 2 or more reports to compare side-by-side.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close compare drawer"
            className="rounded-md p-2 text-slate-400 hover:bg-navy-800 hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px] min-w-[24px]"
          >
            <IconX size={18} weight="bold" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-4">
          {reports.length === 0 ? (
            <p className="text-sm text-slate-500">No reports available.</p>
          ) : (
            <div className="space-y-2">
              {reports.map((r) => {
                const key = `${r.ticker}|${r.analysis_date}`;
                const isSelected = selected.has(key);
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => onToggle(key)}
                    aria-pressed={isSelected}
                    className={`flex w-full items-center justify-between rounded-lg border p-3 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 min-h-[24px] ${
                      isSelected
                        ? "border-sky-500/50 bg-sky-500/10"
                        : "border-slate-800 bg-slate-900/50 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        aria-hidden="true"
                        className={`flex h-4 w-4 items-center justify-center rounded border text-[10px] ${
                          isSelected ? "border-sky-400 bg-sky-400 text-white" : "border-slate-600"
                        }`}
                      >
                        {isSelected && <IconCheck size={12} weight="bold" />}
                      </span>
                      <span className="font-mono font-bold text-slate-200">{r.ticker}</span>
                      {r.company_name && (
                        <span className="text-sm text-slate-400">{r.company_name}</span>
                      )}
                      <span className="text-xs text-slate-500">
                        {new Date(r.analysis_date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-sm text-sky-300">
                        {r.final_score.toFixed(2)}
                      </span>
                      <span className={`text-xs font-medium ${scoreColor(r.recommendation)}`}>
                        {r.recommendation}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="border-t border-navy-700 px-6 py-4 flex items-center justify-between">
          <p className="text-xs text-slate-500">
            {selected.size} selected
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-navy-800 min-h-[24px]"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => {
                onStartCompare();
                onClose();
              }}
              disabled={comparing || selected.size < 2}
              className="rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40 min-h-[24px]"
            >
              {comparing ? "Comparing..." : "Compare"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
