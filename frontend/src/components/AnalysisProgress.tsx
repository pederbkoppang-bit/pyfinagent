"use client";

import type { AnalysisStatusResponse } from "@/lib/types";

const ALL_STEPS = [
  { key: "market_intel", label: "Market Intel", icon: "📰" },
  { key: "ingestion", label: "Ingestion", icon: "📥" },
  { key: "quant", label: "Financials", icon: "🔢" },
  { key: "rag", label: "Document Analysis", icon: "📄" },
  { key: "market", label: "Sentiment", icon: "🎯" },
  { key: "competitor", label: "Competitors", icon: "🏆" },
  { key: "data_enrichment", label: "Data Enrichment", icon: "📡" },
  { key: "enrichment_analysis", label: "Signal Analysis", icon: "🧠" },
  { key: "debate", label: "Agent Debate", icon: "⚖️" },
  { key: "macro", label: "Macro Economy", icon: "🌍" },
  { key: "deep_dive", label: "Deep Dive", icon: "🔍" },
  { key: "synthesis", label: "Synthesis", icon: "🧪" },
  { key: "bias_audit", label: "Bias Audit", icon: "🛡️" },
];

export function AnalysisProgress({
  status,
}: {
  status: AnalysisStatusResponse;
}) {
  const completed = new Set(status.steps_completed);
  const current = status.current_step;
  const pct =
    ALL_STEPS.length > 0
      ? Math.round((completed.size / ALL_STEPS.length) * 100)
      : 0;

  return (
    <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
          {status.status === "running" && (
            <span className="gemini-spinner">
              <span className="gemini-bar" />
              <span className="gemini-bar" />
              <span className="gemini-bar" />
              <span className="gemini-bar" />
            </span>
          )}
          Analysis Progress
        </h3>
        <span className="font-mono text-sm text-sky-400">{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="mb-6 h-2 w-full rounded-full bg-slate-700">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Step list */}
      <div className="space-y-2">
        {ALL_STEPS.map((step) => {
          const isDone = completed.has(step.key);
          const isCurrent = current === step.key && !isDone;

          return (
            <div key={step.key} className="flex items-center gap-3 text-sm">
              {isDone ? (
                <span className="text-emerald-400">✓</span>
              ) : isCurrent ? (
                <span className="animate-spin-slow text-sky-400">◌</span>
              ) : (
                <span className="text-slate-600">○</span>
              )}
              <span className="w-5 text-center">{step.icon}</span>
              <span
                className={
                  isDone
                    ? "text-slate-300"
                    : isCurrent
                    ? "font-medium text-sky-300"
                    : "text-slate-600"
                }
              >
                {step.label}
              </span>
              {isCurrent && (
                <span className="ml-auto text-xs text-sky-400/60">
                  in progress...
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
