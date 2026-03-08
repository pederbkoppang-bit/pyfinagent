"use client";

import { clsx } from "clsx";
import type { BiasReportData, ConflictReportData } from "@/lib/types";

function SeverityBadge({ severity }: { severity: string }) {
  const s = severity.toUpperCase();
  const color =
    s === "HIGH"
      ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
      : s === "MEDIUM"
      ? "text-amber-400 bg-amber-500/10 border-amber-500/30"
      : "text-sky-400 bg-sky-500/10 border-sky-500/30";

  return (
    <span className={clsx("rounded-full border px-2 py-0.5 text-xs font-semibold", color)}>
      {severity}
    </span>
  );
}

function BiasTypeIcon({ type }: { type: string }) {
  const icons: Record<string, string> = {
    tech_bias: "🖥️",
    confirmation_bias: "🔄",
    recency_bias: "⏰",
    anchoring: "⚓",
    source_diversity: "📊",
  };
  return <span className="text-lg">{icons[type] || "⚠️"}</span>;
}

function ReliabilityBadge({ reliability }: { reliability: string }) {
  const r = reliability.toUpperCase();
  const color =
    r === "HIGH"
      ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
      : r === "MEDIUM"
      ? "text-amber-400 bg-amber-500/10 border-amber-500/30"
      : "text-rose-400 bg-rose-500/10 border-rose-500/30";

  return (
    <span className={clsx("rounded-full border px-3 py-1 text-sm font-bold", color)}>
      {reliability} Reliability
    </span>
  );
}

export function BiasReport({
  biasReport,
  conflictReport,
}: {
  biasReport?: BiasReportData;
  conflictReport?: ConflictReportData;
}) {
  if (!biasReport && !conflictReport) return null;

  return (
    <div className="space-y-6">
      {/* Bias Flags */}
      {biasReport && (
        <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
              🛡️ LLM Bias Audit
            </h3>
            <div className="flex items-center gap-3">
              {biasReport.raw_score != null && biasReport.adjusted_score != null && (
                <span className="text-xs text-slate-500">
                  Raw: {biasReport.raw_score} → Adjusted: {biasReport.adjusted_score}
                </span>
              )}
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs font-semibold",
                  biasReport.bias_count === 0
                    ? "bg-emerald-500/10 text-emerald-400"
                    : biasReport.bias_count <= 2
                    ? "bg-amber-500/10 text-amber-400"
                    : "bg-rose-500/10 text-rose-400"
                )}
              >
                {biasReport.bias_count} bias flag{biasReport.bias_count !== 1 ? "s" : ""}
              </span>
            </div>
          </div>

          {biasReport.flags.length === 0 ? (
            <p className="text-sm text-slate-500">
              No significant biases detected. Analysis appears balanced.
            </p>
          ) : (
            <div className="space-y-3">
              {biasReport.flags.map((flag, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-navy-700 bg-navy-900/50 p-4"
                >
                  <div className="mb-2 flex items-center gap-2">
                    <BiasTypeIcon type={flag.bias_type} />
                    <span className="text-sm font-medium text-slate-200">
                      {flag.bias_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </span>
                    <SeverityBadge severity={flag.severity} />
                  </div>
                  <p className="mb-1 text-sm text-slate-300">{flag.description}</p>
                  <p className="text-xs text-slate-500">{flag.evidence}</p>
                  {flag.adjustment_suggestion && (
                    <p className="mt-2 text-xs text-sky-400/80">
                      💡 {flag.adjustment_suggestion}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Knowledge Conflicts */}
      {conflictReport && conflictReport.conflicts.length > 0 && (
        <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
              ⚡ Knowledge Conflicts
            </h3>
            <ReliabilityBadge reliability={conflictReport.overall_reliability} />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-navy-700 text-xs uppercase text-slate-500">
                  <th className="pb-2 pr-4">Field</th>
                  <th className="pb-2 pr-4">LLM Belief</th>
                  <th className="pb-2 pr-4">Actual Data</th>
                  <th className="pb-2 pr-4">Severity</th>
                  <th className="pb-2">Explanation</th>
                </tr>
              </thead>
              <tbody>
                {conflictReport.conflicts.map((c, i) => (
                  <tr
                    key={i}
                    className="border-b border-navy-700/50 text-slate-300"
                  >
                    <td className="py-2 pr-4 font-mono text-xs">{c.field}</td>
                    <td className="py-2 pr-4 text-amber-400/80">{c.llm_belief}</td>
                    <td className="py-2 pr-4 text-emerald-400/80">{c.actual_value}</td>
                    <td className="py-2 pr-4">
                      <SeverityBadge severity={c.severity} />
                    </td>
                    <td className="py-2 text-xs text-slate-400">{c.explanation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
