"use client";

import { useState } from "react";
import { clsx } from "clsx";
import type { DecisionTrace } from "@/lib/types";
import { TraceIcon } from "@/lib/icons";

function confidenceColor(c: number): string {
  if (c >= 0.7) return "text-emerald-400";
  if (c >= 0.4) return "text-amber-400";
  return "text-rose-400";
}

function confidenceBg(c: number): string {
  if (c >= 0.7) return "bg-emerald-500";
  if (c >= 0.4) return "bg-amber-500";
  return "bg-rose-500";
}

function signalBadge(signal: string): string {
  const s = signal.toUpperCase();
  if (s.includes("BULLISH") || s.includes("BUY") || s.includes("BREAKOUT") || s.includes("RISING") || s.includes("CONFIDENT") || s.includes("OPPORTUNITY"))
    return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  if (s.includes("BEARISH") || s.includes("SELL") || s.includes("DECLINING") || s.includes("EVASIVE") || s.includes("RISK"))
    return "text-rose-400 bg-rose-500/10 border-rose-500/30";
  return "text-amber-400 bg-amber-500/10 border-amber-500/30";
}

function TraceDetail({ trace }: { trace: DecisionTrace }) {
  const sourceUrl = trace.source_url;
  return (
    <div className="mt-4 space-y-3 border-t border-navy-700 pt-3">
      {trace.timestamp ? (
        <div className="text-[10px] text-slate-600">
          Timestamp: {new Date(trace.timestamp).toLocaleString()}
        </div>
      ) : null}

      {trace.evidence_citations && trace.evidence_citations.length > 0 ? (
        <div>
          <h5 className="mb-1 text-xs font-semibold text-sky-400">Evidence Citations</h5>
          <ul className="space-y-1">
            {trace.evidence_citations.map((e, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-slate-400">
                <span className="mt-0.5 text-sky-500">•</span>
                <span>{e}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {trace.reasoning_steps && trace.reasoning_steps.length > 0 ? (
        <div>
          <h5 className="mb-1 text-xs font-semibold text-purple-400">Reasoning Chain</h5>
          <ol className="space-y-1">
            {trace.reasoning_steps.map((step, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                <span className="font-mono text-[10px] text-purple-500">{i + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      {sourceUrl ? (
        <div className="text-xs">
          <span className="text-slate-500">Source: </span>
          <a
            href={sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sky-400 underline hover:text-sky-300"
          >
            View original data ↗
          </a>
        </div>
      ) : null}

      {trace.input_data_hash ? (
        <div className="text-[10px] text-slate-600">
          Input hash: <span className="font-mono">{trace.input_data_hash}</span>
        </div>
      ) : null}
    </div>
  );
}

export function DecisionTraceView({ traces }: { traces: DecisionTrace[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (!traces || traces.length === 0) {
    return (
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
        No decision traces available for this analysis.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-200">
          <TraceIcon size={20} weight="duotone" className="text-sky-400" />
          Decision Trace — Full Audit Trail
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Every agent&apos;s inputs, reasoning, and outputs are visible below. Click any step to expand.
        </p>
        <div className="mt-2 flex items-center gap-4 text-xs text-slate-500">
          <span>{traces.length} pipeline steps</span>
          <span>·</span>
          <span>Total latency: {(traces.reduce((sum, t) => sum + (t.latency_ms || 0), 0) / 1000).toFixed(1)}s</span>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical timeline line */}
        <div className="absolute left-5 top-0 bottom-0 w-0.5 bg-navy-700" />

        {traces.map((trace, idx) => {
          const isExpanded = expandedIdx === idx;
          return (
            <div key={idx} className="relative mb-3 ml-10">
              {/* Timeline dot */}
              <div
                className={clsx(
                  "absolute -left-[26px] top-4 h-3 w-3 rounded-full border-2 border-navy-800",
                  confidenceBg(trace.confidence)
                )}
              />

              <div
                className={clsx(
                  "cursor-pointer rounded-xl border border-navy-700 bg-navy-800/60 p-4 transition-all hover:border-sky-500/30",
                  isExpanded && "border-sky-500/40"
                )}
                onClick={() => setExpandedIdx(isExpanded ? null : idx)}
              >
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-slate-600">#{idx + 1}</span>
                    <span className="text-sm font-medium text-slate-200">
                      {trace.agent_name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={clsx("rounded-full border px-2 py-0.5 text-[10px] font-bold", signalBadge(trace.output_signal))}>
                      {trace.output_signal}
                    </span>
                    <span className={clsx("font-mono text-xs", confidenceColor(trace.confidence))}>
                      {Math.round(trace.confidence * 100)}%
                    </span>
                    {trace.latency_ms > 0 && (
                      <span className="text-[10px] text-slate-600">
                        {(trace.latency_ms / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="mt-2 h-1 w-full rounded-full bg-slate-700">
                  <div
                    className={clsx("h-1 rounded-full transition-all", confidenceBg(trace.confidence))}
                    style={{ width: `${trace.confidence * 100}%` }}
                  />
                </div>

                {/* Expanded detail */}
                {isExpanded ? <TraceDetail trace={trace} /> : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
