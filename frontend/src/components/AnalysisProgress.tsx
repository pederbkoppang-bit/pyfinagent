"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisStatusResponse, StepLogEntry } from "@/lib/types";
import type { Icon } from "@phosphor-icons/react";
import {
  StepMarketIntel, StepIngestion, StepQuant, StepRag,
  StepSentiment, StepCompetitor, StepEnrichment, StepInfoGap,
  StepSignalAnalysis, StepDebate, StepMacro, StepDeepDive,
  StepSynthesis, StepBiasAudit, StepRiskAssessment, IconTimer,
  IconCheck, IconX, IconCaretDown, IconCaretRight,
} from "@/lib/icons";

const ALL_STEPS: { key: string; label: string; icon: Icon }[] = [
  { key: "market_intel", label: "Market Intel", icon: StepMarketIntel },
  { key: "ingestion", label: "Ingestion", icon: StepIngestion },
  { key: "quant", label: "Financials", icon: StepQuant },
  { key: "rag", label: "Document Analysis", icon: StepRag },
  { key: "market", label: "Sentiment", icon: StepSentiment },
  { key: "competitor", label: "Competitors", icon: StepCompetitor },
  { key: "data_enrichment", label: "Data Enrichment", icon: StepEnrichment },
  { key: "info_gap", label: "Info-Gap Detection", icon: StepInfoGap },
  { key: "enrichment_analysis", label: "Signal Analysis", icon: StepSignalAnalysis },
  { key: "debate", label: "Agent Debate", icon: StepDebate },
  { key: "macro", label: "Macro Economy", icon: StepMacro },
  { key: "deep_dive", label: "Deep Dive", icon: StepDeepDive },
  { key: "synthesis", label: "Synthesis", icon: StepSynthesis },
  { key: "bias_audit", label: "Bias Audit", icon: StepBiasAudit },
  { key: "risk_assessment", label: "Risk Assessment", icon: StepRiskAssessment },
];

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function stepDuration(log: StepLogEntry[], stepKey: string): string | null {
  const entries = log.filter((e) => e.step === stepKey);
  const started = entries.find((e) => e.status === "started");
  const completed = entries.find((e) => e.status === "completed");
  if (!started || !completed) return null;
  const ms =
    new Date(completed.timestamp).getTime() -
    new Date(started.timestamp).getTime();
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function timeLabel(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}

export function AnalysisProgress({
  status,
}: {
  status: AnalysisStatusResponse;
}) {
  const completed = new Set(status.steps_completed);
  const current = status.current_step;
  const log = status.step_log ?? [];
  const pct =
    ALL_STEPS.length > 0
      ? Math.round((completed.size / ALL_STEPS.length) * 100)
      : 0;

  // Elapsed timer — stops when analysis completes or fails
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number>(Date.now());
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    startRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);
  useEffect(() => {
    if (
      (status.status === "completed" || status.status === "failed") &&
      timerRef.current
    ) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [status.status]);

  // Left panel: completed accordion toggle
  const [showCompleted, setShowCompleted] = useState(false);
  // Right panel: step filter (click completed step on left to filter)
  const [filterStep, setFilterStep] = useState<string | null>(null);

  // Categorise steps
  const completedSteps = ALL_STEPS.filter((s) => completed.has(s.key));
  const activeStep = ALL_STEPS.find(
    (s) => s.key === current && !completed.has(s.key)
  );
  const pendingSteps = ALL_STEPS.filter(
    (s) => !completed.has(s.key) && s.key !== current
  );

  // Total duration of completed steps
  const totalDuration = completedSteps.reduce((acc, s) => {
    const d = stepDuration(log, s.key);
    if (!d) return acc;
    return acc + (d.endsWith("ms") ? parseFloat(d) / 1000 : parseFloat(d));
  }, 0);

  // Right panel messages: filtered or all, reverse chronological (newest first)
  const allMessages = log.filter((e) => e.message);
  const displayMessages = (
    filterStep
      ? allMessages.filter((e) => e.step === filterStep)
      : allMessages
  ).slice().reverse();

  return (
    <div className="rounded-2xl border border-zinc-700/50 bg-zinc-800/70 backdrop-blur-lg flex flex-col h-[calc(100vh-200px)] min-h-[400px]">
      {/* ── Header ── */}
      <div className="px-5 pt-5 pb-4 shrink-0">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="flex items-center gap-2.5 text-lg font-semibold text-slate-200">
            {status.status === "running" && (
              <span className="gemini-spinner">
                <span className="gemini-bar" />
                <span className="gemini-bar" />
                <span className="gemini-bar" />
                <span className="gemini-bar" />
              </span>
            )}
            Analyzing {status.ticker}
          </h3>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-zinc-400 flex items-center gap-1"><IconTimer size={14} /> {formatTime(elapsed)}</span>
            <span className="font-mono text-sky-400">{pct}%</span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-1.5 w-full rounded-full bg-zinc-700">
          <div
            className="h-1.5 rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* ── Split layout: steps left, running log right ── */}
      <div className="flex flex-col md:grid md:grid-cols-[260px_1fr] border-t border-zinc-700/50 flex-1 min-h-0">
        {/* ── LEFT: No logs. Completed collapsed, active on top, pending below ── */}
        <div className="md:border-r border-b md:border-b-0 border-zinc-700/50 overflow-y-auto scrollbar-thin">
          <div className="py-2 px-2 space-y-0.5">
            {/* Completed steps: collapsible accordion */}
            {completedSteps.length > 0 && (
              <div className="mb-1">
                <button
                  onClick={() => setShowCompleted((v) => !v)}
                  className="flex items-center gap-2 w-full px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:bg-zinc-700/40 transition-colors"
                >
                  <span className="text-[10px]">
                    {showCompleted ? <IconCaretDown size={10} /> : <IconCaretRight size={10} />}
                  </span>
                  <IconCheck size={12} className="text-emerald-400" />
                  <span>
                    {completedSteps.length} completed
                  </span>
                  {totalDuration > 0 && (
                    <span className="ml-auto font-mono text-[10px] text-zinc-500">
                      {totalDuration.toFixed(1)}s
                    </span>
                  )}
                </button>
                {showCompleted && (
                  <div className="ml-2 mt-0.5 space-y-0.5">
                    {completedSteps.map((step) => {
                      const duration = stepDuration(log, step.key);
                      return (
                        <button
                          key={step.key}
                          onClick={() =>
                            setFilterStep((prev) =>
                              prev === step.key ? null : step.key
                            )
                          }
                          className={`flex items-center gap-2 w-full px-3 py-1 rounded-lg text-xs transition-colors ${
                            filterStep === step.key
                              ? "bg-sky-500/15 text-sky-300"
                              : "text-zinc-400 hover:bg-zinc-700/30"
                          }`}
                        >
                          <IconCheck size={12} className="text-emerald-400 shrink-0" />
                          <span className="w-4 text-center shrink-0">
                            <step.icon size={14} />
                          </span>
                          <span className="truncate">{step.label}</span>
                          {duration && (
                            <span className="ml-auto rounded-full bg-zinc-700/60 px-1.5 py-0.5 text-[10px] font-mono shrink-0">
                              {duration}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Active step: pinned at top (after completed accordion) */}
            {activeStep && (
              <div className="rounded-lg bg-sky-500/10 border border-sky-500/20 px-3 py-2">
                <div className="flex items-center gap-2">
                  <span className="relative flex h-3.5 w-3.5 items-center justify-center shrink-0">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-30" />
                    <span className="inline-flex h-1.5 w-1.5 rounded-full bg-sky-400" />
                  </span>
                  <span className="w-4 text-center text-sm shrink-0">
                    <activeStep.icon size={16} className="text-sky-300" />
                  </span>
                  <span className="font-medium text-sky-300 text-sm">
                    {activeStep.label}
                  </span>
                </div>
                <div className="ml-6 mt-1">
                  <span className="text-[11px] text-sky-400/50 animate-pulse">
                    thinking...
                  </span>
                </div>
              </div>
            )}

            {/* Pending steps: dimmed below active */}
            {pendingSteps.map((step) => (
              <div
                key={step.key}
                className="flex items-center gap-2 px-3 py-1 rounded-lg text-xs opacity-35"
              >
                <span className="text-zinc-600 shrink-0">&#x25CB;</span>
                <span className="w-4 text-center shrink-0">
                  <step.icon size={14} />
                </span>
                <span className="text-zinc-600 truncate">{step.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── RIGHT: Reverse chronological log (newest first) ── */}
        <div className="flex flex-col min-h-0 flex-1">
          {/* Panel header */}
          <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-700/40 shrink-0">
            <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Live Activity
            </span>
            {filterStep && (
              <button
                onClick={() => setFilterStep(null)}
                className="ml-2 flex items-center gap-1 rounded-full bg-sky-500/15 px-2 py-0.5 text-[10px] text-sky-300 hover:bg-sky-500/25 transition-colors"
              >
                {(() => { const StepIcon = ALL_STEPS.find((s) => s.key === filterStep)?.icon; return StepIcon ? <StepIcon size={12} /> : null; })()}
                {ALL_STEPS.find((s) => s.key === filterStep)?.label}
                <IconX size={10} className="ml-0.5" />
              </button>
            )}
            <span className="ml-auto text-[10px] text-zinc-600 font-mono">
              {displayMessages.length} events
            </span>
          </div>

          {/* Reverse chronological log */}
          <div className="flex-1 overflow-y-auto px-4 py-2 space-y-0.5 scrollbar-thin min-h-0">
            {/* Thinking indicator at the very top while running */}
            {status.status === "running" && !filterStep && (
              <div className="flex gap-2 text-xs items-center pb-1">
                <span className="text-sky-400/60 animate-pulse">●</span>
                <span className="text-sky-400/50 text-[11px] animate-pulse">
                  {status.message || "Processing..."}
                </span>
              </div>
            )}

            {displayMessages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <span className="text-sky-400/60 text-sm animate-pulse">
                  {filterStep ? "No activity for this step yet" : "Starting analysis..."}
                </span>
              </div>
            ) : (
              displayMessages.map((entry, i) => {
                const stepMeta = ALL_STEPS.find((s) => s.key === entry.step);
                const isStepStart = entry.status === "started";
                const isStepDone = entry.status === "completed";

                // Step transitions get a separator heading
                if (isStepStart) {
                  return (
                    <div key={i} className="flex items-center gap-2 pt-2 pb-1">
                      <span className="text-xs">{stepMeta ? <stepMeta.icon size={14} /> : <IconCaretRight size={14} />}</span>
                      <span className="text-xs font-medium text-zinc-300">
                        {stepMeta?.label ?? entry.step}
                      </span>
                      <div className="flex-1 border-t border-zinc-700/40" />
                      <span className="text-zinc-600 font-mono text-[10px]">
                        {timeLabel(entry.timestamp)}
                      </span>
                    </div>
                  );
                }

                // Step completion marker
                if (isStepDone) {
                  return (
                    <div key={i} className="flex items-center gap-2 pb-1 text-[11px]">
                      <IconCheck size={12} className="text-emerald-400/70" />
                      <span className="text-emerald-400/70">
                        {stepMeta?.label ?? entry.step} complete
                      </span>
                      <span className="ml-auto text-zinc-600 font-mono text-[10px]">
                        {timeLabel(entry.timestamp)}
                      </span>
                    </div>
                  );
                }

                // Regular log entry
                return (
                  <div key={i} className="flex gap-2 text-xs leading-relaxed">
                    <span className="text-zinc-600 font-mono shrink-0 text-[11px]">
                      {timeLabel(entry.timestamp)}
                    </span>
                    <span className="text-zinc-300">{entry.message}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
