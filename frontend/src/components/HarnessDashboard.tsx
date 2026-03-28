"use client";

import { useEffect, useState } from "react";
import { BentoCard } from "@/components/BentoCard";
import {
  getHarnessLog,
  getHarnessCritique,
  getHarnessContract,
  getHarnessValidation,
} from "@/lib/api";
import type { HarnessCycle, HarnessValidation } from "@/lib/types";
import {
  CheckCircle,
  XCircle,
  Warning,
  ClockCounterClockwise,
  Target,
  FileText,
} from "@phosphor-icons/react";

// ── Verdict badge ───────────────────────────────────────────────

function VerdictBadge({ verdict }: { verdict?: string }) {
  if (!verdict) return null;
  const v = verdict.toUpperCase();
  if (v.includes("PASS")) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
        <CheckCircle size={14} weight="fill" /> PASS
      </span>
    );
  }
  if (v.includes("FAIL")) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-red-500/15 px-2.5 py-0.5 text-xs font-medium text-red-400">
        <XCircle size={14} weight="fill" /> FAIL
      </span>
    );
  }
  if (v.includes("CONDITIONAL") || v.includes("DRY_RUN")) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs font-medium text-amber-400">
        <Warning size={14} weight="fill" /> {v}
      </span>
    );
  }
  return (
    <span className="inline-flex rounded-full bg-slate-700 px-2.5 py-0.5 text-xs text-slate-400">
      {verdict}
    </span>
  );
}

// ── Score bar ───────────────────────────────────────────────────

function ScoreBar({ label, score, max = 10 }: { label: string; score: number; max?: number }) {
  const pct = Math.min(100, (score / max) * 100);
  const color =
    score >= 7 ? "bg-emerald-500" : score >= 5 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">
          {score}/{max}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-700">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── Validation table ────────────────────────────────────────────

function ValidationTable({ data }: { data: HarnessValidation }) {
  const sub = data.subperiod;
  if (!sub || Object.keys(sub).length === 0) return null;

  // Extract periods (skip metadata keys)
  const periods = Object.entries(sub).filter(
    ([k]) => k.startsWith("Period") || k.startsWith("Full")
  );

  return (
    <div className="overflow-hidden rounded-xl border border-navy-700">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-navy-700 bg-navy-800/80">
          <tr>
            <th className="px-4 py-2.5 font-medium text-slate-400">Period</th>
            <th className="px-4 py-2.5 text-right font-medium text-slate-400">Sharpe</th>
            <th className="px-4 py-2.5 text-right font-medium text-slate-400">DSR</th>
            <th className="px-4 py-2.5 text-right font-medium text-slate-400">Return</th>
            <th className="px-4 py-2.5 text-right font-medium text-slate-400">Max DD</th>
            <th className="px-4 py-2.5 text-right font-medium text-slate-400">Trades</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-navy-700/50">
          {periods.map(([name, vals]) => {
            const v = vals as Record<string, number>;
            const sharpe = v.sharpe ?? 0;
            return (
              <tr key={name} className="transition-colors hover:bg-navy-700/40">
                <td className="px-4 py-2.5 text-xs text-slate-300">{name}</td>
                <td
                  className={`px-4 py-2.5 text-right font-mono text-xs ${
                    sharpe >= 0.8
                      ? "text-emerald-400"
                      : sharpe >= 0.5
                        ? "text-amber-400"
                        : "text-red-400"
                  }`}
                >
                  {sharpe.toFixed(4)}
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-400">
                  {(v.dsr ?? 0).toFixed(4)}
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-300">
                  {(v.return_pct ?? 0).toFixed(1)}%
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-red-400">
                  {(v.max_drawdown_pct ?? 0).toFixed(1)}%
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-400">
                  {v.trades ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Markdown renderer (simple) ──────────────────────────────────

function MarkdownBlock({ content }: { content: string }) {
  // Very simple markdown: headers, bold, lists
  const lines = content.split("\n");
  return (
    <div className="space-y-1 text-xs text-slate-400">
      {lines.map((line, i) => {
        if (line.startsWith("# "))
          return (
            <p key={i} className="text-sm font-semibold text-slate-200 pt-2">
              {line.slice(2)}
            </p>
          );
        if (line.startsWith("## "))
          return (
            <p key={i} className="text-sm font-medium text-slate-300 pt-2">
              {line.slice(3)}
            </p>
          );
        if (line.startsWith("### "))
          return (
            <p key={i} className="text-xs font-medium text-slate-300 pt-1">
              {line.slice(4)}
            </p>
          );
        if (line.startsWith("- ") || line.startsWith("* "))
          return (
            <p key={i} className="pl-3">
              <span className="text-slate-600 mr-1">•</span>
              {line.slice(2)}
            </p>
          );
        if (line.trim() === "") return <div key={i} className="h-1" />;
        return <p key={i}>{line}</p>;
      })}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────

export function HarnessDashboard() {
  const [cycles, setCycles] = useState<HarnessCycle[]>([]);
  const [critique, setCritique] = useState<string | null>(null);
  const [contract, setContract] = useState<string | null>(null);
  const [validation, setValidation] = useState<HarnessValidation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getHarnessLog().catch(() => ({ cycles: [] })),
      getHarnessCritique().catch(() => ({ content: null, raw: null })),
      getHarnessContract().catch(() => ({ content: null })),
      getHarnessValidation().catch(() => ({ validation: {}, subperiod: {} })),
    ])
      .then(([log, crit, cont, val]) => {
        setCycles(log.cycles);
        setCritique(crit.content);
        setContract(cont.content);
        setValidation(val);
      })
      .catch(() => setError("Failed to load harness data"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-12 text-sm text-slate-500">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
        Loading harness data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-red-400">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current Contract */}
      {contract && (
        <BentoCard>
          <div className="flex items-center gap-2 mb-3">
            <Target size={18} className="text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-300">
              Current Contract
            </h3>
          </div>
          <div className="rounded-lg bg-navy-900/50 p-4 max-h-48 overflow-y-auto scrollbar-thin">
            <MarkdownBlock content={contract} />
          </div>
        </BentoCard>
      )}

      {/* Validation Results */}
      {validation &&
        Object.keys(validation.subperiod).length > 0 && (
          <BentoCard>
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle size={18} className="text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-300">
                Sub-Period Validation
              </h3>
              {validation.subperiod.test_date && (
                <span className="text-[10px] text-slate-500 ml-auto">
                  {String(validation.subperiod.test_date)}
                </span>
              )}
            </div>
            {validation.subperiod.code_version && (
              <p className="text-[10px] text-slate-500 mb-3">
                Code: {String(validation.subperiod.code_version)}
              </p>
            )}
            <ValidationTable data={validation} />
          </BentoCard>
        )}

      {/* Evaluator Critique */}
      {critique && (
        <BentoCard>
          <div className="flex items-center gap-2 mb-3">
            <FileText size={18} className="text-amber-400" />
            <h3 className="text-sm font-semibold text-slate-300">
              Latest Evaluator Critique
            </h3>
          </div>
          <div className="rounded-lg bg-navy-900/50 p-4 max-h-64 overflow-y-auto scrollbar-thin">
            <MarkdownBlock content={critique} />
          </div>
        </BentoCard>
      )}

      {/* Harness Cycles */}
      {cycles.length > 0 ? (
        <BentoCard>
          <div className="flex items-center gap-2 mb-4">
            <ClockCounterClockwise size={18} className="text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-300">
              Harness Cycles ({cycles.length})
            </h3>
          </div>
          <div className="space-y-4">
            {cycles.map((cycle, i) => (
              <details
                key={i}
                className="rounded-lg border border-slate-700/60 bg-navy-900/30"
              >
                <summary className="flex cursor-pointer items-center gap-3 px-4 py-3 text-sm">
                  <span className="font-medium text-slate-200">
                    {cycle.cycle}
                  </span>
                  <VerdictBadge verdict={cycle.verdict} />
                  <span className="ml-auto text-[10px] text-slate-500">
                    {cycle.timestamp}
                  </span>
                </summary>
                <div className="space-y-3 border-t border-slate-700/40 px-4 py-3">
                  {cycle.hypothesis && (
                    <div>
                      <p className="text-[10px] uppercase tracking-wider text-slate-500">
                        Hypothesis
                      </p>
                      <p className="text-xs text-slate-300">
                        {cycle.hypothesis}
                      </p>
                    </div>
                  )}
                  {cycle.generator && (
                    <div>
                      <p className="text-[10px] uppercase tracking-wider text-slate-500">
                        Generator
                      </p>
                      <p className="font-mono text-xs text-slate-400">
                        {cycle.generator}
                      </p>
                    </div>
                  )}
                  {Object.keys(cycle.scores).length > 0 && (
                    <div className="grid grid-cols-2 gap-3">
                      {Object.entries(cycle.scores).map(([k, v]) => (
                        <ScoreBar
                          key={k}
                          label={k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                          score={v}
                        />
                      ))}
                    </div>
                  )}
                  {cycle.decision && (
                    <div>
                      <p className="text-[10px] uppercase tracking-wider text-slate-500">
                        Decision
                      </p>
                      <p className="text-xs text-slate-300">
                        {cycle.decision}
                      </p>
                    </div>
                  )}
                  {cycle.duration && (
                    <p className="text-[10px] text-slate-500">
                      Duration: {cycle.duration}
                    </p>
                  )}
                </div>
              </details>
            ))}
          </div>
        </BentoCard>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <ClockCounterClockwise
            size={48}
            weight="duotone"
            className="text-slate-600"
          />
          <p className="mt-4 text-sm text-slate-400">No harness cycles yet</p>
          <p className="mt-1 text-xs text-slate-600">
            Run the harness to see cycle history here
          </p>
        </div>
      )}
    </div>
  );
}
