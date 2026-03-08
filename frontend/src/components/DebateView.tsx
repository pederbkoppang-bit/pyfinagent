"use client";

import { clsx } from "clsx";

interface Contradiction {
  topic: string;
  bull_view: string;
  bear_view: string;
  resolution: string;
  winner?: string;
}

interface Dissent {
  agent: string;
  position: string;
  reason: string;
}

interface CaseDetail {
  thesis: string;
  confidence: number;
  key_catalysts?: string[];
  key_threats?: string[];
  evidence?: Array<{ source: string; data_point: string; interpretation: string }>;
}

export interface DebateResult {
  bull_case: CaseDetail;
  bear_case: CaseDetail;
  consensus: string;
  consensus_confidence: number;
  contradictions: Contradiction[];
  dissent_registry: Dissent[];
  moderator_analysis?: string;
}

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 flex-1 rounded-full bg-slate-700">
        <div
          className={clsx("h-2 rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-xs text-slate-400">{pct}%</span>
    </div>
  );
}

function ConsensusLabel({ consensus }: { consensus: string }) {
  const c = consensus.toUpperCase();
  const color = c.includes("STRONG_BUY") || c.includes("BUY")
    ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
    : c.includes("STRONG_SELL") || c.includes("SELL")
    ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
    : "text-amber-400 bg-amber-500/10 border-amber-500/30";

  return (
    <span className={clsx("rounded-full border px-3 py-1 text-sm font-bold", color)}>
      {consensus}
    </span>
  );
}

export function DebateView({ debate }: { debate: DebateResult }) {
  if (!debate) return null;

  return (
    <div className="space-y-4">
      {/* Consensus Header */}
      <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-200">
              ⚖️ Agent Debate Consensus
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              4-round adversarial debate: Bull vs Bear with Moderator resolution
            </p>
          </div>
          <ConsensusLabel consensus={debate.consensus || "HOLD"} />
        </div>
        <div className="mt-4">
          <span className="text-xs text-slate-500">Consensus Confidence</span>
          <ConfidenceBar
            value={debate.consensus_confidence || 0.5}
            color="bg-gradient-to-r from-sky-500 to-cyan-400"
          />
        </div>
      </div>

      {/* Bull vs Bear Side-by-Side */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Bull Case */}
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h4 className="flex items-center gap-2 font-semibold text-emerald-400">
              <span>🐂</span> Bull Case
            </h4>
            <span className="font-mono text-xs text-emerald-400/70">
              {Math.round((debate.bull_case?.confidence || 0) * 100)}%
            </span>
          </div>
          <ConfidenceBar
            value={debate.bull_case?.confidence || 0}
            color="bg-emerald-500"
          />
          <p className="mt-3 text-sm leading-relaxed text-slate-300">
            {debate.bull_case?.thesis || "No bull case provided."}
          </p>
          {debate.bull_case?.key_catalysts && debate.bull_case.key_catalysts.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-emerald-400/70">Key Catalysts</span>
              <ul className="mt-1 space-y-1">
                {debate.bull_case.key_catalysts.slice(0, 5).map((c, i) => (
                  <li key={i} className="text-xs text-slate-400">
                    <span className="text-emerald-500">▸</span> {c}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Evidence with source provenance */}
          {debate.bull_case?.evidence && debate.bull_case.evidence.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-emerald-400/70">Evidence</span>
              <div className="mt-1 space-y-1.5">
                {debate.bull_case.evidence.slice(0, 4).map((e, i) => (
                  <div key={i} className="rounded-lg bg-emerald-500/5 p-2 text-xs">
                    <div className="flex items-center gap-1.5">
                      <span className="rounded bg-emerald-500/20 px-1 py-0.5 text-[9px] font-semibold text-emerald-400">
                        {e.source}
                      </span>
                    </div>
                    <p className="mt-0.5 text-slate-400">{e.data_point}</p>
                    <p className="mt-0.5 italic text-slate-500">{e.interpretation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Bear Case */}
        <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h4 className="flex items-center gap-2 font-semibold text-rose-400">
              <span>🐻</span> Bear Case
            </h4>
            <span className="font-mono text-xs text-rose-400/70">
              {Math.round((debate.bear_case?.confidence || 0) * 100)}%
            </span>
          </div>
          <ConfidenceBar
            value={debate.bear_case?.confidence || 0}
            color="bg-rose-500"
          />
          <p className="mt-3 text-sm leading-relaxed text-slate-300">
            {debate.bear_case?.thesis || "No bear case provided."}
          </p>
          {debate.bear_case?.key_threats && debate.bear_case.key_threats.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-rose-400/70">Key Threats</span>
              <ul className="mt-1 space-y-1">
                {debate.bear_case.key_threats.slice(0, 5).map((t, i) => (
                  <li key={i} className="text-xs text-slate-400">
                    <span className="text-rose-500">▸</span> {t}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Evidence with source provenance */}
          {debate.bear_case?.evidence && debate.bear_case.evidence.length > 0 && (
            <div className="mt-3">
              <span className="text-xs font-medium text-rose-400/70">Evidence</span>
              <div className="mt-1 space-y-1.5">
                {debate.bear_case.evidence.slice(0, 4).map((e, i) => (
                  <div key={i} className="rounded-lg bg-rose-500/5 p-2 text-xs">
                    <div className="flex items-center gap-1.5">
                      <span className="rounded bg-rose-500/20 px-1 py-0.5 text-[9px] font-semibold text-rose-400">
                        {e.source}
                      </span>
                    </div>
                    <p className="mt-0.5 text-slate-400">{e.data_point}</p>
                    <p className="mt-0.5 italic text-slate-500">{e.interpretation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Contradictions */}
      {debate.contradictions && debate.contradictions.length > 0 && (
        <div className="rounded-xl border border-amber-500/20 bg-navy-800/60 p-5">
          <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-amber-400">
            <span>⚡</span> Contradictions Identified
          </h4>
          <div className="space-y-3">
            {debate.contradictions.map((c, i) => (
              <div key={i} className="rounded-lg border border-navy-700 bg-navy-900/50 p-3">
                <div className="mb-1 text-xs font-medium text-slate-300">{c.topic}</div>
                <div className="grid gap-2 md:grid-cols-2">
                  <div className="text-xs text-emerald-400/70">
                    <span className="font-medium">Bull:</span> {c.bull_view}
                  </div>
                  <div className="text-xs text-rose-400/70">
                    <span className="font-medium">Bear:</span> {c.bear_view}
                  </div>
                </div>
                <div className="mt-2 text-xs text-sky-400/70">
                  <span className="font-medium">Resolution:</span> {c.resolution}
                  {c.winner && (
                    <span className={clsx(
                      "ml-2 rounded-full px-1.5 py-0.5 text-[10px] font-bold",
                      c.winner === "bull"
                        ? "bg-emerald-500/20 text-emerald-400"
                        : "bg-rose-500/20 text-rose-400"
                    )}>
                      {c.winner.toUpperCase()} WINS
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dissent Registry */}
      {debate.dissent_registry && debate.dissent_registry.length > 0 && (
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <h4 className="mb-3 text-sm font-semibold text-slate-300">
            📋 Dissent Registry
          </h4>
          <div className="space-y-2">
            {debate.dissent_registry.map((d, i) => (
              <div key={i} className="flex items-start gap-3 text-xs">
                <span className="rounded bg-slate-700 px-1.5 py-0.5 font-medium text-slate-300">
                  {d.agent}
                </span>
                <span className={clsx(
                  "rounded px-1.5 py-0.5 font-bold",
                  d.position?.toUpperCase().includes("BEAR")
                    ? "bg-rose-500/20 text-rose-400"
                    : "bg-emerald-500/20 text-emerald-400"
                )}>
                  {d.position}
                </span>
                <span className="text-slate-500">{d.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
