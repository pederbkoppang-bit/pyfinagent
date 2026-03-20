"use client";

import { useState } from "react";
import { clsx } from "clsx";
import type { ScoringMatrix } from "@/lib/types";
import type { Icon } from "@phosphor-icons/react";
import { BentoCard } from "./BentoCard";
import {
  PillarCorporate, PillarIndustry, PillarValuation,
  PillarSentiment, PillarGovernance, DebateConsensus,
} from "@/lib/icons";

interface PillarConfig {
  key: keyof ScoringMatrix;
  label: string;
  weight: number;
  icon: Icon;
  sources: string[];
  description: string;
}

const PILLARS: PillarConfig[] = [
  {
    key: "pillar_1_corporate", label: "Corporate Profile", weight: 0.35, icon: PillarCorporate,
    sources: ["RAG Agent (10-K/10-Q)", "yfinance Fundamentals", "Earnings Tone"],
    description: "Business model strength, moat durability, management quality, financial health",
  },
  {
    key: "pillar_2_industry", label: "Industry & Macro", weight: 0.20, icon: PillarIndustry,
    sources: ["Competitor Agent", "Sector Analysis", "FRED Macro", "Alt Data"],
    description: "Competitive landscape, market share trajectory, sector tailwinds/headwinds",
  },
  {
    key: "pillar_3_valuation", label: "Valuation", weight: 0.20, icon: PillarValuation,
    sources: ["yfinance Fundamentals", "Monte Carlo VaR", "Quant Agent"],
    description: "P/E, PEG, FCF yield, price-to-book relative to growth, margin of safety",
  },
  {
    key: "pillar_4_sentiment", label: "Market Sentiment", weight: 0.15, icon: PillarSentiment,
    sources: ["Insider Activity", "Options Flow", "NLP Sentiment", "Social Sentiment"],
    description: "News/social sentiment, insider activity, options flow, institutional signals",
  },
  {
    key: "pillar_5_governance", label: "Governance", weight: 0.10, icon: PillarGovernance,
    sources: ["RAG Agent (10-K/10-Q)", "Insider Activity"],
    description: "Executive compensation alignment, board independence, shareholder structure",
  },
];

function scoreGrade(score: number): { color: string; bg: string } {
  if (score >= 8) return { color: "text-emerald-400", bg: "bg-emerald-400" };
  if (score >= 6) return { color: "text-sky-400", bg: "bg-sky-400" };
  if (score >= 4) return { color: "text-amber-400", bg: "bg-amber-400" };
  return { color: "text-rose-400", bg: "bg-rose-400" };
}

interface EvaluationTableProps {
  scores: ScoringMatrix;
  previousScores?: ScoringMatrix | null;
}

export function EvaluationTable({ scores, previousScores }: EvaluationTableProps) {
  const [expandedPillar, setExpandedPillar] = useState<string | null>(null);

  const weightedTotal = PILLARS.reduce((sum, p) => sum + scores[p.key] * p.weight, 0);

  return (
    <BentoCard>
      <div className="mb-5 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-400">
          <DebateConsensus size={20} weight="duotone" className="text-slate-400" /> Evaluation Score Breakdown
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Weighted Score:</span>
          <span className={clsx("font-mono text-lg font-bold", scoreGrade(weightedTotal).color)}>
            {weightedTotal.toFixed(2)}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-5 gap-3">
        {PILLARS.map((p) => {
          const value = scores[p.key];
          const { color, bg } = scoreGrade(value);
          const pct = (value / 10) * 100;
          const isExpanded = expandedPillar === p.key;
          const prevValue = previousScores?.[p.key];
          const delta = prevValue != null ? value - prevValue : null;

          return (
            <div
              key={p.key}
              className={clsx(
                "flex cursor-pointer flex-col items-center rounded-xl border bg-slate-900/50 p-4 transition-all hover:border-sky-500/30",
                isExpanded ? "border-sky-500/40 col-span-5 sm:col-span-2" : "border-slate-800"
              )}
              onClick={() => setExpandedPillar(isExpanded ? null : p.key)}
            >
              <p.icon size={22} weight="duotone" className="mb-1 text-slate-400" />
              <p className="text-center text-xs font-medium text-slate-400">
                {p.label}
              </p>
              <div className="mt-3 flex items-baseline gap-1.5">
                <p className={`font-mono text-3xl font-bold ${color}`}>
                  {value.toFixed(1)}
                </p>
                {delta != null && delta !== 0 && (
                  <span className={clsx("font-mono text-xs", delta > 0 ? "text-emerald-400" : "text-rose-400")}>
                    {delta > 0 ? "+" : ""}{delta.toFixed(1)}
                  </span>
                )}
              </div>
              <div className="mt-2 h-1.5 w-full rounded-full bg-slate-700">
                <div
                  className={`h-1.5 rounded-full ${bg}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <p className="mt-2 font-mono text-[10px] text-slate-500">
                Weight: {(p.weight * 100).toFixed(0)}%
              </p>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="mt-3 w-full space-y-2 border-t border-slate-800 pt-3 text-left">
                  <p className="text-xs text-slate-400">{p.description}</p>
                  <div>
                    <p className="text-[10px] font-semibold text-slate-500">Contributing Sources</p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {p.sources.map((s) => (
                        <span key={s} className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-400">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </BentoCard>
  );
}
