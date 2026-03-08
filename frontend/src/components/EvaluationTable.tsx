"use client";

import type { ScoringMatrix } from "@/lib/types";
import { BentoCard } from "./BentoCard";

interface PillarConfig {
  key: keyof ScoringMatrix;
  label: string;
  weight: number;
  icon: string;
}

const PILLARS: PillarConfig[] = [
  { key: "pillar_1_corporate", label: "Corporate Profile", weight: 0.35, icon: "🏢" },
  { key: "pillar_2_industry", label: "Industry & Macro", weight: 0.20, icon: "🌐" },
  { key: "pillar_3_valuation", label: "Valuation", weight: 0.20, icon: "💰" },
  { key: "pillar_4_sentiment", label: "Market Sentiment", weight: 0.15, icon: "📊" },
  { key: "pillar_5_governance", label: "Governance", weight: 0.10, icon: "🏛️" },
];

function scoreGrade(score: number): { color: string; bg: string } {
  if (score >= 8) return { color: "text-emerald-400", bg: "bg-emerald-400" };
  if (score >= 6) return { color: "text-sky-400", bg: "bg-sky-400" };
  if (score >= 4) return { color: "text-amber-400", bg: "bg-amber-400" };
  return { color: "text-rose-400", bg: "bg-rose-400" };
}

export function EvaluationTable({ scores }: { scores: ScoringMatrix }) {
  return (
    <BentoCard>
      <h3 className="mb-5 flex items-center gap-2 text-lg font-semibold text-slate-400">
        <span>⚖️</span> Evaluation Score Breakdown
      </h3>
      <div className="grid grid-cols-5 gap-3">
        {PILLARS.map((p) => {
          const value = scores[p.key];
          const { color, bg } = scoreGrade(value);
          const pct = (value / 10) * 100;

          return (
            <div
              key={p.key}
              className="flex flex-col items-center rounded-xl border border-slate-800 bg-slate-900/50 p-4"
            >
              <span className="mb-1 text-xl">{p.icon}</span>
              <p className="text-center text-xs font-medium text-slate-400">
                {p.label}
              </p>
              <p className={`mt-3 font-mono text-3xl font-bold ${color}`}>
                {value.toFixed(1)}
              </p>
              <div className="mt-2 h-1.5 w-full rounded-full bg-slate-700">
                <div
                  className={`h-1.5 rounded-full ${bg}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <p className="mt-2 font-mono text-[10px] text-slate-500">
                Weight: {(p.weight * 100).toFixed(0)}%
              </p>
            </div>
          );
        })}
      </div>
    </BentoCard>
  );
}
