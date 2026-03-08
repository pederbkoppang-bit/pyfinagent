"use client";

import type { SynthesisReport } from "@/lib/types";
import { BentoCard } from "./BentoCard";

function formatNumber(num: number | null | undefined): string {
  if (num == null) return "N/A";
  if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
  return num.toLocaleString();
}

function scoreColor(action: string): string {
  const lower = action.toLowerCase();
  if (lower.includes("buy")) return "text-emerald-400";
  if (lower.includes("sell")) return "text-rose-400";
  return "text-slate-400";
}

export function AlphaScoreCard({ report }: { report: SynthesisReport }) {
  const score = report.final_weighted_score ?? 0;
  const pct = (score / 10) * 100;
  const action = report.recommendation.action;

  return (
    <BentoCard glow className="flex flex-col justify-between">
      <div>
        <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-400">
          <span>🧠</span> Alpha Score
        </h3>
        <p className="mt-4 font-mono text-7xl font-bold text-sky-300">
          {score.toFixed(2)}
        </p>
      </div>
      <div className="mt-6">
        <p className={`text-xl font-semibold ${scoreColor(action)}`}>
          {action}
        </p>
        <div className="mt-2 h-2.5 w-full rounded-full bg-slate-700">
          <div
            className="h-2.5 rounded-full bg-gradient-to-r from-sky-500 to-cyan-400"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </BentoCard>
  );
}

export function InvestmentThesisCard({
  report,
  financials,
}: {
  report: SynthesisReport;
  financials?: { revenue?: number; net_income?: number; market_cap?: number };
}) {
  return (
    <BentoCard className="flex flex-col">
      <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-400">
        <span>🔬</span> Investment Thesis
      </h3>
      <p className="mb-6 text-sm leading-relaxed text-slate-300">
        {report.final_summary}
      </p>
      {financials && (
        <div className="mt-auto grid grid-cols-3 gap-4 border-t border-slate-800 pt-4">
          <div className="text-center">
            <p className="text-xs text-slate-400">Revenue</p>
            <p className="font-mono text-lg font-semibold text-emerald-400">
              {formatNumber(financials.revenue)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400">Net Income</p>
            <p className="font-mono text-lg font-semibold text-emerald-400">
              {formatNumber(financials.net_income)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400">Market Cap</p>
            <p className="font-mono text-lg font-semibold text-sky-400">
              {formatNumber(financials.market_cap)}
            </p>
          </div>
        </div>
      )}
    </BentoCard>
  );
}

export function ScoringMatrixCard({ report }: { report: SynthesisReport }) {
  const pillars = [
    { key: "pillar_1_corporate", label: "Corporate", weight: "35%" },
    { key: "pillar_2_industry", label: "Industry", weight: "20%" },
    { key: "pillar_3_valuation", label: "Valuation", weight: "20%" },
    { key: "pillar_4_sentiment", label: "Sentiment", weight: "15%" },
    { key: "pillar_5_governance", label: "Governance", weight: "10%" },
  ] as const;

  return (
    <BentoCard>
      <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-400">
        <span>📊</span> Scoring Matrix
      </h3>
      <div className="space-y-3">
        {pillars.map((p) => {
          const value =
            report.scoring_matrix[p.key as keyof typeof report.scoring_matrix];
          const pct = (value / 10) * 100;
          return (
            <div key={p.key}>
              <div className="flex justify-between text-sm">
                <span className="text-slate-300">
                  {p.label}{" "}
                  <span className="text-slate-500">({p.weight})</span>
                </span>
                <span className="font-mono text-sky-300">
                  {value.toFixed(1)}
                </span>
              </div>
              <div className="mt-1 h-2 w-full rounded-full bg-slate-700">
                <div
                  className="h-2 rounded-full bg-sky-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </BentoCard>
  );
}

export function RisksCard({ risks }: { risks: string[] }) {
  return (
    <BentoCard>
      <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-400">
        <span>⚠️</span> Key Risks
      </h3>
      <ul className="space-y-3">
        {risks.map((risk, i) => (
          <li
            key={i}
            className="rounded-lg border border-rose-900/50 bg-rose-950/30 p-3 text-sm text-rose-200"
          >
            {risk}
          </li>
        ))}
      </ul>
    </BentoCard>
  );
}
