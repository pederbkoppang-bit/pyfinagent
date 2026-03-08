"use client";

import { clsx } from "clsx";
import type { SynthesisReport } from "@/lib/types";

interface ReportHeaderProps {
  ticker: string;
  report: SynthesisReport;
  financials?: Record<string, unknown> | null;
}

function formatNumber(num: number | null | undefined): string {
  if (num == null) return "N/A";
  if (Math.abs(num) >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
  if (Math.abs(num) >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (Math.abs(num) >= 1e6) return `$${(num / 1e6).toFixed(1)}M`;
  return `$${num.toLocaleString()}`;
}

function actionColor(action: string): string {
  const a = action.toUpperCase();
  if (a.includes("STRONG") && a.includes("BUY"))
    return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
  if (a.includes("BUY"))
    return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  if (a.includes("STRONG") && a.includes("SELL"))
    return "bg-rose-500/20 text-rose-300 border-rose-500/40";
  if (a.includes("SELL"))
    return "bg-rose-500/15 text-rose-400 border-rose-500/30";
  return "bg-amber-500/15 text-amber-400 border-amber-500/30";
}

function scoreRing(score: number) {
  const pct = (score / 10) * 100;
  const radius = 40;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (pct / 100) * circ;
  const color =
    score >= 7 ? "stroke-emerald-400" : score >= 5 ? "stroke-sky-400" : score >= 3 ? "stroke-amber-400" : "stroke-rose-400";

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width="100" height="100" className="-rotate-90">
        <circle cx="50" cy="50" r={radius} fill="none" stroke="#1e293b" strokeWidth="6" />
        <circle
          cx="50" cy="50" r={radius} fill="none"
          className={color}
          strokeWidth="6" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-bold text-slate-100">
          {score.toFixed(1)}
        </span>
        <span className="text-[10px] text-slate-500">/ 10</span>
      </div>
    </div>
  );
}

export function ReportHeader({ ticker, report, financials }: ReportHeaderProps) {
  const score = report.final_weighted_score ?? 0;
  const action = report.recommendation.action;
  const valuation = financials?.valuation as Record<string, number | null> | undefined;
  const companyName = (financials?.company_name as string) || ticker;
  const sector = (financials?.sector as string) || null;
  const industry = (financials?.industry as string) || null;
  const w52High = financials?.week_52_high as number | undefined;
  const w52Low = financials?.week_52_low as number | undefined;
  const currentPrice = valuation?.["Current Price"];

  const metrics = [
    { label: "Price", value: currentPrice != null ? `$${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "N/A" },
    { label: "Market Cap", value: formatNumber(valuation?.["Market Cap"]) },
    { label: "P/E", value: valuation?.["P/E Ratio"] != null ? valuation["P/E Ratio"].toFixed(1) : "N/A" },
    { label: "PEG", value: valuation?.["PEG Ratio"] != null ? valuation["PEG Ratio"].toFixed(2) : "N/A" },
    { label: "Div Yield", value: valuation?.["Dividend Yield"] != null ? `${valuation["Dividend Yield"].toFixed(2)}%` : "N/A" },
  ];

  // 52-week range bar
  let rangePosition: number | null = null;
  if (w52Low != null && w52High != null && currentPrice != null && w52High > w52Low) {
    rangePosition = ((currentPrice - w52Low) / (w52High - w52Low)) * 100;
  }

  return (
    <div className="rounded-2xl border border-navy-700 bg-gradient-to-r from-navy-800/90 to-navy-900/90 p-6 backdrop-blur-lg">
      {/* Top row: Company info + Score ring + Action badge */}
      <div className="flex items-start justify-between gap-6">
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-100">{companyName}</h1>
            <span className="rounded-md bg-sky-500/15 px-2.5 py-0.5 font-mono text-sm font-semibold text-sky-400">
              {ticker}
            </span>
          </div>
          {(sector || industry) && (
            <div className="mt-1.5 flex items-center gap-2 text-sm text-slate-500">
              {sector && <span>{sector}</span>}
              {sector && industry && <span>·</span>}
              {industry && <span>{industry}</span>}
            </div>
          )}
          <p className="mt-3 text-xs text-slate-600">
            Analysis completed {new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
          </p>
        </div>

        <div className="flex items-center gap-5">
          {scoreRing(score)}
          <div className="flex flex-col items-center gap-1.5">
            <span className={clsx("rounded-lg border px-4 py-1.5 text-sm font-bold", actionColor(action))}>
              {action}
            </span>
            <span className="text-[10px] text-slate-600">Recommendation</span>
          </div>
        </div>
      </div>

      {/* Metrics strip */}
      <div className="mt-5 flex items-center gap-6 border-t border-navy-700 pt-4">
        {metrics.map((m) => (
          <div key={m.label} className="flex flex-col">
            <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{m.label}</span>
            <span className="font-mono text-sm font-semibold text-slate-200">{m.value}</span>
          </div>
        ))}

        {/* 52-week range */}
        {rangePosition != null && (
          <div className="ml-auto flex flex-col">
            <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">52-Week Range</span>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] text-slate-500">${w52Low!.toFixed(0)}</span>
              <div className="relative h-1.5 w-24 rounded-full bg-slate-700">
                <div
                  className="absolute top-0 h-1.5 w-2 rounded-full bg-sky-400"
                  style={{ left: `${Math.min(Math.max(rangePosition, 2), 96)}%`, transform: "translateX(-50%)" }}
                />
              </div>
              <span className="font-mono text-[11px] text-slate-500">${w52High!.toFixed(0)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Data source provenance */}
      <div className="mt-3 flex items-center gap-1 text-[10px] text-slate-600">
        <span>Sources:</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">yfinance</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">SEC EDGAR</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">Alpha Vantage</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">FRED</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">USPTO</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">Vertex AI</span>
      </div>
    </div>
  );
}
