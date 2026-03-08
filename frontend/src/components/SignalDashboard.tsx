"use client";

import { useState } from "react";
import { clsx } from "clsx";
import type { EnrichmentSignals } from "@/lib/types";

interface SignalMeta {
  label: string;
  icon: string;
  description: string;
  source: string;
  sourceTag: string;
}

const SIGNAL_META: Record<keyof EnrichmentSignals, SignalMeta> = {
  insider: {
    label: "Insider Activity",
    icon: "👔",
    description: "SEC Form 4 insider trading patterns — cluster buys, buy/sell ratio",
    source: "SEC EDGAR",
    sourceTag: "sec-edgar",
  },
  options: {
    label: "Options Flow",
    icon: "📊",
    description: "Put/call ratio, unusual activity (vol > 3× open interest), skew analysis",
    source: "yfinance Options Chain",
    sourceTag: "yfinance",
  },
  social_sentiment: {
    label: "Social Sentiment",
    icon: "💬",
    description: "News & social media sentiment velocity, source divergence detection",
    source: "Alpha Vantage News API",
    sourceTag: "alphavantage",
  },
  patent: {
    label: "Innovation",
    icon: "💡",
    description: "USPTO patent filing velocity, YoY growth, technology domain classification",
    source: "Google BigQuery Patents",
    sourceTag: "bigquery",
  },
  earnings_tone: {
    label: "Earnings Tone",
    icon: "🎙️",
    description: "Management confidence analysis from earnings call transcripts",
    source: "API Ninjas / GCS Cache",
    sourceTag: "api-ninjas",
  },
  fred_macro: {
    label: "Macro Climate",
    icon: "🏛️",
    description: "Fed Funds, CPI, GDP, unemployment, yield curve, consumer sentiment",
    source: "Federal Reserve FRED",
    sourceTag: "fred",
  },
  alt_data: {
    label: "Alt Data",
    icon: "📈",
    description: "Google search interest momentum, lead/lag revenue relationship",
    source: "Google Trends",
    sourceTag: "google-trends",
  },
  sector: {
    label: "Sector Strength",
    icon: "🏭",
    description: "Sector rotation positioning, relative strength vs sector ETF and SPY",
    source: "yfinance + 11 SPDR ETFs",
    sourceTag: "yfinance",
  },
  nlp_sentiment: {
    label: "NLP Sentiment",
    icon: "🧠",
    description: "Transformer-based contextual sentiment via Vertex AI embeddings",
    source: "Vertex AI text-embedding-005",
    sourceTag: "vertex-ai",
  },
  anomaly: {
    label: "Anomaly Scan",
    icon: "⚠️",
    description: "Multi-dimensional Z-score anomaly detection across 13+ metrics",
    source: "Multi-source (yfinance + enrichment)",
    sourceTag: "computed",
  },
  monte_carlo: {
    label: "Risk Scenario",
    icon: "🎲",
    description: "1,000 GBM path simulations — VaR (95%/99%), expected shortfall",
    source: "Monte Carlo Engine",
    sourceTag: "computed",
  },
};

const SOURCE_COLORS: Record<string, string> = {
  "sec-edgar": "bg-blue-500/15 text-blue-400",
  yfinance: "bg-purple-500/15 text-purple-400",
  alphavantage: "bg-orange-500/15 text-orange-400",
  bigquery: "bg-cyan-500/15 text-cyan-400",
  "api-ninjas": "bg-pink-500/15 text-pink-400",
  fred: "bg-green-500/15 text-green-400",
  "google-trends": "bg-yellow-500/15 text-yellow-400",
  "vertex-ai": "bg-indigo-500/15 text-indigo-400",
  computed: "bg-slate-500/15 text-slate-400",
};

function signalColor(signal: string): string {
  const s = signal.toUpperCase();
  if (
    s.includes("BULLISH") || s.includes("BREAKOUT") || s.includes("RISING") ||
    s.includes("TAILWIND") || s.includes("OUTPERFORMING") || s.includes("EASING") ||
    s.includes("FAVORABLE") || s.includes("CONFIDENT") || s.includes("LOW_RISK") ||
    s.includes("ANOMALY_OPPORTUNITY")
  )
    return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  if (
    s.includes("BEARISH") || s.includes("DECLINING") || s.includes("LAGGING") ||
    s.includes("DEFENSIVE") || s.includes("UNFAVORABLE") || s.includes("EVASIVE") ||
    s.includes("HIGH_RISK") || s.includes("EXTREME_RISK") || s.includes("ANOMALY_RISK")
  )
    return "text-rose-400 bg-rose-500/10 border-rose-500/30";
  if (s.includes("ERROR"))
    return "text-slate-500 bg-slate-500/10 border-slate-500/30";
  return "text-amber-400 bg-amber-500/10 border-amber-500/30";
}

function signalSide(signal: string): "bullish" | "bearish" | "neutral" {
  const s = signal.toUpperCase();
  if (
    s.includes("BULLISH") || s.includes("BREAKOUT") || s.includes("RISING") ||
    s.includes("TAILWIND") || s.includes("OUTPERFORMING") || s.includes("CONFIDENT") ||
    s.includes("ANOMALY_OPPORTUNITY") || s.includes("FAVORABLE") || s.includes("LOW_RISK")
  )
    return "bullish";
  if (
    s.includes("BEARISH") || s.includes("DECLINING") || s.includes("LAGGING") ||
    s.includes("DEFENSIVE") || s.includes("EVASIVE") || s.includes("ANOMALY_RISK") ||
    s.includes("HIGH_RISK") || s.includes("EXTREME_RISK") || s.includes("UNFAVORABLE")
  )
    return "bearish";
  return "neutral";
}

export function SignalDashboard({ signals }: { signals: EnrichmentSignals }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const keys = Object.keys(SIGNAL_META) as (keyof EnrichmentSignals)[];

  // Counts for consensus bar
  let bullish = 0, bearish = 0, neutral = 0;
  for (const key of keys) {
    const data = signals[key];
    if (!data) continue;
    const side = signalSide(data.signal);
    if (side === "bullish") bullish++;
    else if (side === "bearish") bearish++;
    else neutral++;
  }
  const total = bullish + bearish + neutral;

  return (
    <div className="space-y-4">
      {/* Consensus divergence bar */}
      <div className="rounded-2xl border border-navy-700 bg-navy-800/70 p-5 backdrop-blur-lg">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-200">📡 Signal Consensus</h3>
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />
              <span className="text-slate-400">{bullish} Bullish</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-amber-500" />
              <span className="text-slate-400">{neutral} Neutral</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-rose-500" />
              <span className="text-slate-400">{bearish} Bearish</span>
            </span>
          </div>
        </div>
        {/* Horizontal divergence bar — bullish from left, bearish from right */}
        <div className="flex h-4 overflow-hidden rounded-full bg-navy-900">
          {total > 0 && (
            <>
              <div className="bg-emerald-500 transition-all" style={{ width: `${(bullish / total) * 100}%` }} />
              <div className="bg-amber-500 transition-all" style={{ width: `${(neutral / total) * 100}%` }} />
              <div className="bg-rose-500 transition-all" style={{ width: `${(bearish / total) * 100}%` }} />
            </>
          )}
        </div>
      </div>

      {/* Signal cards grid */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {keys.map((key) => {
          const meta = SIGNAL_META[key];
          const data = signals[key];
          if (!data) return null;
          const isExpanded = expanded === key;

          return (
            <div
              key={key}
              className={clsx(
                "cursor-pointer rounded-xl border border-navy-700 bg-navy-800/60 p-4 transition-all hover:border-sky-500/30",
                isExpanded && "col-span-1 sm:col-span-2 lg:col-span-2 border-sky-500/40"
              )}
              onClick={() => setExpanded(isExpanded ? null : key)}
            >
              {/* Header row */}
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{meta.icon}</span>
                  <span className="text-sm font-medium text-slate-300">{meta.label}</span>
                </div>
                <span
                  className={clsx(
                    "rounded-full border px-2.5 py-0.5 text-xs font-semibold",
                    signalColor(data.signal)
                  )}
                >
                  {data.signal}
                </span>
              </div>

              {/* Source tag */}
              <div className="mb-2">
                <span className={clsx("rounded px-1.5 py-0.5 text-[10px] font-medium", SOURCE_COLORS[meta.sourceTag])}>
                  {meta.source}
                </span>
              </div>

              {/* Summary */}
              <p className={clsx("text-xs leading-relaxed text-slate-500", !isExpanded && "line-clamp-2")}>
                {data.summary || meta.description}
              </p>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="mt-3 space-y-2 border-t border-navy-700 pt-3">
                  <p className="text-[10px] text-slate-600">{meta.description}</p>
                  <div className="flex items-center gap-2 text-[10px] text-slate-600">
                    <span>Data source: <span className="text-slate-400">{meta.source}</span></span>
                  </div>
                  <p className="text-[10px] italic text-slate-600">
                    Click to collapse · Full analysis available in the Audit tab
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
