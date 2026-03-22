"use client";

import { clsx } from "clsx";
import type { EnrichmentSignals } from "@/lib/types";
import type { Icon } from "@phosphor-icons/react";
import {
  SignalInsider, SignalOptions, SignalSocial, SignalPatent,
  SignalEarnings, SignalMacro, SignalAltData, SignalSector,
  SignalNlp, SignalAnomaly, SignalMonteCarlo, SignalQuantModel,
} from "@/lib/icons";

const SIGNAL_META: Record<
  keyof EnrichmentSignals,
  { label: string; icon: Icon; description: string }
> = {
  insider: {
    label: "Insider Activity",
    icon: SignalInsider,
    description: "SEC Form 4 insider trading patterns",
  },
  options: {
    label: "Options Flow",
    icon: SignalOptions,
    description: "Put/Call ratio & unusual activity",
  },
  social_sentiment: {
    label: "Social Sentiment",
    icon: SignalSocial,
    description: "News & social media sentiment velocity",
  },
  patent: {
    label: "Innovation",
    icon: SignalPatent,
    description: "USPTO patent filing trends",
  },
  earnings_tone: {
    label: "Earnings Tone",
    icon: SignalEarnings,
    description: "Management confidence from transcripts",
  },
  fred_macro: {
    label: "Macro Climate",
    icon: SignalMacro,
    description: "FRED economic indicators",
  },
  alt_data: {
    label: "Alt Data",
    icon: SignalAltData,
    description: "Google Trends & search interest",
  },
  sector: {
    label: "Sector Strength",
    icon: SignalSector,
    description: "Relative sector & peer performance",
  },
  nlp_sentiment: {
    label: "NLP Sentiment",
    icon: SignalNlp,
    description: "Transformer-based contextual sentiment",
  },
  anomaly: {
    label: "Anomaly Scan",
    icon: SignalAnomaly,
    description: "Statistical anomaly detection (Z-score)",
  },
  monte_carlo: {
    label: "Risk Scenario",
    icon: SignalMonteCarlo,
    description: "Monte Carlo VaR simulation",
  },
  quant_model: {
    label: "Quant Model",
    icon: SignalQuantModel,
    description: "MDA-weighted ML factor signal",
  },
};

function signalColor(signal: string): string {
  const s = signal.toUpperCase();
  if (
    s.includes("BULLISH") ||
    s.includes("BREAKOUT") ||
    s.includes("RISING") ||
    s.includes("TAILWIND") ||
    s.includes("OUTPERFORMING") ||
    s.includes("EASING") ||
    s.includes("FAVORABLE") ||
    s.includes("CONFIDENT") ||
    s.includes("LOW_RISK") ||
    s.includes("ANOMALY_OPPORTUNITY")
  )
    return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  if (
    s.includes("BEARISH") ||
    s.includes("DECLINING") ||
    s.includes("LAGGING") ||
    s.includes("DEFENSIVE") ||
    s.includes("UNFAVORABLE") ||
    s.includes("EVASIVE") ||
    s.includes("HIGH_RISK") ||
    s.includes("EXTREME_RISK") ||
    s.includes("ANOMALY_RISK")
  )
    return "text-rose-400 bg-rose-500/10 border-rose-500/30";
  if (s.includes("ERROR"))
    return "text-slate-500 bg-slate-500/10 border-slate-500/30";
  return "text-amber-400 bg-amber-500/10 border-amber-500/30";
}

function SignalBadge({ signal }: { signal: string }) {
  return (
    <span
      className={clsx(
        "inline-block rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        signalColor(signal)
      )}
    >
      {signal}
    </span>
  );
}

export function SignalCards({
  signals,
}: {
  signals: EnrichmentSignals;
}) {
  const keys = Object.keys(SIGNAL_META) as (keyof EnrichmentSignals)[];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {keys.map((key) => {
        const meta = SIGNAL_META[key];
        const data = signals[key];
        if (!data) return null;

        return (
          <div
            key={key}
            className="rounded-xl border border-navy-700 bg-navy-800/60 p-4"
          >
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <meta.icon size={20} weight="duotone" className="text-slate-400" />
                <span className="text-sm font-medium text-slate-300">
                  {meta.label}
                </span>
              </div>
              <SignalBadge signal={data.signal} />
            </div>
            <p className="text-xs leading-relaxed text-slate-500">
              {data.summary || meta.description}
            </p>
          </div>
        );
      })}
    </div>
  );
}

export function SignalSummaryBar({
  signals,
}: {
  signals: EnrichmentSignals;
}) {
  const keys = Object.keys(SIGNAL_META) as (keyof EnrichmentSignals)[];
  let bullish = 0;
  let bearish = 0;
  let neutral = 0;

  for (const key of keys) {
    const s = signals[key]?.signal?.toUpperCase() || "";
    if (
      s.includes("BULLISH") ||
      s.includes("BREAKOUT") ||
      s.includes("RISING") ||
      s.includes("TAILWIND") ||
      s.includes("OUTPERFORMING")
    )
      bullish++;
    else if (
      s.includes("BEARISH") ||
      s.includes("DECLINING") ||
      s.includes("LAGGING") ||
      s.includes("DEFENSIVE")
    )
      bearish++;
    else neutral++;
  }

  const total = bullish + bearish + neutral;
  const bullPct = total ? (bullish / total) * 100 : 0;
  const bearPct = total ? (bearish / total) * 100 : 0;
  const neutPct = total ? (neutral / total) * 100 : 0;

  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-4">
      <div className="mb-3 flex items-center justify-between text-sm">
        <span className="font-medium text-slate-300">Signal Consensus</span>
        <span className="text-xs text-slate-500">
          {bullish} bullish · {neutral} neutral · {bearish} bearish
        </span>
      </div>
      <div className="flex h-3 overflow-hidden rounded-full bg-navy-900">
        {bullPct > 0 && (
          <div
            className="bg-emerald-500 transition-all"
            style={{ width: `${bullPct}%` }}
          />
        )}
        {neutPct > 0 && (
          <div
            className="bg-amber-500 transition-all"
            style={{ width: `${neutPct}%` }}
          />
        )}
        {bearPct > 0 && (
          <div
            className="bg-rose-500 transition-all"
            style={{ width: `${bearPct}%` }}
          />
        )}
      </div>
    </div>
  );
}
