"use client";

// phase-44.6 -- useEnrichmentSignals hook.
//
// Extracts the 52-LoC inline type-coercion block from
// `app/signals/page.tsx` into a reusable hook. The backend AllSignals
// response uses `Record<string, unknown>` for several fields (per
// `lib/types.ts:244-247` -- nlp_sentiment, anomalies, monte_carlo,
// quant_model), so consumers MUST coerce. Encoding that coercion ONCE
// in this hook removes ~52 LoC of duplication risk from the consumer
// and gives future signal consumers (e.g. /reports compare wizard) a
// single source of truth.
//
// Defensive: never throws on missing fields; defaults to {"signal": "N/A",
// "summary": ""} so the SignalCards renderer always has structured data.

import { useMemo } from "react";
import type { AllSignals, EnrichmentSignals } from "@/lib/types";

interface SignalEntry {
  signal: string;
  summary: string;
}

function pick(source: unknown): SignalEntry {
  if (source && typeof source === "object") {
    const obj = source as Record<string, unknown>;
    const signal = typeof obj.signal === "string" ? obj.signal : "N/A";
    const summary = typeof obj.summary === "string" ? obj.summary : "";
    return { signal, summary };
  }
  return { signal: "N/A", summary: "" };
}

export function useEnrichmentSignals(
  data: AllSignals | null,
): EnrichmentSignals | null {
  return useMemo(() => {
    if (!data) return null;
    return {
      insider: pick(data.insider),
      options: pick(data.options),
      social_sentiment: pick(data.social_sentiment),
      patent: pick(data.patent),
      earnings_tone: pick(data.earnings_tone),
      fred_macro: pick(data.fred_macro),
      alt_data: pick(data.alt_data),
      sector: pick(data.sector),
      nlp_sentiment: pick(
        (data as unknown as Record<string, unknown>).nlp_sentiment,
      ),
      anomaly: pick(
        (data as unknown as Record<string, unknown>).anomalies,
      ),
      monte_carlo: pick(
        (data as unknown as Record<string, unknown>).monte_carlo,
      ),
      quant_model: pick(
        (data as unknown as Record<string, unknown>).quant_model,
      ),
    };
  }, [data]);
}
