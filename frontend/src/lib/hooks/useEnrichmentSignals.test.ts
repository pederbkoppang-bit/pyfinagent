import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useEnrichmentSignals } from "./useEnrichmentSignals";
import type { AllSignals } from "@/lib/types";

describe("useEnrichmentSignals (phase-44.6 hook extraction)", () => {
  it("returns null when data is null", () => {
    const { result } = renderHook(() => useEnrichmentSignals(null));
    expect(result.current).toBeNull();
  });

  it("populates 12 keys with default N/A when fields are absent", () => {
    const minimal: AllSignals = {} as AllSignals;
    const { result } = renderHook(() => useEnrichmentSignals(minimal));
    expect(result.current).not.toBeNull();
    expect(Object.keys(result.current!)).toEqual([
      "insider",
      "options",
      "social_sentiment",
      "patent",
      "earnings_tone",
      "fred_macro",
      "alt_data",
      "sector",
      "nlp_sentiment",
      "anomaly",
      "monte_carlo",
      "quant_model",
    ]);
    for (const v of Object.values(result.current!)) {
      expect(v.signal).toBe("N/A");
      expect(v.summary).toBe("");
    }
  });

  it("extracts the signal + summary from typed fields", () => {
    const data = {
      insider: { signal: "BUY", summary: "10x buys" },
      options: { signal: "PUT-HEAVY", summary: "0.6 ratio" },
    } as unknown as AllSignals;
    const { result } = renderHook(() => useEnrichmentSignals(data));
    expect(result.current!.insider).toEqual({ signal: "BUY", summary: "10x buys" });
    expect(result.current!.options).toEqual({ signal: "PUT-HEAVY", summary: "0.6 ratio" });
  });

  it("coerces nlp_sentiment / anomalies / monte_carlo / quant_model via unknown casts", () => {
    const data = {
      nlp_sentiment: { signal: "POS", summary: "0.74" },
      anomalies: { signal: "Z-SCORE-3", summary: "abnormal" },
      monte_carlo: { signal: "P5", summary: "5pct downside" },
      quant_model: { signal: "FACTOR-HIGH", summary: "M2 = 1.4" },
    } as unknown as AllSignals;
    const { result } = renderHook(() => useEnrichmentSignals(data));
    expect(result.current!.nlp_sentiment.signal).toBe("POS");
    expect(result.current!.anomaly.signal).toBe("Z-SCORE-3");
    expect(result.current!.monte_carlo.signal).toBe("P5");
    expect(result.current!.quant_model.signal).toBe("FACTOR-HIGH");
  });

  it("returns defaults when nested field is a non-object value", () => {
    const data = {
      insider: "not an object" as unknown,
    } as unknown as AllSignals;
    const { result } = renderHook(() => useEnrichmentSignals(data));
    expect(result.current!.insider).toEqual({ signal: "N/A", summary: "" });
  });

  it("returns defaults when nested field has wrong type for signal/summary", () => {
    const data = {
      insider: { signal: 42, summary: ["bad"] },
    } as unknown as AllSignals;
    const { result } = renderHook(() => useEnrichmentSignals(data));
    expect(result.current!.insider).toEqual({ signal: "N/A", summary: "" });
  });
});
