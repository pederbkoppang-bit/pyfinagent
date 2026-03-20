"use client";

import { useCallback, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { SignalCards, SignalSummaryBar } from "@/components/SignalCards";
import { SectorDashboard } from "@/components/SectorDashboard";
import { MacroDashboard } from "@/components/MacroDashboard";
import { getAllSignals } from "@/lib/api";
import type { AllSignals, EnrichmentSignals } from "@/lib/types";
import { TabSignals } from "@/lib/icons";

export default function SignalsPage() {
  const [ticker, setTicker] = useState("");
  const [data, setData] = useState<AllSignals | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFetch = useCallback(async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await getAllSignals(ticker);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch signals");
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  // Build enrichment signals summary from the full data
  const enrichmentSignals: EnrichmentSignals | null = data
    ? {
        insider: {
          signal: data.insider?.signal || "N/A",
          summary: data.insider?.summary || "",
        },
        options: {
          signal: data.options?.signal || "N/A",
          summary: data.options?.summary || "",
        },
        social_sentiment: {
          signal: (data.social_sentiment as Record<string, string>)?.signal || "N/A",
          summary: (data.social_sentiment as Record<string, string>)?.summary || "",
        },
        patent: {
          signal: (data.patent as Record<string, string>)?.signal || "N/A",
          summary: (data.patent as Record<string, string>)?.summary || "",
        },
        earnings_tone: {
          signal: (data.earnings_tone as Record<string, string>)?.signal || "N/A",
          summary: (data.earnings_tone as Record<string, string>)?.summary || "",
        },
        fred_macro: {
          signal: (data.fred_macro as Record<string, string>)?.signal || "N/A",
          summary: (data.fred_macro as Record<string, string>)?.summary || "",
        },
        alt_data: {
          signal: (data.alt_data as Record<string, string>)?.signal || "N/A",
          summary: (data.alt_data as Record<string, string>)?.summary || "",
        },
        sector: {
          signal: data.sector?.signal || "N/A",
          summary: data.sector?.summary || "",
        },
        nlp_sentiment: {
          signal: (data as unknown as Record<string, Record<string, string>>).nlp_sentiment?.signal || "N/A",
          summary: (data as unknown as Record<string, Record<string, string>>).nlp_sentiment?.summary || "",
        },
        anomaly: {
          signal: (data as unknown as Record<string, Record<string, string>>).anomalies?.signal || "N/A",
          summary: (data as unknown as Record<string, Record<string, string>>).anomalies?.summary || "",
        },
        monte_carlo: {
          signal: (data as unknown as Record<string, Record<string, string>>).monte_carlo?.signal || "N/A",
          summary: (data as unknown as Record<string, Record<string, string>>).monte_carlo?.summary || "",
        },
      }
    : null;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-100">
            Market Signals & Intelligence
          </h2>
          <p className="text-sm text-slate-500">
            Real-time enrichment data from 11 independent signal sources
          </p>
        </div>

        {/* Input */}
        <div className="mb-8 flex gap-3">
          <input
            type="text"
            placeholder="e.g. NVDA"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && handleFetch()}
            className="w-48 rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 font-mono text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
          />
          <button
            onClick={handleFetch}
            disabled={loading || !ticker.trim()}
            className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Loading..." : "Fetch Signals"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-lg border border-rose-900 bg-rose-950/50 p-4">
            <p className="text-sm font-medium text-rose-200">{error}</p>
            {error.includes("Cannot reach backend") && (
              <p className="mt-2 text-xs text-rose-300/60">
                Run: <code className="rounded bg-rose-900/40 px-1.5 py-0.5 font-mono">uvicorn backend.main:app --port 8000</code>
              </p>
            )}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-3 py-12 text-slate-400">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
            Fetching signals for {ticker}...
          </div>
        )}

        {/* Results */}
        {data && enrichmentSignals && (
          <div className="space-y-6">
            {/* Consensus Bar */}
            <SignalSummaryBar signals={enrichmentSignals} />

            {/* Signal Cards Grid */}
            <SignalCards signals={enrichmentSignals} />

            {/* Sector Dashboard */}
            {data.sector && data.sector.signal !== "ERROR" && (
              <SectorDashboard data={data.sector} />
            )}

            {/* Macro Dashboard */}
            {data.fred_macro && (data.fred_macro as Record<string, string>).signal !== "ERROR" && (
              <MacroDashboard
                data={
                  data.fred_macro as {
                    signal: string;
                    summary: string;
                    indicators: Record<
                      string,
                      { current: number; previous: number; change: number; series_id: string }
                    >;
                    warnings: string[];
                  }
                }
              />
            )}
          </div>
        )}

        {/* Empty state */}
        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <TabSignals size={48} weight="duotone" className="text-slate-600" />
            <p className="mt-4 text-lg text-slate-400">
              Enter a ticker to view market signals
            </p>
            <p className="mt-1 text-sm text-slate-600">
              Insider trades, options flow, patents, sentiment, macro indicators, and more
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
