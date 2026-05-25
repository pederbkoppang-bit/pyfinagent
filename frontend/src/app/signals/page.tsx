"use client";

import { useCallback, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { SignalCards, SignalSummaryBar } from "@/components/SignalCards";
import { SectorDashboard } from "@/components/SectorDashboard";
import { MacroDashboard } from "@/components/MacroDashboard";
import { RecentTickerChips } from "@/components/RecentTickerChips";
import { getAllSignals } from "@/lib/api";
import type { AllSignals } from "@/lib/types";
import { useEnrichmentSignals } from "@/lib/hooks";
import { TabSignals } from "@/lib/icons";

export default function SignalsPage() {
  const [ticker, setTicker] = useState("");
  const [data, setData] = useState<AllSignals | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // phase-44.6: track the most-recently-submitted ticker so the chip
  // row records it (deduped + LRU + persisted via localStorage).
  const [lastSubmitted, setLastSubmitted] = useState<string | null>(null);

  const handleFetch = useCallback(async (tickerArg?: string) => {
    const t = (tickerArg ?? ticker).trim().toUpperCase();
    if (!t) return;
    setTicker(t);
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await getAllSignals(t);
      setData(result);
      setLastSubmitted(t);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch signals");
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  // phase-44.6: extracted the 52-LoC inline type coercion into useEnrichmentSignals.
  const enrichmentSignals = useEnrichmentSignals(data);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* phase-16.48: canonical two-zone shell -- header pinned, content scrolls */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-slate-100">
              Market Signals & Intelligence
            </h2>
            <p className="text-sm text-slate-500">
              Real-time enrichment data from 12 independent signal sources
            </p>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
        {/* phase-44.6 -- labeled ticker input + recent-tickers chip row */}
        <div className="mb-2 flex gap-3 items-end">
          <div>
            <label
              htmlFor="signals-ticker-input"
              className="mb-1 block text-xs uppercase tracking-wider text-slate-500"
            >
              Ticker symbol
            </label>
            <input
              id="signals-ticker-input"
              type="text"
              placeholder="e.g. NVDA"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleFetch()}
              aria-label="Ticker symbol"
              className="w-48 rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 font-mono text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => handleFetch()}
            disabled={loading || !ticker.trim()}
            className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Loading..." : "Fetch Signals"}
          </button>
        </div>
        <RecentTickerChips
          onSelect={(t) => {
            setTicker(t);
            void handleFetch(t);
          }}
          recentlySubmitted={lastSubmitted}
        />

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

        {/* Results -- phase-44.6 progressive disclosure shape:
            level 1 = consensus pill (SignalSummaryBar)
            level 2 = 12 cards (SignalCards)
            level 3 = Sector + Macro deep dives behind native <details>
            Per NN/G progressive disclosure: never go beyond 2 levels deep;
            here levels 1+2 are co-primary and level 3 is the only opt-in. */}
        {data && enrichmentSignals && (
          <div className="space-y-6">
            <SignalSummaryBar signals={enrichmentSignals} />
            <SignalCards signals={enrichmentSignals} />
            {((data.sector && data.sector.signal !== "ERROR") ||
              (data.fred_macro && (data.fred_macro as Record<string, string>).signal !== "ERROR")) && (
              <details className="rounded-xl border border-navy-700 bg-navy-800/40">
                <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-slate-200 hover:bg-navy-700/30 rounded-xl">
                  Sector + Macro deep dive
                </summary>
                <div className="space-y-6 p-4">
                  {data.sector && data.sector.signal !== "ERROR" && (
                    <SectorDashboard data={data.sector} />
                  )}
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
              </details>
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
        </div>
      </main>
    </div>
  );
}
