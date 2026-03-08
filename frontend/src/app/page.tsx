"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import {
  AlphaScoreCard,
  InvestmentThesisCard,
  RisksCard,
  ScoringMatrixCard,
} from "@/components/GlassBoxCards";
import { StockChart } from "@/components/StockChart";
import { EvaluationTable } from "@/components/EvaluationTable";
import { ValuationRange } from "@/components/ValuationRange";
import { ResearchInvestigator } from "@/components/ResearchInvestigator";
import { PdfDownload } from "@/components/PdfDownload";
import { SignalCards, SignalSummaryBar } from "@/components/SignalCards";
import { DebateView } from "@/components/DebateView";
import type { DebateResult } from "@/components/DebateView";
import { RiskDashboard } from "@/components/RiskDashboard";
import type { RiskDataPayload } from "@/components/RiskDashboard";
import { BiasReport } from "@/components/BiasReport";
import { getAnalysisStatus, startAnalysis } from "@/lib/api";
import type { AnalysisStatusResponse, SynthesisReport, EnrichmentSignals } from "@/lib/types";

export default function DashboardPage() {
  const [ticker, setTicker] = useState("");
  const [activeTicker, setActiveTicker] = useState("");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [status, setStatus] = useState<AnalysisStatusResponse | null>(null);
  const [report, setReport] = useState<SynthesisReport | null>(null);
  const [financials, setFinancials] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleStart = useCallback(async () => {
    if (!ticker.trim()) return;
    setError(null);
    setReport(null);
    setStatus(null);
    setFinancials(null);
    setLoading(true);
    setActiveTicker(ticker.toUpperCase());

    try {
      const res = await startAnalysis(ticker);
      setAnalysisId(res.analysis_id);

      // Start polling every 3 seconds
      pollRef.current = setInterval(async () => {
        try {
          const s = await getAnalysisStatus(res.analysis_id);
          setStatus(s);

          if (s.status === "completed") {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setReport(s.report ?? null);
            // Fetch yfinance financials for the valuation cards
            try {
              const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
              const fRes = await fetch(`${API_BASE}/api/charts/${encodeURIComponent(ticker)}/financials`);
              if (fRes.ok) setFinancials(await fRes.json());
            } catch { /* best-effort */ }
            setLoading(false);
          } else if (s.status === "failed") {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setError(s.error ?? "Analysis failed");
            setLoading(false);
          }
        } catch (e) {
          // Network hiccup — keep polling
        }
      }, 3000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start analysis");
      setLoading(false);
    }
  }, [ticker]);

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        {/* Header + Input */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-100">
            AI Financial Analyst
          </h2>
          <p className="text-sm text-slate-500">
            Enter a ticker to run comprehensive analysis
          </p>
        </div>

        <div className="mb-8 flex gap-3">
          <input
            type="text"
            placeholder="e.g. NVDA"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && handleStart()}
            className="w-48 rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 font-mono text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
          />
          <button
            onClick={handleStart}
            disabled={loading || !ticker.trim()}
            className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Run Analysis"}
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

        {/* Progress (while running) */}
        {status && !report && (
          <div className="mb-8 max-w-md">
            <AnalysisProgress status={status} />
          </div>
        )}

        {/* Glass Box Report (when complete) */}
        {report && (
          <div className="grid grid-cols-12 gap-6">
            {/* Row 1: Alpha Score + Investment Thesis */}
            <div className="col-span-12 md:col-span-4">
              <AlphaScoreCard report={report} />
            </div>
            <div className="col-span-12 md:col-span-8">
              <InvestmentThesisCard
                report={report}
                financials={financials ? {
                  revenue: (financials as Record<string, Record<string, number>>).health?.["Free Cash Flow"],
                  net_income: undefined,
                  market_cap: (financials as Record<string, Record<string, number>>).valuation?.["Market Cap"],
                } : undefined}
              />
            </div>

            {/* Row 2: Evaluation Table (5 pillar cards) */}
            <div className="col-span-12">
              <EvaluationTable scores={report.scoring_matrix} />
            </div>

            {/* Row 2.5: Enrichment Signals (if available) */}
            {(report as unknown as Record<string, unknown>).enrichment_signals ? (
              <div className="col-span-12 space-y-4">
                <SignalSummaryBar
                  signals={
                    (report as unknown as Record<string, unknown>)
                      .enrichment_signals as EnrichmentSignals
                  }
                />
                <SignalCards
                  signals={
                    (report as unknown as Record<string, unknown>)
                      .enrichment_signals as EnrichmentSignals
                  }
                />
              </div>
            ) : null}

            {/* Row 2.7: Agent Debate */}
            {report.debate_result ? (
              <div className="col-span-12">
                <DebateView debate={report.debate_result as DebateResult} />
              </div>
            ) : null}

            {/* Row 2.8: Risk Dashboard */}
            {report.risk_data ? (
              <div className="col-span-12">
                <RiskDashboard data={report.risk_data as RiskDataPayload} />
              </div>
            ) : null}

            {/* Row 2.9: Bias & Conflict Report */}
            {(report.bias_report || report.conflict_report) ? (
              <div className="col-span-12">
                <BiasReport
                  biasReport={report.bias_report}
                  conflictReport={report.conflict_report}
                />
              </div>
            ) : null}

            {/* Row 3: Stock Chart */}
            <div className="col-span-12">
              <StockChart
                ticker={activeTicker}
                currentPrice={(financials as Record<string, Record<string, number>> | null)?.valuation?.["Current Price"] ?? undefined}
              />
            </div>

            {/* Row 4: Valuation Football Field + Risks */}
            <div className="col-span-12 md:col-span-6">
              <ValuationRange
                valuation={(financials as Record<string, Record<string, number | null>> | null)?.valuation}
                health={(financials as Record<string, Record<string, number | null>> | null)?.health}
              />
            </div>
            <div className="col-span-12 md:col-span-6">
              <RisksCard risks={report.key_risks} />
            </div>

            {/* Row 5: Scoring Matrix + PDF download */}
            <div className="col-span-12 md:col-span-8">
              <ScoringMatrixCard report={report} />
            </div>
            <div className="col-span-12 md:col-span-4 flex flex-col gap-4">
              <PdfDownload ticker={activeTicker} report={report} className="w-full" />
            </div>

            {/* Row 6: Research Investigator */}
            <div className="col-span-12" style={{ minHeight: 400 }}>
              <ResearchInvestigator ticker={activeTicker} />
            </div>
          </div>
        )}

        {/* Empty state */}
        {!status && !report && !error && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <span className="text-6xl">🔍</span>
            <p className="mt-4 text-lg text-slate-400">
              Enter a ticker symbol to begin analysis
            </p>
            <p className="mt-1 text-sm text-slate-600">
              The system will orchestrate 20+ specialized AI agents to produce an
              evidence-based investment report
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
