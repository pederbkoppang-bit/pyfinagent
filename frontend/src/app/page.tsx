"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import {
  InvestmentThesisCard,
  RisksCard,
  ScoringMatrixCard,
} from "@/components/GlassBoxCards";
import { ReportHeader } from "@/components/ReportHeader";
import { ReportTabs, type TabDef } from "@/components/ReportTabs";
import { SignalDashboard } from "@/components/SignalDashboard";
import { DecisionTraceView } from "@/components/DecisionTraceView";
import { StockChart } from "@/components/StockChart";
import { EvaluationTable } from "@/components/EvaluationTable";
import { ValuationRange } from "@/components/ValuationRange";
import { ResearchInvestigator } from "@/components/ResearchInvestigator";
import { DebateView } from "@/components/DebateView";
import type { DebateResult } from "@/components/DebateView";
import { RiskDashboard, RiskAssessmentPanel } from "@/components/RiskDashboard";
import type { RiskDataPayload } from "@/components/RiskDashboard";
import { BiasReport } from "@/components/BiasReport";
import { CostDashboard } from "@/components/CostDashboard";
import { getAnalysisStatus, startAnalysis } from "@/lib/api";
import type {
  AnalysisStatusResponse,
  SynthesisReport,
  EnrichmentSignals,
  CostSummary,
  DecisionTrace,
} from "@/lib/types";

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
          <div className="mb-8">
            <AnalysisProgress status={status} />
          </div>
        )}

        {/* Glass Box Report (when complete) */}
        {report && (() => {
          const enrichmentSignals = report.enrichment_signals as EnrichmentSignals | undefined;
          const debateResult = report.debate_result as DebateResult | undefined;
          const riskData = report.risk_data as RiskDataPayload | undefined;
          const decisionTraces = report.decision_traces as DecisionTrace[] | undefined;
        const costSummary = report.cost_summary as CostSummary | undefined;

        // Count badges for tabs
        const signalCount = enrichmentSignals ? Object.keys(enrichmentSignals).length : 0;
        const anomalyCount = (riskData?.anomalies?.anomaly_count) ?? 0;
        const biasCount = (report.bias_report as Record<string, unknown> | undefined)?.bias_count as number | undefined;
        const costBadge = costSummary?.total_cost_usd != null ? `$${costSummary.total_cost_usd < 0.01 ? costSummary.total_cost_usd.toFixed(4) : costSummary.total_cost_usd.toFixed(2)}` : null;

        const tabs: TabDef[] = [
          { id: "overview", label: "Overview", icon: "📋" },
          { id: "signals", label: "Signals", icon: "📡", badge: signalCount > 0 ? signalCount : null },
          { id: "debate", label: "Debate", icon: "⚖️", badge: debateResult?.consensus || null },
          { id: "risk", label: "Risk", icon: "🎯", badge: anomalyCount > 0 ? `${anomalyCount} anomalies` : null },
          { id: "audit", label: "Audit", icon: "🔍", badge: biasCount != null && biasCount > 0 ? `${biasCount} flags` : null },
          { id: "cost", label: "Cost", icon: "💰", badge: costBadge },
        ];

          return (
            <div className="space-y-6">
              {/* Report Header — always visible */}
              <ReportHeader ticker={activeTicker} report={report} financials={financials} />

              {/* Tabbed content */}
              <ReportTabs tabs={tabs}>
                {(activeTab) => (
                  <>
                    {/* ── OVERVIEW TAB ── */}
                    {activeTab === "overview" && (
                      <div className="grid grid-cols-12 gap-6">
                        {/* Executive Summary */}
                        <div className="col-span-12">
                          <InvestmentThesisCard
                            report={report}
                            financials={financials ? {
                              revenue: (financials as Record<string, unknown>).revenue as number | undefined,
                              net_income: (financials as Record<string, unknown>).net_income as number | undefined,
                              market_cap: (financials as Record<string, Record<string, number>>).valuation?.["Market Cap"],
                            } : undefined}
                          />
                        </div>

                        {/* Evaluation Table (5 pillars) */}
                        <div className="col-span-12">
                          <EvaluationTable scores={report.scoring_matrix} />
                        </div>

                        {/* Valuation + Risks side-by-side */}
                        <div className="col-span-12 md:col-span-6">
                          <ValuationRange
                            valuation={(financials as Record<string, Record<string, number | null>> | null)?.valuation}
                            health={(financials as Record<string, Record<string, number | null>> | null)?.health}
                          />
                        </div>
                        <div className="col-span-12 md:col-span-6">
                          <RisksCard risks={report.key_risks} />
                        </div>

                        {/* Scoring Matrix */}
                        <div className="col-span-12">
                          <ScoringMatrixCard report={report} />
                        </div>
                      </div>
                    )}

                    {/* ── SIGNALS TAB ── */}
                    {activeTab === "signals" && (
                      <div className="space-y-6">
                        {enrichmentSignals ? (
                          <SignalDashboard signals={enrichmentSignals} />
                        ) : (
                          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
                            Enrichment signals not available for this analysis.
                          </div>
                        )}
                      </div>
                    )}

                    {/* ── DEBATE TAB ── */}
                    {activeTab === "debate" && (
                      <div className="space-y-6">
                        {debateResult ? (
                          <DebateView debate={debateResult} />
                        ) : (
                          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
                            Agent debate data not available for this analysis.
                          </div>
                        )}
                      </div>
                    )}

                    {/* ── RISK TAB ── */}
                    {activeTab === "risk" && (
                      <div className="space-y-6">
                        {riskData ? (
                          <RiskDashboard data={riskData} />
                        ) : (
                          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
                            Risk data not available for this analysis.
                          </div>
                        )}
                        {report?.risk_assessment && (
                          <RiskAssessmentPanel data={report.risk_assessment} />
                        )}
                        <StockChart
                          ticker={activeTicker}
                          currentPrice={(financials as Record<string, Record<string, number>> | null)?.valuation?.["Current Price"] ?? undefined}
                        />
                      </div>
                    )}

                    {/* ── AUDIT TAB ── */}
                    {activeTab === "audit" && (
                      <div className="space-y-6">
                        {(report.bias_report || report.conflict_report) && (
                          <BiasReport
                            biasReport={report.bias_report}
                            conflictReport={report.conflict_report}
                          />
                        )}
                        <DecisionTraceView traces={decisionTraces ?? []} />
                        <div style={{ minHeight: 400 }}>
                          <ResearchInvestigator ticker={activeTicker} />
                        </div>
                      </div>
                    )}

                    {/* ── COST TAB ── */}
                    {activeTab === "cost" && (
                      <CostDashboard costSummary={costSummary} />
                    )}
                  </>
                )}
              </ReportTabs>
            </div>
          );
        })()}

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
