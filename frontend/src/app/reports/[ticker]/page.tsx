"use client";

// phase-86 UI restore -- full-page per-ticker report.
//
// This revives the six-tab report view (Overview / Signals / Debate / Risk /
// Audit / Cost) that lived at /analyze from 2026-03-08 until it was deleted
// 2026-04-18 ("removed unused analyze/compare/portfolio routes") with no
// replacement. The supporting components (ReportHeader, ReportTabs,
// GlassBoxCards, EvaluationTable, ValuationRange, SignalDashboard,
// DebateView, RiskDashboard, BiasReport, DecisionTraceView, StockChart,
// ResearchInvestigator, CostDashboard) were never deleted -- this page is
// the first thing to wire them back up.
//
// Unlike the old /analyze (which kicked off a NEW live analysis and polled
// for it), this route reads an ALREADY-PERSISTED report via getReport() --
// reached by clicking a row in the /reports History table.

import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Sidebar } from "@/components/Sidebar";
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
import {
  TabOverview, TabSignals, TabDebate, TabRisk, TabAudit, TabCost,
  CaretLeft, IconWarning,
} from "@/lib/icons";
import { getReport, getChartFinancials } from "@/lib/api";
import type {
  SynthesisReport,
  EnrichmentSignals,
  CostSummary,
  DecisionTrace,
} from "@/lib/types";

function Fallback({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

export default function ReportDetailPage() {
  const params = useParams<{ ticker: string }>();
  const searchParams = useSearchParams();
  const ticker = decodeURIComponent(params?.ticker || "").toUpperCase();
  const date = searchParams.get("date");

  const [companyName, setCompanyName] = useState<string>(ticker);
  const [analysisDate, setAnalysisDate] = useState<string>("");
  const [report, setReport] = useState<SynthesisReport | null>(null);
  const [rawScore, setRawScore] = useState<number | null>(null);
  const [rawRecommendation, setRawRecommendation] = useState<string | null>(null);
  const [degraded, setDegraded] = useState(false);
  const [financials, setFinancials] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ticker) {
      setLoading(false);
      return;
    }
    let ignore = false;
    setLoading(true);
    setError(null);
    getReport(ticker, date ?? undefined)
      .then((full) => {
        if (ignore) return;
        const fullJson = (full.full_report_json ?? {}) as Record<string, unknown>;
        const synth =
          (fullJson.final_synthesis as SynthesisReport | undefined) ??
          (Object.keys(fullJson).length > 0 ? (fullJson as unknown as SynthesisReport) : null);
        setDegraded(Boolean(fullJson._degraded));
        setReport(synth);
        setRawScore((full.final_score as number | null) ?? null);
        setRawRecommendation((full.recommendation as string | null) ?? null);
        setCompanyName((full.company_name as string) || ticker);
        setAnalysisDate((full.analysis_date as string) || date || "");
      })
      .catch((e: unknown) => {
        if (!ignore) setError(e instanceof Error ? e.message : "Failed to load report");
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, [ticker, date]);

  // Best-effort live yfinance snapshot for the header metrics strip + the
  // Valuation Football Field chart. Independent failure path -- a stale or
  // delisted ticker still shows the persisted report even if this 404s.
  useEffect(() => {
    if (!ticker) return;
    let ignore = false;
    getChartFinancials(ticker)
      .then((d) => {
        if (!ignore) setFinancials(d);
      })
      .catch(() => {
        /* best-effort enrichment only */
      });
    return () => {
      ignore = true;
    };
  }, [ticker]);

  // ReportHeader and the GlassBox cards assume `report.recommendation.action`
  // and `report.final_summary` exist unconditionally (they did, always, on
  // the old page's source -- a synchronously-completed live analysis). A
  // persisted BQ row can be genuinely sparse (measured: some full-path rows
  // carry score 0.0 / no _degraded flag / an empty final_synthesis). Gate
  // the rich UI on the one field every real synthesis object has, and fall
  // back to the flat DB columns (still populated even when the nested JSON
  // is thin) rather than crashing the page.
  const hasFullReport = Boolean(report?.recommendation?.action);

  const enrichmentSignals = report?.enrichment_signals as EnrichmentSignals | undefined;
  const debateResult = report?.debate_result as DebateResult | undefined;
  const riskData = report?.risk_data as unknown as RiskDataPayload | undefined;
  const decisionTraces = report?.decision_traces as DecisionTrace[] | undefined;
  const costSummary = report?.cost_summary as CostSummary | undefined;

  const signalCount = enrichmentSignals ? Object.keys(enrichmentSignals).length : 0;
  const anomalyCount = riskData?.anomalies?.anomaly_count ?? 0;
  const biasCount = report?.bias_report?.bias_count;
  const costBadge =
    costSummary?.total_cost_usd != null
      ? `$${costSummary.total_cost_usd < 0.01 ? costSummary.total_cost_usd.toFixed(4) : costSummary.total_cost_usd.toFixed(2)}`
      : null;

  const tabs: TabDef[] = [
    { id: "overview", label: "Overview", icon: TabOverview },
    { id: "signals", label: "Signals", icon: TabSignals, badge: signalCount > 0 ? signalCount : null },
    { id: "debate", label: "Debate", icon: TabDebate, badge: debateResult?.consensus || null },
    { id: "risk", label: "Risk", icon: TabRisk, badge: anomalyCount > 0 ? `${anomalyCount} anomalies` : null },
    { id: "audit", label: "Audit", icon: TabAudit, badge: biasCount != null && biasCount > 0 ? `${biasCount} flags` : null },
    { id: "cost", label: "Cost", icon: TabCost, badge: costBadge },
  ];

  const valuationFinancials = financials as Record<string, Record<string, number | null>> | null;

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <Link
            href="/reports"
            className="mb-4 inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300"
          >
            <CaretLeft size={12} weight="bold" />
            Back to Reports
          </Link>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {loading && <p className="py-12 text-center text-sm text-slate-500">Loading report...</p>}

          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-rose-400">
              {error}
            </div>
          )}

          {!loading && !error && !report && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <IconWarning size={48} weight="duotone" className="text-slate-600" />
              <p className="mt-4 text-lg text-slate-400">No report content found for {ticker}.</p>
              <p className="mt-1 text-sm text-slate-600">
                The analysis row exists but carries no synthesis payload.
              </p>
            </div>
          )}

          {!loading && !error && report && !hasFullReport && (
            <div className="space-y-4">
              <div className="flex items-center gap-4 rounded-xl border border-navy-700 bg-navy-800/60 p-4">
                <div className="flex flex-col items-center">
                  <span className="font-mono text-2xl font-bold text-slate-100">
                    {rawScore != null ? rawScore.toFixed(2) : "—"}
                  </span>
                  <span className="text-[10px] text-slate-500">/ 10</span>
                </div>
                <div className="h-10 w-px bg-navy-700" />
                <span className="rounded-lg border border-slate-600/40 bg-slate-700/40 px-4 py-1.5 text-sm font-bold text-slate-300">
                  {rawRecommendation ?? "N/A"}
                </span>
              </div>
              <div className="flex items-start gap-2 rounded-lg border border-navy-700 bg-navy-800/40 p-3">
                <IconWarning size={16} className="mt-0.5 flex-shrink-0 text-slate-500" />
                <p className="text-sm text-slate-400">
                  {degraded
                    ? "This analysis was degraded (both the full and lite pipelines failed to produce a scored report)."
                    : "No justification, pillar scores, summary, or risks were captured for this run"}
                  {" -- only the score and recommendation above were persisted."}
                </p>
              </div>
            </div>
          )}

          {!loading && !error && report && hasFullReport && (
            <div className="space-y-6">
              <ReportHeader ticker={ticker} report={report} financials={financials} />
              {analysisDate && (
                <p className="-mt-4 text-xs text-slate-600">
                  {companyName !== ticker ? `${companyName} -- ` : ""}
                  {new Date(analysisDate).toLocaleDateString(undefined, {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
              )}

              {/* phase-86 UI redesign: the price chart is now persistent and
                  always visible directly below the header, matching how
                  Yahoo Finance / Robinhood / TradingView / Bloomberg treat
                  the chart as hero content -- not buried in a sub-tab (it
                  previously only rendered inside the Risk tab). Research:
                  setproduct.com's dashboard-anatomy guide places the KPI row
                  above the hero chart, which is exactly ReportHeader's stat
                  strip followed by this. */}
              <StockChart
                ticker={ticker}
                currentPrice={valuationFinancials?.valuation?.["Current Price"] ?? undefined}
              />

              <ReportTabs tabs={tabs}>
                {(activeTab) => (
                  <>
                    {activeTab === "overview" && (
                      <div className="grid grid-cols-12 gap-6">
                        <div className="col-span-12">
                          <InvestmentThesisCard
                            report={report}
                            financials={
                              financials
                                ? {
                                    revenue: financials.revenue as number | undefined,
                                    net_income: financials.net_income as number | undefined,
                                    market_cap: valuationFinancials?.valuation?.["Market Cap"] ?? undefined,
                                  }
                                : undefined
                            }
                          />
                        </div>

                        <div className="col-span-12">
                          {report.scoring_matrix ? (
                            <EvaluationTable scores={report.scoring_matrix} />
                          ) : (
                            <Fallback>Pillar scoring breakdown not available for this analysis.</Fallback>
                          )}
                        </div>

                        <div className="col-span-12 md:col-span-6">
                          <ValuationRange
                            valuation={valuationFinancials?.valuation ?? undefined}
                            health={valuationFinancials?.health ?? undefined}
                          />
                        </div>
                        <div className="col-span-12 md:col-span-6">
                          {report.key_risks?.length > 0 ? (
                            <RisksCard risks={report.key_risks} />
                          ) : (
                            <Fallback>No key risks were recorded for this analysis.</Fallback>
                          )}
                        </div>

                        <div className="col-span-12">
                          {report.scoring_matrix ? (
                            <ScoringMatrixCard report={report} />
                          ) : (
                            <Fallback>Scoring matrix not available for this analysis.</Fallback>
                          )}
                        </div>
                      </div>
                    )}

                    {activeTab === "signals" && (
                      <div className="space-y-6">
                        {enrichmentSignals ? (
                          <SignalDashboard signals={enrichmentSignals} />
                        ) : (
                          <Fallback>Enrichment signals not available for this analysis.</Fallback>
                        )}
                      </div>
                    )}

                    {activeTab === "debate" && (
                      <div className="space-y-6">
                        {debateResult ? (
                          <DebateView debate={debateResult} />
                        ) : (
                          <Fallback>Agent debate data not available for this analysis.</Fallback>
                        )}
                      </div>
                    )}

                    {activeTab === "risk" && (
                      <div className="space-y-6">
                        {riskData ? (
                          <RiskDashboard data={riskData} />
                        ) : (
                          <Fallback>Risk data not available for this analysis.</Fallback>
                        )}
                        {report.risk_assessment && <RiskAssessmentPanel data={report.risk_assessment} />}
                      </div>
                    )}

                    {activeTab === "audit" && (
                      <div className="space-y-6">
                        {(report.bias_report || report.conflict_report) && (
                          <BiasReport biasReport={report.bias_report} conflictReport={report.conflict_report} />
                        )}
                        <DecisionTraceView traces={decisionTraces ?? []} />
                        <div style={{ minHeight: 400 }}>
                          <ResearchInvestigator ticker={ticker} />
                        </div>
                      </div>
                    )}

                    {activeTab === "cost" && <CostDashboard costSummary={costSummary} />}
                  </>
                )}
              </ReportTabs>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
