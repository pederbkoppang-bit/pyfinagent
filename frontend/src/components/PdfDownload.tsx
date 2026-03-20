"use client";

import { useCallback, useRef, useState } from "react";
import { IconDownload } from "@/lib/icons";
import type {
  SynthesisReport,
  EnrichmentSignals,
  DecisionTrace,
  BiasReportData,
  ConflictReportData,
} from "@/lib/types";
import { InvestmentThesisCard, RisksCard, ScoringMatrixCard } from "./GlassBoxCards";
import { EvaluationTable } from "./EvaluationTable";
import { ValuationRange } from "./ValuationRange";
import { SignalDashboard } from "./SignalDashboard";
import { DebateView } from "./DebateView";
import type { DebateResult } from "./DebateView";
import { RiskDashboard } from "./RiskDashboard";
import type { RiskDataPayload } from "./RiskDashboard";
import { BiasReport } from "./BiasReport";
import { DecisionTraceView } from "./DecisionTraceView";

interface PdfDownloadProps {
  ticker: string;
  report: SynthesisReport;
  financials?: Record<string, unknown> | null;
  className?: string;
}

/* ── Cover page (custom JSX, not a reused component) ── */
function CoverPage({
  ticker,
  report,
  financials,
}: {
  ticker: string;
  report: SynthesisReport;
  financials?: Record<string, unknown> | null;
}) {
  const score = report.final_weighted_score ?? 0;
  const action = report.recommendation.action;
  const companyName = (financials?.company_name as string) || ticker;
  const sector = (financials?.sector as string) || "";
  const industry = (financials?.industry as string) || "";
  const pct = (score / 10) * 100;

  const actionColor = (() => {
    const a = action.toUpperCase();
    if (a.includes("STRONG") && a.includes("BUY")) return "#34d399";
    if (a.includes("BUY")) return "#6ee7b7";
    if (a.includes("STRONG") && a.includes("SELL")) return "#fb7185";
    if (a.includes("SELL")) return "#fda4af";
    return "#fbbf24";
  })();

  return (
    <div
      style={{
        padding: "48px 40px",
        minHeight: 600,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}
    >
      <div style={{ color: "#38bdf8", fontSize: 32, fontWeight: 700, marginBottom: 4 }}>
        PyFinAgent
      </div>
      <div style={{ color: "#94a3b8", fontSize: 16, marginBottom: 40 }}>
        AI Financial Analysis Report
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 24, marginBottom: 32 }}>
        <div>
          <div style={{ fontSize: 44, fontWeight: 800, color: "#e2e8f0" }}>{ticker}</div>
          <div style={{ fontSize: 18, color: "#94a3b8", marginTop: 4 }}>{companyName}</div>
          {(sector || industry) && (
            <div style={{ fontSize: 13, color: "#64748b", marginTop: 4 }}>
              {[sector, industry].filter(Boolean).join(" · ")}
            </div>
          )}
        </div>
        <div style={{ marginLeft: "auto", textAlign: "center" }}>
          <svg width="120" height="120" style={{ transform: "rotate(-90deg)" }}>
            <circle cx="60" cy="60" r="48" fill="none" stroke="#1e293b" strokeWidth="8" />
            <circle
              cx="60"
              cy="60"
              r="48"
              fill="none"
              stroke={score >= 7 ? "#34d399" : score >= 5 ? "#38bdf8" : score >= 3 ? "#fbbf24" : "#fb7185"}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={2 * Math.PI * 48}
              strokeDashoffset={2 * Math.PI * 48 - (pct / 100) * 2 * Math.PI * 48}
            />
          </svg>
          <div
            style={{
              marginTop: -80,
              position: "relative",
              zIndex: 1,
              fontSize: 28,
              fontWeight: 800,
              color: "#e2e8f0",
            }}
          >
            {score.toFixed(1)}
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 2, position: "relative", zIndex: 1 }}>
            / 10
          </div>
          <div
            style={{
              marginTop: 16,
              display: "inline-block",
              padding: "6px 18px",
              borderRadius: 8,
              fontSize: 15,
              fontWeight: 700,
              color: actionColor,
              border: `1px solid ${actionColor}44`,
              background: `${actionColor}15`,
            }}
          >
            {action}
          </div>
        </div>
      </div>
      <div
        style={{
          borderTop: "1px solid #334155",
          paddingTop: 20,
          display: "flex",
          gap: 32,
          flexWrap: "wrap",
        }}
      >
        {(() => {
          const val = financials?.valuation as Record<string, number | null> | undefined;
          const items = [
            { label: "Market Cap", value: val?.["Market Cap"] },
            { label: "P/E", value: val?.["P/E Ratio"] },
            { label: "PEG", value: val?.["PEG Ratio"] },
            { label: "Price", value: val?.["Current Price"] },
            { label: "Div Yield", value: val?.["Dividend Yield"] },
          ];
          return items.map((m) => (
            <div key={m.label}>
              <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>
                {m.label}
              </div>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#e2e8f0", fontFamily: "monospace" }}>
                {m.value != null
                  ? m.label === "Market Cap"
                    ? `$${(m.value / 1e9).toFixed(1)}B`
                    : typeof m.value === "number" && m.label === "Price"
                    ? `$${m.value.toFixed(2)}`
                    : typeof m.value === "number" && m.label === "Div Yield"
                    ? `${m.value.toFixed(2)}%`
                    : m.value.toFixed(2)
                  : "N/A"}
              </div>
            </div>
          ));
        })()}
      </div>
      <div style={{ marginTop: 32, fontSize: 11, color: "#475569" }}>
        Generated{" "}
        {new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
      </div>
      <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>
        Sources: yfinance · SEC EDGAR · Alpha Vantage · FRED · USPTO · Vertex AI
      </div>
    </div>
  );
}

/* ── Section wrapper for consistent page padding ── */
function PdfSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ padding: "32px 24px" }}>
      <h2
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: "#38bdf8",
          marginBottom: 16,
          borderBottom: "1px solid #1e293b",
          paddingBottom: 8,
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

/* ── Main component ── */
export function PdfDownload({ ticker, report, financials, className }: PdfDownloadProps) {
  const [generating, setGenerating] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const generate = useCallback(async () => {
    setGenerating(true);
    try {
      // Dynamic imports — browser-only libs can't be resolved during SSR
      const [{ default: jsPDF }, { default: html2canvas }, { createRoot }] = await Promise.all([
        import("jspdf"),
        import("html2canvas-pro"),
        import("react-dom/client"),
      ]);

      // Create off-screen container
      const container = document.createElement("div");
      container.style.cssText =
        "position:fixed;left:-10000px;top:0;width:794px;background:#0f172a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;";
      document.body.appendChild(container);
      containerRef.current = container;

      // Extract data from report
      const enrichmentSignals = report.enrichment_signals as EnrichmentSignals | undefined;
      const debateResult = report.debate_result as DebateResult | undefined;
      const riskData = report.risk_data as RiskDataPayload | undefined;
      const decisionTraces = report.decision_traces as DecisionTrace[] | undefined;
      const biasReport = report.bias_report as BiasReportData | undefined;
      const conflictReport = report.conflict_report as ConflictReportData | undefined;

      // Build section elements
      const sections: { id: string; element: React.ReactNode }[] = [
        {
          id: "cover",
          element: <CoverPage ticker={ticker} report={report} financials={financials} />,
        },
        {
          id: "thesis",
          element: (
            <PdfSection title="1. Executive Summary">
              <InvestmentThesisCard
                report={report}
                financials={
                  financials
                    ? {
                        revenue: (financials as Record<string, unknown>).revenue as number | undefined,
                        net_income: (financials as Record<string, unknown>).net_income as number | undefined,
                        market_cap: (financials as Record<string, Record<string, number>>).valuation?.["Market Cap"],
                      }
                    : undefined
                }
              />
            </PdfSection>
          ),
        },
        {
          id: "scoring",
          element: (
            <PdfSection title="2. Scoring Matrix">
              <EvaluationTable scores={report.scoring_matrix} />
            </PdfSection>
          ),
        },
        {
          id: "valuation",
          element: (
            <PdfSection title="3. Valuation & Risks">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <ValuationRange
                  valuation={(financials as Record<string, Record<string, number | null>> | null)?.valuation}
                  health={(financials as Record<string, Record<string, number | null>> | null)?.health}
                />
                <RisksCard risks={report.key_risks} />
              </div>
              <ScoringMatrixCard report={report} />
            </PdfSection>
          ),
        },
      ];

      if (enrichmentSignals) {
        sections.push({
          id: "signals",
          element: (
            <PdfSection title="4. Enrichment Signals">
              <SignalDashboard signals={enrichmentSignals} />
            </PdfSection>
          ),
        });
      }

      if (debateResult) {
        sections.push({
          id: "debate",
          element: (
            <PdfSection title="5. Agent Debate">
              <DebateView debate={debateResult} />
            </PdfSection>
          ),
        });
      }

      if (riskData) {
        sections.push({
          id: "risk",
          element: (
            <PdfSection title="6. Risk Analysis">
              <RiskDashboard data={riskData} />
            </PdfSection>
          ),
        });
      }

      if ((biasReport || conflictReport) && decisionTraces) {
        sections.push({
          id: "audit",
          element: (
            <PdfSection title="7. Audit & Bias">
              {(biasReport || conflictReport) && (
                <BiasReport biasReport={biasReport} conflictReport={conflictReport} />
              )}
              <div style={{ marginTop: 16 }}>
                <DecisionTraceView traces={decisionTraces ?? []} />
              </div>
            </PdfSection>
          ),
        });
      } else if (decisionTraces && decisionTraces.length > 0) {
        sections.push({
          id: "audit",
          element: (
            <PdfSection title="7. Decision Audit Trail">
              <DecisionTraceView traces={decisionTraces} />
            </PdfSection>
          ),
        });
      }

      // Render each section into the off-screen container and capture
      const A4_W = 595.28; // A4 width in points (jsPDF)
      const A4_H = 841.89; // A4 height in points
      const FOOTER_H = 20;
      const PAGE_CONTENT_H = A4_H - FOOTER_H;
      const doc = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });
      let isFirstPage = true;
      let pageNum = 0;

      for (const section of sections) {
        // Create a wrapper div for this section
        const wrapper = document.createElement("div");
        wrapper.style.cssText = "width:794px;background:#0f172a;color:#e2e8f0;";
        container.innerHTML = "";
        container.appendChild(wrapper);

        // Mount React component into the wrapper
        const root = createRoot(wrapper);
        root.render(section.element as React.ReactElement);

        // Wait for render + fonts/images to settle
        await new Promise((r) => setTimeout(r, 300));

        // Capture with html2canvas
        const canvas = await html2canvas(wrapper, {
          backgroundColor: "#0f172a",
          scale: 2,
          useCORS: true,
          logging: false,
        });

        root.unmount();

        // Convert canvas to image data
        const imgData = canvas.toDataURL("image/png");
        const imgW = A4_W;
        const imgH = (canvas.height / canvas.width) * A4_W;

        // Handle multi-page slicing for tall sections
        if (imgH <= PAGE_CONTENT_H) {
          // Fits in one page
          if (!isFirstPage) doc.addPage();
          isFirstPage = false;
          pageNum++;
          doc.setFillColor(15, 23, 42);
          doc.rect(0, 0, A4_W, A4_H, "F");
          doc.addImage(imgData, "PNG", 0, 0, imgW, imgH);
          // Footer
          addFooter(doc, pageNum, A4_W, A4_H);
        } else {
          // Slice into pages
          const totalPages = Math.ceil(imgH / PAGE_CONTENT_H);
          for (let p = 0; p < totalPages; p++) {
            if (!isFirstPage) doc.addPage();
            isFirstPage = false;
            pageNum++;
            doc.setFillColor(15, 23, 42);
            doc.rect(0, 0, A4_W, A4_H, "F");

            // Create a sliced canvas for this page portion
            const srcY = (p * PAGE_CONTENT_H / imgH) * canvas.height;
            const srcH = (PAGE_CONTENT_H / imgH) * canvas.height;
            const sliceCanvas = document.createElement("canvas");
            sliceCanvas.width = canvas.width;
            sliceCanvas.height = Math.min(srcH, canvas.height - srcY);
            const ctx = sliceCanvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(
                canvas,
                0,
                srcY,
                canvas.width,
                sliceCanvas.height,
                0,
                0,
                sliceCanvas.width,
                sliceCanvas.height,
              );
            }
            const sliceData = sliceCanvas.toDataURL("image/png");
            const sliceH = (sliceCanvas.height / sliceCanvas.width) * A4_W;
            doc.addImage(sliceData, "PNG", 0, 0, imgW, Math.min(sliceH, PAGE_CONTENT_H));
            addFooter(doc, pageNum, A4_W, A4_H);
          }
        }
      }

      // Clean up
      document.body.removeChild(container);
      containerRef.current = null;

      doc.save(`PyFinAgent_${ticker}_Report.pdf`);
    } catch (err) {
      console.error("PDF generation failed:", err);
    } finally {
      // Safety cleanup
      if (containerRef.current && document.body.contains(containerRef.current)) {
        document.body.removeChild(containerRef.current);
        containerRef.current = null;
      }
      setGenerating(false);
    }
  }, [ticker, report, financials]);

  return (
    <button
      onClick={generate}
      disabled={generating}
      className={`rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 transition-colors hover:bg-sky-500/20 disabled:opacity-50 disabled:cursor-wait ${className ?? ""}`}
    >
      {generating ? (
        <span className="flex items-center gap-2">
          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
            <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" />
          </svg>
          Generating PDF...
        </span>
      ) : (
        <><IconDownload size={16} className="inline" /> Download PDF</>
      )}
    </button>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function addFooter(doc: any, pageNum: number, w: number, h: number) {
  doc.setFontSize(7);
  doc.setTextColor(100, 116, 139);
  doc.text("PyFinAgent AI Analysis — This is not financial advice.", 20, h - 8);
  doc.text(`Page ${pageNum}`, w - 40, h - 8);
}

