"use client";

import { useCallback } from "react";
import jsPDF from "jspdf";
import type { SynthesisReport, EnrichmentSignals, DecisionTrace } from "@/lib/types";

interface PdfDownloadProps {
  ticker: string;
  report: SynthesisReport;
  financials?: Record<string, unknown> | null;
  className?: string;
}

export function PdfDownload({ ticker, report, financials, className }: PdfDownloadProps) {
  const generate = useCallback(() => {
    const doc = new jsPDF();
    const margin = 20;
    const pageWidth = 170;
    let y = margin;

    const addText = (text: string, size: number, style: "normal" | "bold" = "normal", color?: [number, number, number]) => {
      doc.setFontSize(size);
      doc.setFont("helvetica", style);
      if (color) doc.setTextColor(...color);
      else doc.setTextColor(30, 30, 30);
      const lines = doc.splitTextToSize(text, pageWidth);
      for (const line of lines) {
        if (y > 270) {
          doc.addPage();
          y = margin;
        }
        doc.text(line, margin, y);
        y += size * 0.5;
      }
    };

    const addLine = () => {
      if (y > 265) { doc.addPage(); y = margin; }
      doc.setDrawColor(200, 200, 200);
      doc.line(margin, y, margin + pageWidth, y);
      y += 4;
    };

    const addSpacer = (h = 6) => { y += h; };

    // ── Page 1: Cover ──────────────────────────────────────────
    y = 60;
    addText("PyFinAgent", 28, "bold", [30, 120, 200]);
    addText("AI Financial Analysis Report", 14, "normal", [100, 100, 100]);
    addSpacer(12);
    addText(ticker, 36, "bold", [20, 20, 20]);
    const companyName = (financials?.company_name as string) || ticker;
    addText(companyName, 16, "normal", [80, 80, 80]);
    addSpacer(20);
    addText(`Score: ${report.final_weighted_score?.toFixed(2) ?? "N/A"} / 10`, 18, "bold");
    addText(`Recommendation: ${report.recommendation.action}`, 16, "bold", [30, 120, 200]);
    addSpacer(10);
    addText(`Generated: ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}`, 10, "normal", [120, 120, 120]);
    addText("Sources: yfinance, SEC EDGAR, Alpha Vantage, FRED, USPTO, Vertex AI", 8, "normal", [150, 150, 150]);

    // ── Page 2: Executive Summary ──────────────────────────────
    doc.addPage();
    y = margin;
    addText("1. Executive Summary", 16, "bold", [30, 120, 200]);
    addSpacer(4);
    addText(report.recommendation.justification, 10, "bold");
    addSpacer(4);
    addText(report.final_summary, 10);
    addSpacer(8);

    // ── Scoring Matrix ─────────────────────────────────────────
    addText("2. Scoring Matrix", 16, "bold", [30, 120, 200]);
    addSpacer(4);
    const pillars = [
      ["Corporate Profile (35%)", report.scoring_matrix.pillar_1_corporate],
      ["Industry & Macro (20%)", report.scoring_matrix.pillar_2_industry],
      ["Valuation (20%)", report.scoring_matrix.pillar_3_valuation],
      ["Market Sentiment (15%)", report.scoring_matrix.pillar_4_sentiment],
      ["Governance (10%)", report.scoring_matrix.pillar_5_governance],
    ] as const;
    for (const [label, val] of pillars) {
      addText(`  ${label}: ${val.toFixed(2)} / 10`, 10);
    }
    addSpacer(2);
    addLine();

    // ── Key Risks ──────────────────────────────────────────────
    addText("3. Key Risks", 16, "bold", [30, 120, 200]);
    addSpacer(4);
    for (const risk of report.key_risks) {
      addText(`  • ${risk}`, 10);
    }
    addSpacer(4);
    addLine();

    // ── Financial Snapshot ──────────────────────────────────────
    if (financials) {
      addText("4. Financial Snapshot", 16, "bold", [30, 120, 200]);
      addSpacer(4);
      const val = financials.valuation as Record<string, number | null> | undefined;
      const eff = financials.efficiency as Record<string, number | null> | undefined;
      if (val) {
        for (const [k, v] of Object.entries(val)) {
          if (v != null) addText(`  ${k}: ${typeof v === "number" && Math.abs(v) >= 1e6 ? `$${(v / 1e9).toFixed(2)}B` : v}`, 9);
        }
      }
      if (eff) {
        addSpacer(2);
        for (const [k, v] of Object.entries(eff)) {
          if (v != null) addText(`  ${k}: ${typeof v === "number" ? v.toFixed(1) + "%" : v}`, 9);
        }
      }
      addSpacer(4);
      addLine();
    }

    // ── Enrichment Signals ─────────────────────────────────────
    const signals = report.enrichment_signals as EnrichmentSignals | undefined;
    if (signals) {
      addText("5. Enrichment Signals", 16, "bold", [30, 120, 200]);
      addSpacer(4);
      for (const [key, data] of Object.entries(signals)) {
        if (data && typeof data === "object" && "signal" in data) {
          const s = data as { signal: string; summary: string };
          addText(`  ${key}: ${s.signal}`, 10, "bold");
          if (s.summary) addText(`    ${s.summary.slice(0, 200)}`, 8);
          addSpacer(2);
        }
      }
      addLine();
    }

    // ── Debate Summary ─────────────────────────────────────────
    const debate = report.debate_result as Record<string, unknown> | undefined;
    if (debate) {
      addText("6. Agent Debate Summary", 16, "bold", [30, 120, 200]);
      addSpacer(4);
      addText(`Consensus: ${debate.consensus ?? "N/A"}  (Confidence: ${debate.consensus_confidence ?? "N/A"})`, 10, "bold");
      addSpacer(2);
      const bull = debate.bull_case as Record<string, unknown> | undefined;
      const bear = debate.bear_case as Record<string, unknown> | undefined;
      if (bull?.thesis) { addText("Bull Case:", 10, "bold"); addText(String(bull.thesis).slice(0, 300), 9); addSpacer(2); }
      if (bear?.thesis) { addText("Bear Case:", 10, "bold"); addText(String(bear.thesis).slice(0, 300), 9); addSpacer(2); }
      addLine();
    }

    // ── Decision Traces ────────────────────────────────────────
    const traces = report.decision_traces as DecisionTrace[] | undefined;
    if (traces && traces.length > 0) {
      addText("7. Decision Audit Trail", 16, "bold", [30, 120, 200]);
      addSpacer(4);
      for (const t of traces) {
        addText(`  ${t.agent_name}: ${t.output_signal} (${Math.round(t.confidence * 100)}% confidence)`, 9, "bold");
        if (t.evidence_citations?.length) {
          addText(`    Evidence: ${t.evidence_citations.slice(0, 2).join("; ")}`, 8);
        }
        addSpacer(1);
      }
    }

    // ── Footer on last page ────────────────────────────────────
    addSpacer(10);
    addText("Disclaimer: This report is generated by AI agents and should not be considered financial advice.", 7, "normal", [150, 150, 150]);
    addText("Sources: yfinance, SEC EDGAR, Alpha Vantage, FRED, USPTO, Google Trends, Vertex AI", 7, "normal", [150, 150, 150]);

    doc.save(`PyFinAgent_${ticker}_Report.pdf`);
  }, [ticker, report, financials]);

  return (
    <button
      onClick={generate}
      className={`rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 transition-colors hover:bg-sky-500/20 ${className ?? ""}`}
    >
      📄 Download PDF Report
    </button>
  );
}
