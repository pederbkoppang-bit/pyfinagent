"use client";

import { useCallback } from "react";
import jsPDF from "jspdf";
import type { SynthesisReport } from "@/lib/types";

interface PdfDownloadProps {
  ticker: string;
  report: SynthesisReport;
  className?: string;
}

export function PdfDownload({ ticker, report, className }: PdfDownloadProps) {
  const generate = useCallback(() => {
    const doc = new jsPDF();
    const margin = 20;
    let y = margin;

    const addText = (text: string, size: number, style: "normal" | "bold" = "normal") => {
      doc.setFontSize(size);
      doc.setFont("helvetica", style);
      const lines = doc.splitTextToSize(text, 170);
      for (const line of lines) {
        if (y > 270) {
          doc.addPage();
          y = margin;
        }
        doc.text(line, margin, y);
        y += size * 0.5;
      }
    };

    // Title
    addText(`PyFinAgent Analysis Report: ${ticker}`, 18, "bold");
    y += 6;

    // Score & Recommendation
    addText(`Final Score: ${report.final_weighted_score?.toFixed(2) ?? "N/A"} / 10`, 14, "bold");
    y += 2;
    addText(`Recommendation: ${report.recommendation.action}`, 12, "bold");
    y += 2;
    addText(`Justification: ${report.recommendation.justification}`, 10);
    y += 6;

    // Summary
    addText("Executive Summary", 14, "bold");
    y += 2;
    addText(report.final_summary, 10);
    y += 6;

    // Scoring Matrix
    addText("Scoring Matrix", 14, "bold");
    y += 2;
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
    y += 6;

    // Key Risks
    addText("Key Risks", 14, "bold");
    y += 2;
    for (const risk of report.key_risks) {
      addText(`• ${risk}`, 10);
    }

    doc.save(`PyFinAgent_${ticker}_Report.pdf`);
  }, [ticker, report]);

  return (
    <button
      onClick={generate}
      className={`rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 transition-colors hover:bg-sky-500/20 ${className ?? ""}`}
    >
      📄 Download PDF
    </button>
  );
}
