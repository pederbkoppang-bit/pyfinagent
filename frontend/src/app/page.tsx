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
import { getAnalysisStatus, startAnalysis } from "@/lib/api";
import type { AnalysisStatusResponse, SynthesisReport } from "@/lib/types";

export default function DashboardPage() {
  const [ticker, setTicker] = useState("");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [status, setStatus] = useState<AnalysisStatusResponse | null>(null);
  const [report, setReport] = useState<SynthesisReport | null>(null);
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
    setLoading(true);

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
          <div className="mb-6 rounded-lg border border-rose-900 bg-rose-950/50 p-4 text-sm text-rose-200">
            {error}
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
            {/* Alpha Score */}
            <div className="col-span-12 md:col-span-4">
              <AlphaScoreCard report={report} />
            </div>

            {/* Investment Thesis */}
            <div className="col-span-12 md:col-span-8">
              <InvestmentThesisCard report={report} />
            </div>

            {/* Scoring Matrix */}
            <div className="col-span-12 md:col-span-6">
              <ScoringMatrixCard report={report} />
            </div>

            {/* Risks */}
            <div className="col-span-12 md:col-span-6">
              <RisksCard risks={report.key_risks} />
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
              The system will orchestrate 9 specialized AI agents to produce an
              evidence-based investment report
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
