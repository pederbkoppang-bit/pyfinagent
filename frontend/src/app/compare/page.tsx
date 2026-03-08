"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { StockChart } from "@/components/StockChart";
import { listReports, getReport } from "@/lib/api";
import type { ReportSummary, SynthesisReport } from "@/lib/types";

function scoreColor(action: string): string {
  const lower = action.toLowerCase();
  if (lower.includes("strong buy")) return "text-emerald-400";
  if (lower.includes("buy")) return "text-emerald-300";
  if (lower.includes("sell")) return "text-rose-400";
  return "text-slate-300";
}

interface FullReport {
  analysis_date: string;
  synthesis: SynthesisReport;
}

export default function ComparePage() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState<FullReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listReports(50)
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const startCompare = async () => {
    setComparing(true);
    setError(null);
    const results: FullReport[] = [];
    for (const key of selected) {
      const [ticker] = key.split("|");
      try {
        const full = await getReport(ticker);
        const synth = full.full_report_json?.final_synthesis ?? full.full_report_json ?? {};
        results.push({
          analysis_date: full.analysis_date ?? key.split("|")[1] ?? "",
          synthesis: synth as SynthesisReport,
        });
      } catch (e) {
        setError(`Failed to load ${ticker}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    setLoaded(results);
    setComparing(false);
  };

  // Detect dominant ticker for chart
  const tickers = [...selected].map((k) => k.split("|")[0]);
  const dominantTicker = tickers.length > 0 ? tickers[0] : null;

  const pillars = [
    { key: "pillar_1_corporate", label: "Corporate" },
    { key: "pillar_2_industry", label: "Industry" },
    { key: "pillar_3_valuation", label: "Valuation" },
    { key: "pillar_4_sentiment", label: "Sentiment" },
    { key: "pillar_5_governance", label: "Governance" },
  ] as const;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <h2 className="mb-2 text-2xl font-bold text-slate-100">Compare Reports</h2>
        <p className="mb-6 text-sm text-slate-500">
          Select multiple reports to compare scoring and recommendations side-by-side
        </p>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
            <pre className="whitespace-pre-wrap text-xs text-rose-300">{error}</pre>
          </div>
        )}

        {/* Report selection */}
        {loading && <p className="text-slate-400">Loading reports...</p>}
        {!loading && reports.length === 0 && (
          <p className="text-slate-500">No reports found yet.</p>
        )}

        {!loading && reports.length > 0 && loaded.length === 0 && (
          <>
            <div className="mb-4 space-y-2">
              {reports.map((r) => {
                const key = `${r.ticker}|${r.analysis_date}`;
                const isSelected = selected.has(key);
                return (
                  <button
                    key={key}
                    onClick={() => toggle(key)}
                    className={`flex w-full items-center justify-between rounded-lg border p-3 text-left transition-colors ${
                      isSelected
                        ? "border-sky-500/50 bg-sky-500/10"
                        : "border-slate-800 bg-slate-900/50 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={`h-4 w-4 rounded border ${
                          isSelected ? "border-sky-400 bg-sky-400" : "border-slate-600"
                        }`}
                      />
                      <span className="font-mono font-bold text-slate-200">{r.ticker}</span>
                      <span className="text-xs text-slate-500">
                        {new Date(r.analysis_date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-sm text-sky-300">{r.final_score.toFixed(2)}</span>
                      <span className={`text-xs font-medium ${scoreColor(r.recommendation)}`}>
                        {r.recommendation}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            <button
              onClick={startCompare}
              disabled={selected.size < 2 || comparing}
              className="rounded-lg bg-sky-600 px-6 py-2.5 font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {comparing ? "Loading..." : `Compare ${selected.size} Reports`}
            </button>
          </>
        )}

        {/* Comparison view */}
        {loaded.length > 0 && (
          <div className="mt-6 space-y-6">
            <button
              onClick={() => setLoaded([])}
              className="text-sm text-sky-400 hover:underline"
            >
              ← Back to selection
            </button>

            {/* Price chart context */}
            {dominantTicker && (
              <StockChart ticker={dominantTicker} />
            )}

            {/* Score comparison table */}
            <BentoCard>
              <h3 className="mb-4 text-lg font-semibold text-slate-300">
                📊 Score Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-left">
                      <th className="px-3 py-2 text-slate-400">Date</th>
                      <th className="px-3 py-2 text-slate-400">Score</th>
                      <th className="px-3 py-2 text-slate-400">Verdict</th>
                      {pillars.map((p) => (
                        <th key={p.key} className="px-3 py-2 text-slate-400">
                          {p.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {loaded.map((r, i) => (
                      <tr key={i} className="border-b border-slate-800">
                        <td className="px-3 py-2 text-xs text-slate-400">
                          {new Date(r.analysis_date).toLocaleDateString()}
                        </td>
                        <td className="px-3 py-2 font-mono font-bold text-sky-300">
                          {r.synthesis.final_weighted_score?.toFixed(2) ?? "N/A"}
                        </td>
                        <td
                          className={`px-3 py-2 font-medium ${scoreColor(
                            r.synthesis.recommendation?.action ?? ""
                          )}`}
                        >
                          {r.synthesis.recommendation?.action ?? "N/A"}
                        </td>
                        {pillars.map((p) => (
                          <td key={p.key} className="px-3 py-2 font-mono text-slate-300">
                            {r.synthesis.scoring_matrix?.[p.key]?.toFixed(1) ?? "—"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </BentoCard>

            {/* Detailed qualitative breakdown */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-300">
                📝 Qualitative Breakdown
              </h3>
              {loaded.map((r, i) => (
                <BentoCard key={i}>
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-xs text-slate-500">
                      {new Date(r.analysis_date).toLocaleDateString()} — {r.synthesis.recommendation?.action}
                    </span>
                    <span className="font-mono text-sm text-sky-400">
                      {r.synthesis.final_weighted_score?.toFixed(2)}
                    </span>
                  </div>
                  <p className="mb-3 text-sm font-medium text-slate-300">Justification</p>
                  <p className="mb-4 text-sm text-slate-400">
                    {r.synthesis.recommendation?.justification ?? "N/A"}
                  </p>
                  <p className="mb-3 text-sm font-medium text-slate-300">Summary</p>
                  <p className="text-sm text-slate-400">{r.synthesis.final_summary ?? "N/A"}</p>
                </BentoCard>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
