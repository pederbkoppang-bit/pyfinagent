"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Sidebar } from "@/components/Sidebar";
import { BentoCard } from "@/components/BentoCard";
import { listReports } from "@/lib/api";
import type { ReportSummary } from "@/lib/types";
import { ArrowsLeftRight } from "@phosphor-icons/react";

function scoreColor(recommendation: string): string {
  const lower = recommendation.toLowerCase();
  if (lower.includes("strong buy")) return "text-emerald-400";
  if (lower.includes("buy")) return "text-emerald-300";
  if (lower.includes("sell")) return "text-rose-400";
  return "text-slate-300";
}

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    listReports(50)
      .then(setReports)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = filter
    ? reports.filter((r) => r.ticker.includes(filter.toUpperCase()))
    : reports;

  // Unique tickers for quick-filter chips
  const tickers = [...new Set(reports.map((r) => r.ticker))];

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Past Reports</h2>
            <p className="text-sm text-slate-500">
              Historical analysis reports stored in BigQuery
            </p>
          </div>
          <Link
            href="/compare"
            className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-400 transition-colors hover:bg-sky-500/20"
          >
            <ArrowsLeftRight size={16} className="inline" /> Compare Reports
          </Link>
        </div>

        {/* Ticker filter */}
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <input
            type="text"
            placeholder="Filter by ticker..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-40 rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
          />
          {tickers.map((t) => (
            <button
              key={t}
              onClick={() => setFilter(filter === t ? "" : t)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                filter === t
                  ? "bg-sky-500/20 text-sky-300"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              {t}
            </button>
          ))}
          {filter && (
            <button
              onClick={() => setFilter("")}
              className="text-xs text-slate-500 hover:text-slate-300"
            >
              Clear
            </button>
          )}
        </div>

        {loading && <p className="text-slate-400">Loading reports...</p>}
        {error && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
            <p className="text-sm font-medium text-rose-400">Error loading reports</p>
            <pre className="mt-2 whitespace-pre-wrap text-xs text-rose-300/80">{error}</pre>
          </div>
        )}

        {!loading && filtered.length === 0 && (
          <p className="text-slate-500">
            {filter ? `No reports found for "${filter}".` : "No reports found yet."}
          </p>
        )}

        <div className="space-y-3">
          {filtered.map((r, i) => {
            const isExpanded = expanded === i;
            return (
              <BentoCard key={i}>
                <button
                  onClick={() => setExpanded(isExpanded ? null : i)}
                  className="flex w-full items-center justify-between text-left"
                >
                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-lg font-bold text-slate-100">
                        {r.ticker}
                      </span>
                      {r.company_name && (
                        <span className="text-sm text-slate-500">
                          {r.company_name}
                        </span>
                      )}
                      <span className="text-xs text-slate-600">
                        {isExpanded ? "▲" : "▼"}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-500">
                      {new Date(r.analysis_date).toLocaleString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-2xl font-bold text-sky-300">
                      {r.final_score.toFixed(2)}
                    </p>
                    <p className={`text-sm font-medium ${scoreColor(r.recommendation)}`}>
                      {r.recommendation}
                    </p>
                  </div>
                </button>

                {/* Expanded detail */}
                {isExpanded && (
                  <div className="mt-4 border-t border-slate-800 pt-4">
                    <p className="text-sm leading-relaxed text-slate-400">
                      {r.summary}
                    </p>
                  </div>
                )}
              </BentoCard>
            );
          })}
        </div>
      </main>
    </div>
  );
}
