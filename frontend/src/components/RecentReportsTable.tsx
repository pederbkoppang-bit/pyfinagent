"use client";

/**
 * phase-16.42: Recent Reports table for the authenticated home page.
 *
 * Wired to the existing GET /api/reports/?limit=5 endpoint via the
 * `reports` prop (parent fetches in useEffect to keep the home page's
 * Promise.allSettled batch intact). Renders TICKER / COMPANY / ALPHA /
 * RECOMMENDATION / UPDATED columns with loading + empty + error states.
 *
 * "ALPHA" column displays `final_score` (0-10 composite quality score).
 * The pipeline does not currently emit a separate alpha field; this is
 * documented in the phase-16.42 contract and in the research brief.
 *
 * Strict no-hardcoded-data: every value comes from props -- there are
 * NO sample tickers, sample company names, or sample scores baked in.
 */

import { useRouter } from "next/navigation";
import Link from "next/link";
import type { ReportSummary } from "@/lib/types";
import { formatRelativeTime } from "@/lib/formatRelativeTime";
import { formatRecommendation } from "@/lib/formatRecommendation";
import { Files } from "@/lib/icons";

type Props = {
  reports: ReportSummary[];
  loaded: boolean;
  loadError: string | null;
};

function recColor(rec: string | null | undefined): string {
  const r = (rec ?? "").toUpperCase();
  if (r.includes("STRONG_BUY") || r.includes("STRONG BUY")) return "bg-emerald-500/20 text-emerald-400";
  if (r.includes("BUY")) return "bg-emerald-500/15 text-emerald-400";
  if (r.includes("STRONG_SELL") || r.includes("STRONG SELL")) return "bg-rose-500/20 text-rose-400";
  if (r.includes("SELL")) return "bg-rose-500/15 text-rose-400";
  return "bg-amber-500/15 text-amber-400";
}

function alphaColor(score: number | null | undefined): string {
  if (score == null) return "text-slate-500";
  if (score >= 8) return "text-emerald-400";
  if (score >= 6.5) return "text-sky-400";
  if (score >= 4.5) return "text-amber-400";
  return "text-rose-400";
}

export function RecentReportsTable({ reports, loaded, loadError }: Props) {
  const router = useRouter();

  return (
    <div className="h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/40">
      <div className="flex items-center justify-between border-b border-navy-700 px-4 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Recent Reports
        </h3>
        <Link href="/reports" className="text-xs text-sky-400 hover:text-sky-300">
          View all →
        </Link>
      </div>

      <div className="flex-1 overflow-x-auto">
        <table className="w-full text-left text-sm" aria-label="Recent reports">
          <thead className="border-b border-navy-700 bg-navy-800/60">
            <tr>
              <th className="px-4 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Ticker</th>
              <th className="px-4 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Company</th>
              <th className="px-4 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Alpha</th>
              <th className="px-4 py-2.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">Recommendation</th>
              <th className="px-4 py-2.5 text-right text-[10px] font-medium uppercase tracking-wider text-slate-500">Updated</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-700/50">
            {!loaded && [0, 1, 2, 3, 4].map((i) => (
              <tr key={`skel-${i}`} className="animate-pulse">
                <td className="px-4 py-3"><div className="h-4 w-12 rounded bg-navy-700/60" /></td>
                <td className="px-4 py-3"><div className="h-4 w-40 rounded bg-navy-700/60" /></td>
                <td className="px-4 py-3 text-right"><div className="ml-auto h-4 w-12 rounded bg-navy-700/60" /></td>
                <td className="px-4 py-3"><div className="h-5 w-20 rounded-full bg-navy-700/60" /></td>
                <td className="px-4 py-3 text-right"><div className="ml-auto h-4 w-16 rounded bg-navy-700/60" /></td>
              </tr>
            ))}

            {loaded && loadError && reports.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-12">
                  <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-center">
                    <p className="text-sm text-rose-300">{loadError}</p>
                  </div>
                </td>
              </tr>
            )}

            {loaded && !loadError && reports.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <Files size={36} weight="duotone" className="text-slate-600" />
                    <p className="mt-3 text-sm text-slate-400">No reports yet</p>
                    <p className="mt-1 text-xs text-slate-600">Run your first analysis from the Quick Actions panel</p>
                  </div>
                </td>
              </tr>
            )}

            {loaded && reports.map((r) => {
              const goto = () => router.push(`/reports?ticker=${encodeURIComponent(r.ticker)}`);
              return (
                <tr
                  key={`${r.ticker}-${r.analysis_date}`}
                  tabIndex={0}
                  role="button"
                  aria-label={`Open ${r.ticker} report`}
                  onClick={goto}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      goto();
                    }
                  }}
                  className="cursor-pointer transition-colors hover:bg-navy-700/40 focus:bg-navy-700/40 focus:outline-none focus:ring-1 focus:ring-sky-500/40"
                >
                  <td className="px-4 py-3 font-mono text-sm font-bold text-slate-100">{r.ticker}</td>
                  <td className="px-4 py-3 text-sm text-slate-300">
                    {r.company_name && r.company_name.trim() && r.company_name.trim().toUpperCase() !== r.ticker.toUpperCase() ? r.company_name : "—"}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono text-sm font-semibold ${alphaColor(r.final_score)}`}>
                    {r.final_score != null ? r.final_score.toFixed(2) : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${recColor(r.recommendation)}`}>
                      {formatRecommendation(r.recommendation)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-slate-500" suppressHydrationWarning>
                    {formatRelativeTime(r.analysis_date)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
