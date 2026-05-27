"use client";

// phase-44.4 -- TanStack v8 column factory for the reports history table.
//
// Sparkline column derives 30d score history per ticker from the same
// `reports` array (no backend change). Uses the Tailwind-SVG MiniSpark
// inline pattern from cycle 64 -- zero new deps.

import type { ColumnDef } from "@tanstack/react-table";
import type { ReportSummary } from "@/lib/types";
import { formatRecommendation } from "@/lib/formatRecommendation";

function scoreColor(r: string | null | undefined): string {
  const norm = (r ?? "").toUpperCase().replace(/_/g, " ").trim();
  if (!norm) return "text-slate-400";
  if (norm === "STRONG BUY" || norm === "BUY") return "text-emerald-400";
  if (norm === "STRONG SELL" || norm === "SELL") return "text-rose-400";
  if (norm === "HOLD") return "text-amber-400";
  return "text-slate-400";
}

function MiniSparkSVG({ data }: { data: number[] }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const W = 60;
  const H = 20;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - ((v - min) / range) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const lastDelta = data[data.length - 1] - data[0];
  const stroke = lastDelta >= 0 ? "#34d399" : "#fb7185";
  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${W} ${H}`}
      className="inline-block h-5 w-16"
      preserveAspectRatio="none"
    >
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

export function reportsColumns(
  tickerHistory: Record<string, number[]>,
): ColumnDef<ReportSummary, unknown>[] {
  return [
    {
      id: "ticker",
      accessorKey: "ticker",
      header: "Ticker",
      cell: ({ row }) => (
        <span className="font-mono font-semibold text-slate-100">{row.original.ticker}</span>
      ),
      meta: { align: "left" },
    },
    {
      id: "company",
      accessorKey: "company_name",
      header: "Company",
      cell: ({ row }) => (
        <span className="text-xs text-slate-400">{row.original.company_name ?? "—"}</span>
      ),
      meta: { align: "left" },
    },
    {
      id: "date",
      accessorKey: "analysis_date",
      header: "Date",
      cell: ({ row }) => (
        <span className="text-xs text-slate-500">
          {new Date(row.original.analysis_date).toLocaleDateString()}
        </span>
      ),
      meta: { align: "left" },
    },
    {
      id: "score",
      accessorKey: "final_score",
      header: "Score",
      cell: ({ row }) => (
        <span className="font-mono text-sky-300">{row.original.final_score.toFixed(2)}</span>
      ),
      meta: { align: "right", className: "tabular-nums" },
    },
    {
      id: "recommendation",
      accessorKey: "recommendation",
      header: "Recommendation",
      cell: ({ row }) => (
        <span className={`text-xs font-medium ${scoreColor(row.original.recommendation)}`}>
          {formatRecommendation(row.original.recommendation)}
        </span>
      ),
      meta: { align: "left" },
    },
    {
      id: "trend",
      accessorFn: (row) => {
        const hist = tickerHistory[row.ticker] ?? [];
        if (hist.length < 2) return 0;
        return hist[hist.length - 1] - hist[0];
      },
      header: "30d trend",
      cell: ({ row }) => {
        const hist = tickerHistory[row.original.ticker] ?? [];
        if (hist.length < 2) return <span className="text-slate-600 text-xs">—</span>;
        return <MiniSparkSVG data={hist} />;
      },
      meta: { align: "right" },
    },
  ];
}

// Helper: build a ticker -> score-series map from a reports list.
// The series is sorted by analysis_date ascending (oldest -> newest) so
// the sparkline reads left-to-right as time progresses. Limited to the
// last 30 entries per ticker.
export function buildTickerHistory(
  reports: ReportSummary[],
): Record<string, number[]> {
  const byTicker = new Map<string, ReportSummary[]>();
  for (const r of reports) {
    const existing = byTicker.get(r.ticker) ?? [];
    existing.push(r);
    byTicker.set(r.ticker, existing);
  }
  const out: Record<string, number[]> = {};
  for (const [ticker, list] of byTicker.entries()) {
    const sorted = [...list].sort((a, b) =>
      a.analysis_date.localeCompare(b.analysis_date),
    );
    const tail = sorted.slice(-30);
    out[ticker] = tail.map((r) => r.final_score);
  }
  return out;
}
