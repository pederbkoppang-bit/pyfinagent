"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { IconWarning } from "@/lib/icons";
import type { PaperReconciliation } from "@/lib/types";

interface Props {
  reconciliation: PaperReconciliation | null;
  loading?: boolean;
}

export function PaperReconciliationChart({ reconciliation, loading = false }: Props) {
  if (loading) {
    return (
      <div className="flex items-center gap-3 py-12 text-slate-400">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
        Loading reconciliation...
      </div>
    );
  }

  if (!reconciliation || reconciliation.note === "insufficient_snapshots" || reconciliation.series.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <IconWarning size={48} weight="duotone" className="text-slate-600" />
        <p className="mt-4 text-lg text-slate-400">Not enough history yet</p>
        <p className="mt-1 text-sm text-slate-600">
          Reality-gap reconciliation needs at least 2 daily snapshots.
        </p>
      </div>
    );
  }

  const { series, summary } = reconciliation;

  return (
    <div className="space-y-4">
      {summary.alert && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
          <p className="text-sm text-rose-300">
            Divergence alert: latest gap {summary.latest_divergence_pct.toFixed(2)}% exceeds{" "}
            {summary.alert_threshold_pct.toFixed(1)}% threshold. Check execution drift,
            stale signals, or schema regressions.
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Points</p>
          <p className="mt-1 text-2xl font-bold text-slate-100">{summary.n_points}</p>
        </div>
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Latest gap</p>
          <p
            className={
              "mt-1 text-2xl font-bold " +
              (summary.alert ? "text-rose-400" : "text-emerald-400")
            }
          >
            {summary.latest_divergence_pct.toFixed(2)}%
          </p>
        </div>
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Max gap</p>
          <p className="mt-1 text-2xl font-bold text-slate-100">
            {summary.max_divergence_pct.toFixed(2)}%
          </p>
        </div>
        <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Threshold</p>
          <p className="mt-1 text-2xl font-bold text-slate-100">
            {summary.alert_threshold_pct.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-300">
          Paper-live vs shadow backtest
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={series}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(5)}
            />
            <YAxis
              yAxisId="nav"
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
            />
            <YAxis
              yAxisId="div"
              orientation="right"
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickFormatter={(v: number) => `${v.toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                border: "1px solid #334155",
                borderRadius: 8,
              }}
              labelStyle={{ color: "#94a3b8" }}
            />
            <Legend />
            <Line
              yAxisId="nav"
              type="monotone"
              dataKey="paper_nav"
              name="Paper NAV"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={false}
            />
            <Line
              yAxisId="nav"
              type="monotone"
              dataKey="backtest_nav"
              name="Shadow backtest NAV"
              stroke="#22c55e"
              strokeWidth={1.5}
              strokeDasharray="5 5"
              dot={false}
            />
            <Line
              yAxisId="div"
              type="monotone"
              dataKey="divergence_pct"
              name="Divergence %"
              stroke="#f97316"
              strokeWidth={1.25}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
