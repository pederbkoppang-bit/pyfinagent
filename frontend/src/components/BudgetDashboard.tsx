"use client";

import { useEffect, useMemo, useState } from "react";
import { BentoCard } from "@/components/BentoCard";
import {
  CurrencyDollar,
  TrendDown,
  TrendUp,
  Gauge,
  Warning,
  CheckCircle,
  Clock,
} from "@phosphor-icons/react";
import {
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  TooltipProps,
  ReferenceLine,
  Area,
} from "recharts";

// ── Types ───────────────────────────────────────────────────────

interface CostItem {
  category: string;
  monthly_nok: number;
  /** @deprecated Use monthly_nok */
  monthly_usd?: number;
  type: "fixed" | "estimated" | "projected" | "actual";
  note: string;
}

interface BudgetSummary {
  total_fixed_monthly: number;
  total_gcp_monthly: number;
  total_monthly: number;
  monthly_budget: number;
  budget_utilization_pct: number;
  runway_months: number;
}

interface MonthlyHistory {
  month: string;
  gcp_net: number;
  claude_max: number;
  other_fixed: number;
  total: number;
  services: Record<string, number>;
}

interface BudgetData {
  currency: string;
  currency_symbol: string;
  fixed_costs: CostItem[];
  gcp_costs: CostItem[];
  monthly_history: MonthlyHistory[];
  summary: BudgetSummary;
  status: string;
  data_source: string;
}

// ── Type badge ──────────────────────────────────────────────────

function TypeBadge({ type }: { type: string }) {
  const styles: Record<string, string> = {
    fixed: "bg-sky-500/15 text-sky-400",
    estimated: "bg-amber-500/15 text-amber-400",
    projected: "bg-purple-500/15 text-purple-400",
    actual: "bg-emerald-500/15 text-emerald-400",
  };
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${styles[type] || "bg-slate-700 text-slate-400"}`}
    >
      {type}
    </span>
  );
}

// ── Utilization bar ─────────────────────────────────────────────

function UtilizationBar({ pct }: { pct: number }) {
  const color =
    pct >= 100 ? "bg-red-500" : pct >= 80 ? "bg-amber-500" : "bg-emerald-500";
  const width = Math.min(100, pct);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">Budget Utilization</span>
        <span
          className={`font-mono font-semibold ${
            pct >= 100
              ? "text-red-400"
              : pct >= 80
                ? "text-amber-400"
                : "text-emerald-400"
          }`}
        >
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-700">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────

export function BudgetDashboard() {
  const [data, setData] = useState<BudgetData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(
      `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/backtest/budget/summary`
    )
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError("Failed to load budget data"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-12 text-sm text-slate-500">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
        Loading budget data...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-red-400">
        {error || "No budget data available"}
      </div>
    );
  }

  const s = data.summary;
  const sym = data.currency_symbol || "kr";

  return (
    <div className="space-y-6">
      {/* Hero metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            This Month
          </p>
          <p className="font-mono text-lg font-semibold text-slate-200">
            {sym}{s.total_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            GCP Actual
          </p>
          <p className="font-mono text-lg font-semibold text-sky-400">
            {sym}{s.total_gcp_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Fixed Costs
          </p>
          <p className="font-mono text-lg font-semibold text-slate-300">
            {sym}{s.total_fixed_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Budget
          </p>
          <p className="font-mono text-lg font-semibold text-slate-300">
            {sym}{s.monthly_budget.toFixed(0)}
          </p>
        </div>
        <div
          className={`rounded-lg border px-3 py-2 ${
            s.budget_utilization_pct >= 100
              ? "border-red-800/50 bg-red-900/20"
              : s.budget_utilization_pct >= 80
                ? "border-amber-800/50 bg-amber-900/20"
                : "border-emerald-800/50 bg-emerald-900/20"
          }`}
        >
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Utilization
          </p>
          <p
            className={`font-mono text-lg font-semibold ${
              s.budget_utilization_pct >= 100
                ? "text-red-400"
                : s.budget_utilization_pct >= 80
                  ? "text-amber-400"
                  : "text-emerald-400"
            }`}
          >
            {s.budget_utilization_pct.toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Utilization bar */}
      <BentoCard>
        <UtilizationBar pct={s.budget_utilization_pct} />
        <p className="mt-2 text-[10px] text-slate-500">
          {data.status === "under_budget" ? (
            <span className="flex items-center gap-1">
              <CheckCircle size={12} className="text-emerald-400" />
              Under budget — {sym}{(s.monthly_budget - s.total_monthly).toFixed(0)}/mo remaining
            </span>
          ) : (
            <span className="flex items-center gap-1">
              <Warning size={12} className="text-red-400" />
              Over budget by {sym}{(s.total_monthly - s.monthly_budget).toFixed(0)}/mo
            </span>
          )}
        </p>
      </BentoCard>

      {/* Cash Flow Chart */}
      {data.monthly_history.length > 0 && (() => {
        // Build chart data: actual months + 3 months forecast
        const history = data.monthly_history;
        const lastMonths = history.slice(-3);
        const avgBurn = lastMonths.reduce((sum, m) => sum + m.total, 0) / lastMonths.length;

        // Parse last month to project forward
        const lastMonth = history[history.length - 1];
        const [lastY, lastM] = lastMonth.month.split("-").map(Number);

        const chartData: { month: string; cashOut?: number; forecast?: number; budget: number; isForcast?: boolean }[] = [];

        // Actual months
        for (const m of history) {
          chartData.push({
            month: m.month,
            cashOut: m.total,
            budget: s.monthly_budget,
          });
        }

        // Forecast 3 months ahead
        for (let i = 1; i <= 3; i++) {
          const fDate = new Date(lastY, lastM - 1 + i, 1);
          const fMonth = `${fDate.getFullYear()}-${String(fDate.getMonth() + 1).padStart(2, "0")}`;
          chartData.push({
            month: fMonth,
            forecast: Math.round(avgBurn),
            budget: s.monthly_budget,
            isForcast: true,
          });
        }

        return (
          <BentoCard>
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
              <TrendDown size={16} className="text-red-400" />
              Cash Flow
              <span className="ml-auto text-[10px] font-normal text-slate-500">
                Forecast based on {lastMonths.length}-month average: {sym}{Math.round(avgBurn)}/mo
              </span>
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="month"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickFormatter={(v: number) => `${sym}${v}`}
                    label={{
                      value: `${data.currency || "NOK"} / month`,
                      angle: -90,
                      position: "insideLeft",
                      offset: -4,
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    content={({ active, payload, label }: TooltipProps<number, string>) => {
                      if (!active || !payload?.length) return null;
                      const isForecast = payload[0]?.payload?.isForcast;
                      return (
                        <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 shadow-xl">
                          <p className="mb-1 text-xs font-semibold text-slate-200">
                            {label} {isForecast ? "(forecast)" : ""}
                          </p>
                          {payload.map((p) => (
                            p.value != null && (
                              <p key={p.dataKey as string} className="text-xs text-slate-400">
                                {p.dataKey === "cashOut" ? "Actual" : p.dataKey === "forecast" ? "Forecast" : "Budget"}:{" "}
                                <span className={`font-mono ${p.dataKey === "cashOut" ? "text-red-400" : p.dataKey === "forecast" ? "text-amber-400" : "text-slate-500"}`}>
                                  {sym}{Number(p.value).toFixed(0)}
                                </span>
                              </p>
                            )
                          ))}
                        </div>
                      );
                    }}
                  />

                  {/* Budget line */}
                  <ReferenceLine
                    y={s.monthly_budget}
                    stroke="#22c55e"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    label={{
                      value: `Budget ${sym}${s.monthly_budget}`,
                      position: "right",
                      fill: "#22c55e",
                      fontSize: 10,
                    }}
                  />

                  {/* Actual spend bars (red) */}
                  <Bar
                    dataKey="cashOut"
                    fill="#ef4444"
                    fillOpacity={0.7}
                    radius={[4, 4, 0, 0]}
                    name="Actual"
                  />

                  {/* Forecast bars (amber, dashed look via lower opacity) */}
                  <Bar
                    dataKey="forecast"
                    fill="#f59e0b"
                    fillOpacity={0.4}
                    radius={[4, 4, 0, 0]}
                    name="Forecast"
                    strokeDasharray="4 4"
                    stroke="#f59e0b"
                    strokeWidth={1}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            {/* Legend */}
            <div className="mt-2 flex items-center justify-center gap-6 text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-4 rounded-sm bg-red-500/70" />
                Actual spend
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-4 rounded-sm border border-amber-500 bg-amber-500/40" />
                Forecast
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 border-t-2 border-dashed border-emerald-500" />
                Budget
              </span>
            </div>
          </BentoCard>
        );
      })()}

      {/* Cost breakdown tables */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Fixed costs */}
        <BentoCard>
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-300">
            <CurrencyDollar size={16} className="text-sky-400" />
            Fixed Costs
          </h3>
          <div className="overflow-hidden rounded-lg border border-navy-700">
            <table className="w-full text-sm">
              <thead className="border-b border-navy-700 bg-navy-800/80">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-slate-400">
                    Category
                  </th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                    Monthly
                  </th>
                  <th className="px-3 py-2 text-xs font-medium text-slate-400">
                    Type
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/50">
                {data.fixed_costs.map((c) => (
                  <tr
                    key={c.category}
                    className="transition-colors hover:bg-navy-700/40"
                  >
                    <td className="px-3 py-2">
                      <p className="text-xs text-slate-200">{c.category}</p>
                      <p className="text-[10px] text-slate-500">{c.note}</p>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-slate-300">
                      {sym}{(c.monthly_nok ?? c.monthly_usd ?? 0).toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-center">
                      <TypeBadge type={c.type} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BentoCard>

        {/* GCP costs (real billing data) */}
        <BentoCard>
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-300">
            <Gauge size={16} className="text-sky-400" />
            GCP Costs (This Month)
          </h3>
          {data.gcp_costs.length > 0 ? (
            <div className="overflow-hidden rounded-lg border border-navy-700">
              <table className="w-full text-sm">
                <thead className="border-b border-navy-700 bg-navy-800/80">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-slate-400">
                      Service
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                      Net Cost
                    </th>
                    <th className="px-3 py-2 text-xs font-medium text-slate-400">
                      Source
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-navy-700/50">
                  {data.gcp_costs.map((c) => (
                    <tr
                      key={c.category}
                      className="transition-colors hover:bg-navy-700/40"
                    >
                      <td className="px-3 py-2">
                        <p className="text-xs text-slate-200">{c.category}</p>
                        <p className="text-[10px] text-slate-500">{c.note}</p>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-slate-300">
                        {sym}{(c.monthly_nok ?? c.monthly_usd ?? 0).toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <TypeBadge type={c.type} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="py-4 text-center text-xs text-slate-500">
              No GCP charges this month
            </p>
          )}
        </BentoCard>
      </div>

      {/* Monthly Trend */}
      {data.monthly_history.length > 0 && (
        <BentoCard>
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-300">
            <Clock size={16} className="text-slate-400" />
            Monthly Cost Trend
          </h3>
          <div className="overflow-hidden rounded-lg border border-navy-700">
            <table className="w-full text-sm">
              <thead className="border-b border-navy-700 bg-navy-800/80">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-slate-400">
                    Month
                  </th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                    GCP (net)
                  </th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                    Claude Max
                  </th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                    Other
                  </th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-slate-400">
                    Total
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/50">
                {data.monthly_history.map((m) => (
                  <tr
                    key={m.month}
                    className="transition-colors hover:bg-navy-700/40"
                  >
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">
                      {m.month}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-sky-400">
                      {sym}{m.gcp_net.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">
                      {sym}{m.claude_max.toFixed(0)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">
                      {sym}{m.other_fixed.toFixed(0)}
                    </td>
                    <td
                      className={`px-3 py-2 text-right font-mono text-xs font-semibold ${
                        m.total > 5400
                          ? "text-red-400"
                          : m.total > 3800
                            ? "text-amber-400"
                            : "text-emerald-400"
                      }`}
                    >
                      {sym}{m.total.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BentoCard>
      )}

      {/* Data source */}
      <p className="text-center text-[10px] text-slate-600">
        {data.data_source}
      </p>
    </div>
  );
}
