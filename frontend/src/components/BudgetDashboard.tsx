"use client";

import { useEffect, useState } from "react";
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

// ── Types ───────────────────────────────────────────────────────

interface CostItem {
  category: string;
  monthly_usd: number;
  type: "fixed" | "estimated" | "projected" | "actual";
  note: string;
}

interface BudgetSummary {
  total_fixed_monthly: number;
  total_variable_monthly: number;
  total_monthly: number;
  monthly_budget: number;
  budget_utilization_pct: number;
  cash_available: number;
  runway_months: number;
}

interface BudgetData {
  fixed_costs: CostItem[];
  variable_costs: CostItem[];
  summary: BudgetSummary;
  status: string;
  note: string;
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

  return (
    <div className="space-y-6">
      {/* Hero metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Monthly Burn
          </p>
          <p className="font-mono text-lg font-semibold text-slate-200">
            ${s.total_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Budget
          </p>
          <p className="font-mono text-lg font-semibold text-slate-300">
            ${s.monthly_budget.toFixed(0)}
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
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Fixed
          </p>
          <p className="font-mono text-lg font-semibold text-slate-300">
            ${s.total_fixed_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            Variable
          </p>
          <p className="font-mono text-lg font-semibold text-slate-300">
            ${s.total_variable_monthly.toFixed(0)}
          </p>
        </div>
        <div className="rounded-lg border border-sky-800/50 bg-sky-900/20 px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-sky-500">
            Runway
          </p>
          <p className="font-mono text-lg font-semibold text-sky-400">
            {s.runway_months} mo
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
              Under budget — ${(s.monthly_budget - s.total_monthly).toFixed(0)}/mo remaining
            </span>
          ) : (
            <span className="flex items-center gap-1">
              <Warning size={12} className="text-red-400" />
              Over budget by ${(s.total_monthly - s.monthly_budget).toFixed(0)}/mo
            </span>
          )}
        </p>
      </BentoCard>

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
                      ${c.monthly_usd.toFixed(2)}
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

        {/* Variable costs */}
        <BentoCard>
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-300">
            <Gauge size={16} className="text-amber-400" />
            Variable Costs
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
                {data.variable_costs.map((c) => (
                  <tr
                    key={c.category}
                    className="transition-colors hover:bg-navy-700/40"
                  >
                    <td className="px-3 py-2">
                      <p className="text-xs text-slate-200">{c.category}</p>
                      <p className="text-[10px] text-slate-500">{c.note}</p>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-slate-300">
                      ${c.monthly_usd.toFixed(2)}
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
      </div>

      {/* Note */}
      <p className="text-center text-[10px] text-slate-600">
        {data.note}
      </p>
    </div>
  );
}
