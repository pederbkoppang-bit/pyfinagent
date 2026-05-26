"use client";

// phase-44.2 -- cockpit helper components hoisted out of the old
// /paper-trading monolith. Pure presentational; consume props.

import { clsx } from "clsx";
import NumberFlow from "@number-flow/react";
import type {
  PaperPerformance,
  PaperPortfolio,
  PaperPosition,
  PaperTradingStatus,
} from "@/lib/types";
// phase-76 (2026-05-26): trend tracker for the data-pyfa-trend host
// attribute. NumberFlow does not ship trend-coloring CSS parts, so we
// emit our own host attribute that globals.css targets via
// number-flow[data-pyfa-trend="up"]::part(digit) etc.
import { useTrend } from "@/lib/use-trend";
// phase-75 (2026-05-26): Google-Finance digit-flip animation via
// @number-flow/react@0.6.0 (researcher ad12953b2b579e884). Cycle-74's
// background-tint flash was the Bloomberg pattern; the operator wanted
// Google's per-digit slide (382.18 -> 382.45 slides only "18"). NumberFlow
// owns its prev-value tracking + animation timing + prefers-reduced-motion
// fallback (instant snap on `respectMotionPreference: true` default), so
// we delete the cycle-74 hook entirely. aria-live="off" stays per MDN
// stock-ticker default. Dollar + PnlBadge are shared by positions table,
// trades table, and SummaryHero MetricCards -- one swap covers all
// consumers. `willChange` per researcher Section 5 perf guidance.

export function PnlBadge({ value }: { value: number | null | undefined }) {
  const trend = useTrend(value);
  if (value == null) return <span className="text-slate-500">—</span>;
  const isPositive = value >= 0;
  const colorClass = isPositive ? "text-emerald-400" : "text-rose-400";
  return (
    <NumberFlow
      value={value / 100}
      format={{
        style: "percent",
        signDisplay: "always",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }}
      transformTiming={{ duration: 700 }}
      willChange
      aria-live="off"
      data-pyfa-trend={trend}
      className={colorClass}
    />
  );
}

export function Dollar({ value }: { value: number | null | undefined }) {
  const trend = useTrend(value);
  if (value == null) return <span className="text-slate-500">—</span>;
  return (
    <NumberFlow
      value={value}
      format={{
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }}
      transformTiming={{ duration: 700 }}
      willChange
      aria-live="off"
      data-pyfa-trend={trend}
      className="text-slate-100"
    />
  );
}

export function MetricCard({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100">{children}</p>
    </div>
  );
}

// phase-44.2 cycle-67 UX-audit fix: KPI tile values need consistent
// color semantics. Neutral values (NAV, Cash, Positions) use slate-100;
// performance ratios (Sharpe) get color thresholds; +/- metrics (P&L,
// vs SPY) keep the PnlBadge green/rose treatment.
function sharpeColor(value: number | null | undefined): string {
  if (value == null) return "text-slate-500";
  if (value >= 1) return "text-emerald-400";
  if (value >= 0) return "text-amber-400";
  return "text-rose-400";
}

export function SharpeValue({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-slate-500">—</span>;
  return <span className={sharpeColor(value)}>{value.toFixed(2)}</span>;
}

export function SummaryHero({
  status,
  perf,
  liveNav,
  liveTotalPnlPct,
}: {
  status: PaperTradingStatus | null;
  perf: PaperPerformance | null;
  liveNav: number | null;
  liveTotalPnlPct: number | null;
}) {
  const navDisplay = liveNav ?? status?.portfolio.nav ?? null;
  const pnlDisplay = liveTotalPnlPct ?? status?.portfolio.pnl_pct ?? null;
  const bench = status?.portfolio.benchmark_return_pct ?? 0;
  const vsBench = (pnlDisplay ?? 0) - bench;
  return (
    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <MetricCard label="NAV"><Dollar value={navDisplay} /></MetricCard>
      <MetricCard label="Cash"><Dollar value={status?.portfolio.cash} /></MetricCard>
      <MetricCard label="Total P&L"><PnlBadge value={pnlDisplay} /></MetricCard>
      <MetricCard label="vs SPY"><PnlBadge value={vsBench} /></MetricCard>
      <MetricCard label="Sharpe"><SharpeValue value={perf?.sharpe_ratio} /></MetricCard>
      <MetricCard label="Positions">
        <span className="text-slate-100">{status?.position_count ?? 0}</span>
      </MetricCard>
    </div>
  );
}

export function PaperVsBacktestCard({
  perf,
  snapshotsLen,
}: {
  perf: PaperPerformance | null;
  snapshotsLen: number;
}) {
  const sharpe = perf?.sharpe_ratio ?? 0;
  const maxDd = perf?.max_drawdown_pct ?? 0;
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
        Paper vs Backtest
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">Sharpe</span>
          <span>
            <span className={`font-mono ${sharpe >= 0.82 ? "text-emerald-400" : "text-rose-400"}`}>
              {perf?.sharpe_ratio?.toFixed(2) ?? "—"}
            </span>
            <span className="mx-1 text-slate-600">/</span>
            <span className="text-slate-500">1.17</span>
            {sharpe >= 0.82 ? (
              <span className="ml-2 text-xs text-emerald-400">OK</span>
            ) : sharpe > 0 ? (
              <span className="ml-2 text-xs text-rose-400">BELOW 0.7x</span>
            ) : null}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Max DD</span>
          <span>
            <span className={`font-mono ${maxDd > -15 ? "text-emerald-400" : "text-rose-400"}`}>
              {perf?.max_drawdown_pct?.toFixed(1) ?? "—"}%
            </span>
            <span className="mx-1 text-slate-600">/</span>
            <span className="text-slate-500">-12.0%</span>
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Snapshots</span>
          <span className="font-mono text-slate-300">{snapshotsLen}</span>
        </div>
      </div>
    </div>
  );
}

export function RiskMonitorCard({
  perf,
  positions,
  portfolio,
  tickerMeta,
}: {
  perf: PaperPerformance | null;
  positions: PaperPosition[];
  portfolio: PaperPortfolio | null;
  tickerMeta: Record<string, { sector?: string }>;
}) {
  const maxDd = perf?.max_drawdown_pct ?? 0;
  const navDenom = portfolio?.total_nav ?? 10000;
  const concentrations = positions.map(
    (p) => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / navDenom) * 100,
  );
  const maxPos = concentrations.length > 0 ? Math.max(...concentrations) : null;
  const concentrationHigh = maxPos != null && maxPos > 20;

  const sectorCounts: Record<string, number> = {};
  for (const p of positions) {
    const s = tickerMeta[p.ticker]?.sector || "Unknown";
    sectorCounts[s] = (sectorCounts[s] ?? 0) + 1;
  }
  const sectorEntries = Object.entries(sectorCounts);
  let maxSectorName = "";
  let maxSectorCount = 0;
  for (const [name, count] of sectorEntries) {
    if (count > maxSectorCount) {
      maxSectorName = name;
      maxSectorCount = count;
    }
  }
  const sectorConcentrationHigh =
    positions.length >= 3 && maxSectorCount / positions.length > 0.5;
  const sectorConcentrationLabel = sectorConcentrationHigh
    ? `HIGH (${maxSectorCount}/${positions.length} ${maxSectorName})`
    : "OK";
  return (
    <div className="rounded-xl border border-navy-700 bg-navy-800/70 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
        Risk Monitor
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-slate-400">Kill switch (-15%)</span>
          <span
            className={clsx(
              "rounded px-2 py-0.5 text-xs font-medium",
              maxDd > -10
                ? "bg-emerald-500/10 text-emerald-400"
                : maxDd > -13
                  ? "bg-amber-500/10 text-amber-400"
                  : "bg-rose-500/10 text-rose-400",
            )}
          >
            {maxDd > -10 ? "SAFE" : maxDd > -13 ? "WARNING" : "DANGER"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Max position</span>
          <span className="font-mono text-slate-300">
            {maxPos != null ? `${maxPos.toFixed(1)}%` : "—"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Position size</span>
          <span
            className={clsx(
              "rounded px-2 py-0.5 text-xs font-medium",
              concentrationHigh
                ? "bg-amber-500/10 text-amber-400"
                : "bg-emerald-500/10 text-emerald-400",
            )}
          >
            {concentrationHigh ? `HIGH (>20%)` : "OK"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Sector concentration</span>
          <span
            className={clsx(
              "rounded px-2 py-0.5 text-xs font-medium",
              sectorConcentrationHigh
                ? "bg-amber-500/10 text-amber-400"
                : "bg-emerald-500/10 text-emerald-400",
            )}
          >
            {sectorConcentrationLabel}
          </span>
        </div>
        <div className="mt-2">
          <div className="mb-1 flex justify-between text-xs text-slate-500">
            <span>Drawdown</span>
            <span>{perf?.max_drawdown_pct?.toFixed(1) ?? "0"}% / -15%</span>
          </div>
          <div className="h-2 rounded-full bg-navy-700">
            <div
              className={clsx(
                "h-2 rounded-full",
                maxDd > -10 ? "bg-emerald-500" : maxDd > -13 ? "bg-amber-500" : "bg-rose-500",
              )}
              style={{ width: `${Math.min(100, (Math.abs(maxDd) / 15) * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Manage tab helpers ────────────────────────────────────────────

export function ReadOnlyField({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div>
      <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">{label}</label>
      <div className="rounded-md border border-navy-700 bg-navy-900/50 px-3 py-2 text-sm text-slate-300">
        {value}
      </div>
      {hint && <p className="mt-1 text-xs text-slate-600">{hint}</p>}
    </div>
  );
}

export type PaperNumKey =
  | "paper_max_positions"
  | "paper_max_per_sector"
  | "paper_max_daily_cost_usd"
  | "paper_default_stop_loss_pct"
  | "paper_screen_top_n"
  | "paper_analyze_top_n"
  | "paper_transaction_cost_pct"
  | "paper_daily_loss_limit_pct"
  | "paper_trailing_dd_limit_pct"
  | "paper_min_cash_reserve_pct";

export function PaperSettingNum({
  label,
  field,
  settings,
  dirty,
  setDirty,
  min,
  max,
  step,
  hint,
}: {
  label: string;
  field: PaperNumKey;
  settings: import("@/lib/types").FullSettings;
  dirty: Partial<import("@/lib/types").FullSettings>;
  setDirty: React.Dispatch<React.SetStateAction<Partial<import("@/lib/types").FullSettings>>>;
  min: number;
  max: number;
  step: number;
  hint?: string;
}) {
  const stored = settings[field];
  const draft = dirty[field];
  const value = (draft ?? stored ?? "") as number | string;
  return (
    <div>
      <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const raw = e.target.value;
          const next = raw === "" ? undefined : Number(raw);
          setDirty((d) => {
            const merged = { ...d };
            if (next === undefined || next === stored) {
              delete merged[field];
            } else {
              merged[field] = next;
            }
            return merged;
          });
        }}
        className="w-full rounded-md border border-navy-600 bg-navy-900 px-3 py-2 text-sm text-slate-100 focus:border-sky-500/50 focus:outline-none"
      />
      {hint && <p className="mt-1 text-xs text-slate-600">{hint}</p>}
      {draft !== undefined && (
        <p className="mt-1 text-[10px] uppercase tracking-wider text-amber-400">unsaved</p>
      )}
    </div>
  );
}
