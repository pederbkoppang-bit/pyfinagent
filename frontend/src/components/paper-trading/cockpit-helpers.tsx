"use client";

// phase-44.2 -- cockpit helper components hoisted out of the old
// /paper-trading monolith. Pure presentational; consume props.

import { useEffect, useState } from "react";
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
// number-flow-react[data-pyfa-trend="up"]::part(digit) etc.
// (Cycle 77 bugfix: lib's React wrapper renders <number-flow-react>,
// NOT <number-flow> -- cycle 76 had the wrong element name in the CSS.)
import { useTrend } from "@/lib/use-trend";
// goal-multimarket-ux: currency/market metadata (pure module). Dollar is
// parameterized by currency; MarketChip renders the per-row market.
import {
  MARKET_BENCHMARK_LABEL,
  MARKET_DOT_CLASS,
  MARKET_EXCHANGE,
  MARKET_EXCHANGE_SHORT,
  numberFlowFormat,
  numberFlowLocale,
  positionMarketValueUsd,
  resolveMarket,
} from "@/lib/format";
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
      transformTiming={{ duration: 900 }}
      willChange
      aria-live="off"
      data-pyfa-trend={trend}
      className={colorClass}
    />
  );
}

// goal-multimarket-ux: `currency` defaults to USD. When USD, the format object is
// the EXACT legacy one (minimumFractionDigits:2, default locales) so every existing
// USD call site is byte-identical. Non-USD uses locale-correct Intl (KRW => 0 dp).
// Use this for LOCAL per-share values (pass the row's currency); leave the default
// for USD VALUE/NAV columns.
export function Dollar({
  value,
  currency = "USD",
}: {
  value: number | null | undefined;
  currency?: string;
}) {
  const trend = useTrend(value);
  if (value == null) return <span className="text-slate-500">—</span>;
  const cur = (currency || "USD").toUpperCase();
  const isUsd = cur === "USD";
  return (
    <NumberFlow
      value={value}
      format={
        isUsd
          ? {
              style: "currency",
              currency: "USD",
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            }
          : numberFlowFormat(cur)
      }
      locales={isUsd ? undefined : numberFlowLocale(cur)}
      transformTiming={{ duration: 900 }}
      willChange
      aria-live="off"
      data-pyfa-trend={trend}
      className="text-slate-100"
    />
  );
}

// goal-multimarket-ux: per-row market indicator. Colored dot (static JIT-safe class)
// + market code. NO flag emoji; the code conveys the market so color is not the only
// signal (WCAG). Shared by the positions + trades tables.
export function MarketChip({
  market,
  ticker,
  showExchange = false,
}: {
  market?: string | null;
  ticker?: string | null;
  // When true, append the compact exchange tag (e.g. "EU · XETRA"). The full
  // exchange name is always available via the title tooltip.
  showExchange?: boolean;
}) {
  const m = resolveMarket({ market, ticker });
  const dot = MARKET_DOT_CLASS[m] ?? "bg-slate-400";
  const exchange = MARKET_EXCHANGE[m] ?? "";
  const exShort = MARKET_EXCHANGE_SHORT[m] ?? "";
  return (
    <span
      className="inline-flex items-center gap-1.5 font-mono text-xs text-slate-300"
      title={exchange || undefined}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full", dot)} aria-hidden="true" />
      <span>{m}</span>
      {showExchange && exShort && (
        <span className="text-[10px] font-normal text-slate-500">· {exShort}</span>
      )}
    </span>
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
  positions = [],
  activeMarket = "ALL",
}: {
  status: PaperTradingStatus | null;
  perf: PaperPerformance | null;
  liveNav: number | null;
  liveTotalPnlPct: number | null;
  // goal-multimarket-ux: market filter context. `positions` is the full set; the
  // hero filters locally for the Positions count + per-market return.
  positions?: PaperPosition[];
  activeMarket?: string;
}) {
  const navDisplay = liveNav ?? status?.portfolio.nav ?? null;
  const pnlDisplay = liveTotalPnlPct ?? status?.portfolio.pnl_pct ?? null;
  const bench = status?.portfolio.benchmark_return_pct ?? 0;

  const isAll = !activeMarket || activeMarket === "ALL";
  const filtered = isAll
    ? positions
    : positions.filter(
        (p) => resolveMarket({ market: p.market, ticker: p.ticker }) === activeMarket,
      );
  const positionCount = isAll ? (status?.position_count ?? 0) : filtered.length;

  // phase-56.1 (55.1 F-12): the card VALUE for a non-US market is that market's
  // holdings return, NOT an index excess (the per-market index is not fetched) --
  // so the LABEL must say so. "vs KOSPI +0.00%" was honest-tooltip'd but
  // misleading at a glance; per the 55.1 verdict we strengthen the disclosure:
  // label per-market cards "<MKT> holdings" until a true ^KS11/^GDAXI excess
  // exists (phase-57-adjacent feature). ALL/US keep the true "vs SPY" excess.
  const benchLabel =
    isAll || activeMarket === "US"
      ? "vs SPY"
      : `${activeMarket} holdings`;

  // Benchmark VALUE. ALL/US: fund P&L minus the (vs-SPY) benchmark return. A specific
  // non-US market: the per-market index return is NOT exposed by the API, so we show
  // that market's holdings return (USD-consistent: sum unrealized_pnl / sum cost_basis,
  // both USD) rather than invent an FX-converted excess. Tooltip makes that explicit.
  let vsValue: number | null;
  let vsTitle: string | undefined;
  if (isAll || activeMarket === "US") {
    vsValue = (pnlDisplay ?? 0) - bench;
  } else {
    let pnl = 0;
    let cost = 0;
    for (const p of filtered) {
      pnl += p.unrealized_pnl ?? 0;
      cost += p.cost_basis ?? 0;
    }
    vsValue = cost > 0 ? (pnl / cost) * 100 : null;
    vsTitle = `${activeMarket} holdings return (USD). Per-market ${
      MARKET_BENCHMARK_LABEL[activeMarket] ?? "benchmark"
    } excess is not yet exposed by the API.`;
  }

  return (
    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <MetricCard label="NAV"><Dollar value={navDisplay} /></MetricCard>
      <MetricCard label="Cash"><Dollar value={status?.portfolio.cash} /></MetricCard>
      <MetricCard label="Total P&L"><PnlBadge value={pnlDisplay} /></MetricCard>
      <MetricCard label={benchLabel}>
        <span title={vsTitle}>
          <PnlBadge value={vsValue} />
        </span>
      </MetricCard>
      <MetricCard label="Sharpe"><SharpeValue value={perf?.sharpe_ratio} /></MetricCard>
      <MetricCard label="Positions">
        <span className="text-slate-100">{positionCount}</span>
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
  // phase-56.1 (55.1 F-1): qty x local current_price treated KRW as USD
  // ("Max position 1527.8%"); use the shared FX-safe USD value instead.
  const concentrations = positions.map(
    (p) => (positionMarketValueUsd(p) / navDenom) * 100,
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
  onValidity,
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
  // phase-70.1: lets the parent (manage/page) disable Save + show a summary
  // when any field is out of range. Optional -- fields without it still show
  // their own inline error and never persist an invalid value into `dirty`.
  onValidity?: (field: PaperNumKey, error: string | undefined) => void;
}) {
  const stored = settings[field];
  // phase-70.1: string-state-then-coerce (React docs + MDN nullish + web.dev
  // constraint validation). The old `value={draft ?? stored ?? ""}` binding
  // made an EMPTY field unrepresentable: clearing it -> onChange next=undefined
  // -> the draft was deleted -> `??` fell back through to `stored`, so the box
  // snapped back to the stored digits and the next keystroke APPENDED
  // (2 -> "25"), which then failed the save-time bound check as a generic 422.
  // Now the input is bound to a defined string that CAN be "", so clear-then-
  // type yields exactly the typed value, and we coerce to a number only for
  // the dirty/save path.
  const [text, setText] = useState<string>(
    dirty[field] !== undefined ? String(dirty[field]) : stored != null ? String(stored) : "",
  );
  // Re-seed from the stored value after a save (dirty cleared) or an external
  // settings reload -- but never clobber an in-progress edit (draft present).
  useEffect(() => {
    if (dirty[field] === undefined) {
      setText(stored != null ? String(stored) : "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stored, field]);

  const trimmed = text.trim();
  const num = trimmed === "" ? undefined : Number(trimmed);
  const error =
    trimmed !== "" && (num === undefined || Number.isNaN(num))
      ? "Enter a number."
      : num !== undefined && (num < min || num > max)
        ? `Must be between ${min} and ${max}.`
        : undefined;

  useEffect(() => {
    onValidity?.(field, error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [error]);

  return (
    <div>
      <label className="mb-1 block text-xs uppercase tracking-wider text-slate-500">{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={text}
        aria-invalid={error ? true : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          const t = raw.trim();
          const n = t === "" ? undefined : Number(t);
          const invalid = n === undefined || Number.isNaN(n) || n < min || n > max;
          setDirty((d) => {
            const merged = { ...d };
            // Never persist an empty / non-numeric / out-of-range value into
            // dirty -- so the PUT /api/settings/ payload is always valid and a
            // value the UI rejected can never produce a silent 422.
            if (invalid || n === stored) {
              delete merged[field];
            } else {
              merged[field] = n;
            }
            return merged;
          });
        }}
        className={clsx(
          "w-full rounded-md border bg-navy-900 px-3 py-2 text-sm text-slate-100 focus:outline-none",
          error
            ? "border-rose-500/60 focus:border-rose-500/60"
            : "border-navy-600 focus:border-sky-500/50",
        )}
      />
      {error ? (
        <p className="mt-1 text-xs text-rose-400">{error}</p>
      ) : (
        hint && <p className="mt-1 text-xs text-slate-600">{hint}</p>
      )}
      {dirty[field] !== undefined && !error && (
        <p className="mt-1 text-[10px] uppercase tracking-wider text-amber-400">unsaved</p>
      )}
    </div>
  );
}

// phase-50.6: live-loop markets multi-select (US/EU/KR). Native fieldset/legend +
// checkboxes (W3C APG; native over bespoke ARIA). Mirrors PaperSettingNum's
// settings/dirty/setDirty contract; writes paper_markets (list -> CSV via
// settings_api). At least one market is required (the last checked box is
// disabled) so the loop never resolves to an empty universe.
const _MARKET_OPTS = ["US", "EU", "KR"];

export function PaperMarketsField({
  settings,
  dirty,
  setDirty,
}: {
  settings: import("@/lib/types").FullSettings;
  dirty: Partial<import("@/lib/types").FullSettings>;
  setDirty: React.Dispatch<React.SetStateAction<Partial<import("@/lib/types").FullSettings>>>;
}) {
  const stored = settings.paper_markets ?? ["US"];
  const cur = dirty.paper_markets ?? stored;
  const sameSet = (a: string[], b: string[]) =>
    a.length === b.length && [...a].sort().join() === [...b].sort().join();
  const toggle = (m: string, checked: boolean) => {
    const next = _MARKET_OPTS.filter((x) => (x === m ? checked : cur.includes(x)));
    const safe = next.length ? next : ["US"]; // never empty -> backend default
    setDirty((d) => {
      const merged = { ...d };
      if (sameSet(safe, stored)) delete merged.paper_markets;
      else merged.paper_markets = safe;
      return merged;
    });
  };
  return (
    <fieldset className="md:col-span-2 rounded-lg border border-navy-700 bg-navy-800/40 p-3">
      <legend className="px-1 text-xs uppercase tracking-wider text-slate-500">Live-loop markets</legend>
      <div className="flex flex-wrap gap-4">
        {_MARKET_OPTS.map((m) => {
          const checked = cur.includes(m);
          const only = checked && cur.length === 1;
          return (
            <label
              key={m}
              className="flex cursor-pointer items-center gap-2 text-sm text-slate-200"
              title={MARKET_EXCHANGE[m] ?? m}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={only}
                onChange={(e) => toggle(m, e.target.checked)}
                className="h-4 w-4 cursor-pointer rounded border-navy-600 bg-navy-900 text-sky-500 focus:ring-2 focus:ring-sky-500/50 disabled:opacity-50"
              />
              <span className="font-mono">{m}</span>
            </label>
          );
        })}
      </div>
      <p className="mt-2 text-xs text-slate-600">
        Markets the live paper loop screens/trades (subset of US/EU/KR). Default US only;
        at least one required. International markets trade only after the data-quality gate.
      </p>
      {dirty.paper_markets !== undefined && (
        <p className="mt-1 text-[10px] uppercase tracking-wider text-amber-400">unsaved</p>
      )}
    </fieldset>
  );
}
