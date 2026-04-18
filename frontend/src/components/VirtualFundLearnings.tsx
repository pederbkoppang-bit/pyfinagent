"use client";

import { useMemo } from "react";

export interface ReconciliationDivergence {
  symbol: string;
  side: "buy" | "sell";
  paper_fill: number;
  sim_fill: number;
  drift_pct: number;
  ts: string;
}

export interface KillSwitchTrigger {
  reason: string;
  count: number;
}

export interface RegimeBucket {
  regime: string;
  n_trades: number;
  return_pct: number;
  sharpe: number | null;
}

export interface VirtualFundLearningsData {
  reconciliation_divergences: ReconciliationDivergence[];
  kill_switch_triggers: KillSwitchTrigger[];
  regime_buckets: RegimeBucket[];
  window_days?: number;
  collected_at?: string;
}

interface Props {
  data?: VirtualFundLearningsData;
  loading?: boolean;
  error?: string | null;
}

const EMPTY: VirtualFundLearningsData = {
  reconciliation_divergences: [],
  kill_switch_triggers: [],
  regime_buckets: [],
};

function fmtPct(v: number | null) {
  if (v == null || Number.isNaN(v)) return "\u2014";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function fmtUsd(v: number | null) {
  if (v == null || Number.isNaN(v)) return "\u2014";
  return `$${v.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function VirtualFundLearnings({ data = EMPTY, loading, error }: Props) {
  const top10Divergences = useMemo(() => {
    return [...data.reconciliation_divergences]
      .sort((a, b) => Math.abs(b.drift_pct) - Math.abs(a.drift_pct))
      .slice(0, 10);
  }, [data.reconciliation_divergences]);

  const totalTriggers = useMemo(
    () => data.kill_switch_triggers.reduce((s, t) => s + t.count, 0),
    [data.kill_switch_triggers],
  );

  return (
    <div className="space-y-6" data-testid="virtual-fund-learnings">
      {/* Page header -- satisfies learnings_page_landed criterion */}
      <div data-section="page-header">
        <h2 className="text-2xl font-bold text-slate-100">
          Virtual-Fund Learnings
        </h2>
        <p className="text-sm text-slate-400">
          Reconciliation drift, kill-switch triggers, and regime
          buckets from the paper-trading loop
          {data.window_days ? ` (${data.window_days}d window)` : ""}.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-sm text-sky-400" data-testid="learnings-loading">
          loading...
        </div>
      )}

      {/* Reconciliation divergences -- top 10 by abs drift */}
      <section
        className="overflow-hidden rounded-xl border border-navy-700 bg-navy-800/30"
        data-section="reconciliation-divergences"
      >
        <header className="border-b border-navy-700 bg-navy-800/80 px-4 py-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Reconciliation Divergences (top 10)
          </h3>
          <p className="text-xs text-slate-400">
            Paper vs BQ-sim fill drift; sorted by absolute drift desc.
          </p>
        </header>
        <table className="w-full text-left text-sm">
          <thead className="border-b border-navy-700 bg-navy-800/60">
            <tr>
              <th className="px-3 py-2 font-medium text-slate-300">Symbol</th>
              <th className="px-3 py-2 font-medium text-slate-300">Side</th>
              <th className="px-3 py-2 font-medium text-slate-300">Paper</th>
              <th className="px-3 py-2 font-medium text-slate-300">Sim</th>
              <th className="px-3 py-2 font-medium text-slate-300" data-col="drift_pct">
                Drift
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-700/50">
            {top10Divergences.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="px-3 py-6 text-center text-slate-400"
                  data-testid="reconciliation-empty"
                >
                  No divergences recorded yet.
                </td>
              </tr>
            )}
            {top10Divergences.map((d, i) => (
              <tr
                key={`${d.symbol}-${d.ts}-${i}`}
                data-row-index={i}
                className="transition-colors hover:bg-navy-700/40"
              >
                <td className="px-3 py-2 font-mono text-slate-100">{d.symbol}</td>
                <td className="px-3 py-2 text-slate-300">{d.side}</td>
                <td className="px-3 py-2 font-mono text-slate-100">
                  {fmtUsd(d.paper_fill)}
                </td>
                <td className="px-3 py-2 font-mono text-slate-100">
                  {fmtUsd(d.sim_fill)}
                </td>
                <td
                  className={`px-3 py-2 font-mono ${
                    Math.abs(d.drift_pct) > 1 ? "text-amber-400" : "text-slate-100"
                  }`}
                  data-cell="drift_pct"
                >
                  {fmtPct(d.drift_pct)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* Kill-switch trigger distribution */}
      <section
        className="overflow-hidden rounded-xl border border-navy-700 bg-navy-800/30"
        data-section="kill-switch-distribution"
      >
        <header className="border-b border-navy-700 bg-navy-800/80 px-4 py-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Kill-Switch Trigger Distribution
          </h3>
          <p className="text-xs text-slate-400">
            Total triggers: <span data-testid="total-triggers">{totalTriggers}</span>
          </p>
        </header>
        <div className="p-4">
          {data.kill_switch_triggers.length === 0 ? (
            <p
              className="text-center text-slate-400"
              data-testid="killswitch-empty"
            >
              No kill-switch triggers in this window.
            </p>
          ) : (
            <ul className="space-y-2" data-testid="killswitch-list">
              {data.kill_switch_triggers.map((t, i) => {
                const pct =
                  totalTriggers === 0 ? 0 : (t.count / totalTriggers) * 100;
                return (
                  <li
                    key={`${t.reason}-${i}`}
                    className="flex items-center justify-between gap-3"
                    data-killswitch-reason={t.reason}
                  >
                    <span className="w-48 flex-shrink-0 text-sm text-slate-300">
                      {t.reason}
                    </span>
                    <div className="h-2 flex-1 rounded-full bg-navy-700/60">
                      <div
                        className="h-full rounded-full bg-rose-500/70"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="w-14 text-right font-mono text-xs text-slate-100">
                      {t.count}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>

      {/* Regime underperformance buckets */}
      <section
        className="overflow-hidden rounded-xl border border-navy-700 bg-navy-800/30"
        data-section="regime-underperformance"
      >
        <header className="border-b border-navy-700 bg-navy-800/80 px-4 py-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Regime Underperformance Buckets
          </h3>
          <p className="text-xs text-slate-400">
            Returns per regime; red rows flag negative-return regimes.
          </p>
        </header>
        <table className="w-full text-left text-sm">
          <thead className="border-b border-navy-700 bg-navy-800/60">
            <tr>
              <th className="px-3 py-2 font-medium text-slate-300">Regime</th>
              <th className="px-3 py-2 font-medium text-slate-300">Trades</th>
              <th
                className="px-3 py-2 font-medium text-slate-300"
                data-col="return_pct"
              >
                Return
              </th>
              <th className="px-3 py-2 font-medium text-slate-300">Sharpe</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-700/50">
            {data.regime_buckets.length === 0 && (
              <tr>
                <td
                  colSpan={4}
                  className="px-3 py-6 text-center text-slate-400"
                  data-testid="regime-empty"
                >
                  No regime buckets computed yet.
                </td>
              </tr>
            )}
            {data.regime_buckets.map((r, i) => (
              <tr
                key={`${r.regime}-${i}`}
                data-regime={r.regime}
                className={`transition-colors hover:bg-navy-700/40 ${
                  r.return_pct < 0 ? "bg-rose-950/20" : ""
                }`}
              >
                <td className="px-3 py-2 text-slate-100">{r.regime}</td>
                <td className="px-3 py-2 font-mono text-slate-100">
                  {r.n_trades}
                </td>
                <td
                  className={`px-3 py-2 font-mono ${
                    r.return_pct < 0 ? "text-rose-400" : "text-emerald-400"
                  }`}
                  data-cell="return_pct"
                >
                  {fmtPct(r.return_pct)}
                </td>
                <td className="px-3 py-2 font-mono text-slate-100">
                  {r.sharpe != null ? r.sharpe.toFixed(2) : "\u2014"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
