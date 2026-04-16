"use client";

import { clsx } from "clsx";
import { IconCheckCircle, IconInfo, IconXCircle } from "@/lib/icons";
import { SkeletonCard } from "@/components/Skeleton";

export interface GoLiveGate {
  booleans: {
    trades_ge_100: boolean;
    psr_ge_95_sustained_30d: boolean;
    dsr_ge_95: boolean;
    sr_gap_le_30pct: boolean;
    max_dd_within_tolerance: boolean;
  };
  promote_eligible: boolean;
  details: {
    n_round_trips: number;
    psr: number | null;
    dsr: number | null;
    rolling_sharpe: number | null;
    n_obs: number;
    latest_reconciliation_divergence_pct: number;
    realized_max_dd_pct: number;
  };
  thresholds: {
    trades: number;
    psr_sustained_days: number;
    psr: number;
    dsr: number;
    sr_gap: number;
    max_dd_pct: number;
  };
  computed_at: string;
}

interface Props {
  gate: GoLiveGate | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onPromote?: () => void;
}

function Check({ ok }: { ok: boolean }) {
  return ok ? (
    <IconCheckCircle size={16} weight="fill" className="text-emerald-400" />
  ) : (
    <IconXCircle size={16} weight="fill" className="text-rose-400" />
  );
}

export function GoLiveGateWidget({
  gate,
  loading = false,
  error,
  onRetry,
  onPromote,
}: Props) {
  if (error && !gate) {
    return (
      <div className="rounded-xl border border-rose-500/30 bg-rose-950/30 p-3">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Go-Live Gate
        </p>
        <p className="mt-1 text-xs text-rose-300">Failed to load: {error}</p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  if (loading || !gate) {
    return (
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-4">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Go-Live Gate
        </p>
        <SkeletonCard h="h-16" className="mt-2" />
      </div>
    );
  }

  const { booleans: b, promote_eligible, details: d, thresholds: t } = gate;

  const items: Array<{ key: string; ok: boolean; label: string; hint: string }> = [
    {
      key: "trades",
      ok: b.trades_ge_100,
      label: `>=${t.trades} trades`,
      hint: `${d.n_round_trips} closed round trips`,
    },
    {
      key: "psr",
      ok: b.psr_ge_95_sustained_30d,
      label: `PSR >= ${t.psr.toFixed(2)} (${t.psr_sustained_days}d)`,
      hint: d.psr != null ? `${d.psr.toFixed(3)}, n_obs=${d.n_obs}` : "insufficient data",
    },
    {
      key: "dsr",
      ok: b.dsr_ge_95,
      label: `DSR >= ${t.dsr.toFixed(2)}`,
      hint: d.dsr != null ? d.dsr.toFixed(3) : "n/a",
    },
    {
      key: "sr_gap",
      ok: b.sr_gap_le_30pct,
      label: `Reality gap <= ${(t.sr_gap * 100).toFixed(0)}%`,
      hint: `${d.latest_reconciliation_divergence_pct.toFixed(2)}%`,
    },
    {
      key: "dd",
      ok: b.max_dd_within_tolerance,
      label: `Max DD <= ${t.max_dd_pct.toFixed(0)}%`,
      hint: `${d.realized_max_dd_pct.toFixed(2)}%`,
    },
  ];

  return (
    <div
      className={clsx(
        "rounded-xl border p-4 transition-colors",
        promote_eligible
          ? "border-emerald-500/40 bg-emerald-950/30"
          : "border-navy-700 bg-navy-800/60",
      )}
    >
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Go-Live Gate
        </p>
        <span
          className={clsx(
            "rounded-full px-2 py-0.5 text-[10px] font-semibold",
            promote_eligible
              ? "bg-emerald-500/20 text-emerald-300"
              : "bg-rose-500/20 text-rose-300",
          )}
        >
          {promote_eligible ? "ELIGIBLE" : "NOT ELIGIBLE"}
        </span>
        <span className="ml-auto flex items-center gap-1 text-[10px] text-slate-500">
          <IconInfo size={12} /> 5 deterministic checks
        </span>
      </div>

      <ul className="mt-3 space-y-1.5">
        {items.map((it) => (
          <li
            key={it.key}
            className={clsx(
              "flex items-center gap-2 rounded-md border px-2 py-1.5",
              it.ok ? "border-emerald-500/20" : "border-rose-500/20",
            )}
          >
            <Check ok={it.ok} />
            <span className="text-xs font-medium text-slate-200">{it.label}</span>
            <span className="ml-auto truncate font-mono text-[10px] text-slate-500">
              {it.hint}
            </span>
          </li>
        ))}
      </ul>

      <div className="mt-3 flex items-center justify-end">
        <button
          type="button"
          disabled={!promote_eligible}
          onClick={onPromote}
          className={clsx(
            "rounded-md px-4 py-2 text-sm font-medium transition-colors",
            promote_eligible
              ? "bg-emerald-500 text-white hover:bg-emerald-600"
              : "cursor-not-allowed bg-navy-800 text-slate-500",
          )}
          title={
            promote_eligible
              ? "Promote paper strategy to live capital"
              : "All gate checks must be green before promotion"
          }
        >
          Promote to live
        </button>
      </div>
    </div>
  );
}
