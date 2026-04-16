"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { clsx } from "clsx";
import { IconWarning } from "@/lib/icons";
import { SkeletonCard } from "@/components/Skeleton";
import { getPaperKillSwitchState, postPaperKillSwitchAction } from "@/lib/api";

interface KillSwitchState {
  paused: boolean;
  pause_reason: string | null;
  sod_nav: number | null;
  peak_nav: number | null;
  current_nav: number;
  breach: {
    daily_loss_breached: boolean;
    daily_loss_pct: number;
    daily_loss_limit_pct: number;
    trailing_dd_breached: boolean;
    trailing_dd_pct: number;
    trailing_dd_limit_pct: number;
    any_breached: boolean;
  };
  thresholds: {
    daily_loss_limit_pct: number;
    trailing_dd_limit_pct: number;
  };
}

type Action = "PAUSE" | "RESUME" | "FLATTEN_ALL";

/**
 * Kill-switch panel: status banner + three action buttons (Pause, Resume,
 * Flatten-all). Every destructive action requires a single-modal confirmation.
 * Auto-flatten on limit breach is enforced server-side at cycle start; this
 * panel is for manual operator intervention.
 */
export function KillSwitchPanel() {
  const [state, setState] = useState<KillSwitchState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<Action | null>(null);
  const failRef = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const j = (await getPaperKillSwitchState()) as KillSwitchState;
      setState(j);
      setError(null);
      failRef.current = 0;
    } catch (e) {
      failRef.current += 1;
      if (failRef.current >= 5) setError(e instanceof Error ? e.message : "kill-switch failed");
    }
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const tick = () => {
      if (!document.hidden) void refresh();
    };
    void refresh();
    const id = window.setInterval(tick, 60_000);
    const onVis = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [refresh]);

  const handleConfirm = async () => {
    if (!pendingAction) return;
    setBusy(true);
    setError(null);
    try {
      await postPaperKillSwitchAction(pendingAction);
      setPendingAction(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (error && !state) {
    return (
      <div className="rounded-xl border border-rose-500/30 bg-rose-950/30 p-3">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Kill switch
        </p>
        <p className="mt-1 text-xs text-rose-300">Failed to load: {error}</p>
        <button
          type="button"
          onClick={() => {
            setError(null);
            void refresh();
          }}
          className="mt-2 rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Kill switch
        </p>
        <SkeletonCard h="h-14" className="mt-2" />
      </div>
    );
  }

  const paused = state.paused;
  const breach = state.breach;
  const alarm = paused || breach.any_breached;

  return (
    <>
      <div
        className={clsx(
          "rounded-xl border p-3",
          alarm ? "border-rose-500/40 bg-rose-950/30" : "border-navy-700 bg-navy-800/60",
        )}
      >
        <div className="flex items-center gap-2">
          <IconWarning
            size={16}
            weight={alarm ? "fill" : "regular"}
            className={alarm ? "text-rose-400" : "text-slate-500"}
          />
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Kill switch
          </p>
          <span
            className={clsx(
              "rounded-full px-2 py-0.5 text-[10px] font-semibold",
              paused ? "bg-rose-500/20 text-rose-300" : "bg-emerald-500/20 text-emerald-300",
            )}
          >
            {paused ? "PAUSED" : "ACTIVE"}
          </span>
          {paused && state.pause_reason && (
            <span className="text-[10px] text-slate-500">({state.pause_reason})</span>
          )}
          <span className="ml-auto font-mono text-[10px] text-slate-500">
            daily {breach.daily_loss_pct.toFixed(2)}% / {state.thresholds.daily_loss_limit_pct}%
            {"   "}
            trail {breach.trailing_dd_pct.toFixed(2)}% / {state.thresholds.trailing_dd_limit_pct}%
          </span>
        </div>

        <div className="mt-3 flex items-center gap-2">
          {!paused && (
            <button
              type="button"
              onClick={() => setPendingAction("PAUSE")}
              disabled={busy}
              className="rounded-md border border-amber-500/40 bg-amber-950/30 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-900/50 disabled:opacity-50"
            >
              Pause
            </button>
          )}
          {paused && (
            <button
              type="button"
              onClick={() => setPendingAction("RESUME")}
              disabled={busy || breach.any_breached}
              title={
                breach.any_breached
                  ? "Cannot resume while a limit is still breached"
                  : "Resume paper-trading cycle"
              }
              className="rounded-md border border-emerald-500/40 bg-emerald-950/30 px-3 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-900/50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Resume
            </button>
          )}
          <button
            type="button"
            onClick={() => setPendingAction("FLATTEN_ALL")}
            disabled={busy}
            className="rounded-md border border-rose-500/40 bg-rose-950/30 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-900/50 disabled:opacity-50"
          >
            Flatten all
          </button>
          {error && (
            <span className="ml-auto text-[11px] text-rose-300">{error}</span>
          )}
        </div>
      </div>

      {pendingAction && (
        <ConfirmModal
          action={pendingAction}
          busy={busy}
          onCancel={() => setPendingAction(null)}
          onConfirm={handleConfirm}
        />
      )}
    </>
  );
}

function ConfirmModal({
  action,
  busy,
  onCancel,
  onConfirm,
}: {
  action: Action;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const label =
    action === "FLATTEN_ALL"
      ? "Flatten all open positions and pause trading?"
      : action === "PAUSE"
      ? "Pause new-order generation? Existing positions stay open."
      : "Resume new-order generation?";
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      role="dialog"
      aria-modal="true"
      onClick={onCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-w-md rounded-xl border border-navy-700 bg-navy-900 p-5"
      >
        <p className="text-sm font-semibold text-slate-100">{label}</p>
        <p className="mt-2 text-xs text-slate-500">
          This action will be recorded in the kill-switch audit log.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-md px-3 py-1.5 text-xs text-slate-300 hover:bg-navy-800 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className={clsx(
              "rounded-md px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50",
              action === "FLATTEN_ALL" ? "bg-rose-600 hover:bg-rose-500" : "bg-sky-600 hover:bg-sky-500",
            )}
          >
            {busy ? "Working..." : `Confirm ${action}`}
          </button>
        </div>
      </div>
    </div>
  );
}
