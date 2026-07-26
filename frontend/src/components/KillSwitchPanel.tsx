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
    // phase-36.7: a missing loss baseline means the breach legs CANNOT fire.
    // Optional so an older backend (no key) keeps rendering as today.
    armed?: boolean;
    daily_baseline_missing?: boolean;
    // phase-36.20: OPTIONAL, and that is deliberate -- an older backend that
    // predates phase-36.9/36.13 simply omits them, and the `=== true` / `!== true`
    // checks at the use sites then keep today's behaviour rather than flipping a
    // badge on `undefined`.
    daily_baseline_stale?: boolean;
    nav_invalid?: boolean;
    nav_invalid_disarmed?: boolean;
    baselines_present?: boolean;
    trailing_baseline_missing?: boolean;
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
  // phase-36.7: DISARMED is a first-class state, distinct from ACTIVE.
  //
  // Between the 2026-07-26 audit-log rotation and the phase-36.7 fix this panel
  // rendered an emerald "ACTIVE" badge next to "daily 0.00% / 4%  trail 0.00% /
  // 10%" while NO drawdown of any size could pause trading -- absence rendered
  // as the most reassuring possible reading. Same rule as phase-80.36 on the
  // cockpit: DISCRIMINATE ON PRESENCE, NEVER ON VALUE. `armed === false` is the
  // signal; `armed === undefined` (an older backend) must keep today's
  // behaviour, so the check is an explicit `=== false`, never `!breach.armed`.
  const disarmed = breach.armed === false;
  // phase-36.20: SPLIT THE ALARM FROM THE STATUS.
  //
  // phase-36.9 made `armed` mean "can this leg fire RIGHT NOW", which is false for
  // a daily anchor that is merely from yesterday. Reading `armed === false` alone
  // therefore painted the DISARMED alarm on a HEALTHY book every day from 00:00 UTC
  // until the first cycle rolled the anchor -- a recurring alarm for a condition
  // that repairs itself and needs no operator action. ISA-18.2 defines an alarm as
  // an abnormal condition REQUIRING A RESPONSE and prescribes reclassification
  // otherwise; the clinical literature (AHRQ) is what happens when you don't.
  //
  // `nav_invalid` MUST be excluded. `evaluate_breach`'s nav_invalid early return
  // passes `daily_baseline_stale` through while both *_missing stay false, and the
  // endpoint falls back to `... or 0.0` on a 5s BQ timeout -- so without this guard
  // a "we cannot measure the book at all" state would render as the friendly badge.
  //
  // Derived CLIENT-SIDE on purpose: `armed` stays a strict boolean because both
  // backend gates use `.get("armed", True)` and fail OPEN, so a third value there
  // would silently allow. Same `=== false` / `=== true` discipline as above so an
  // older backend without these keys keeps today's behaviour.
  const reanchoring =
    disarmed &&
    breach.daily_baseline_stale === true &&
    breach.daily_baseline_missing === false &&
    breach.trailing_baseline_missing === false &&
    breach.nav_invalid !== true &&
    breach.nav_invalid_disarmed !== true;
  // A leg that cannot be evaluated must never print a number (phase-36.7's rule,
  // extended to staleness): 0.00% is a legitimate healthy reading and the `0.0`
  // initialiser is indistinguishable from it.
  const dailyUnevaluable =
    breach.daily_baseline_missing === true || breach.daily_baseline_stale === true;
  // RE-ANCHORING is status, not alarm -- it must not tint the card or the icon.
  const alarm = paused || breach.any_breached || (disarmed && !reanchoring);

  return (
    <>
      <div
        className={clsx(
          "rounded-xl border p-3",
          reanchoring && !paused && !breach.any_breached
            ? // phase-36.20: SKY, and deliberately NOT amber -- amber is DISARMED's
              // token and this project's degraded colour. A self-clearing status must
              // not borrow the durable-fault tint.
              "border-sky-500/40 bg-sky-950/30"
            : disarmed && !paused && !breach.any_breached
              ? "border-amber-500/40 bg-amber-950/30"
              : alarm
                ? "border-rose-500/40 bg-rose-950/30"
                : "border-navy-700 bg-navy-800/60",
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
              paused
                ? "bg-rose-500/20 text-rose-300"
                : disarmed
                  ? "bg-amber-500/20 text-amber-300"
                  : "bg-emerald-500/20 text-emerald-300",
            )}
            title={
              reanchoring
                ? // phase-36.20: the baselines are INTACT; only the daily anchor is
                  // from an earlier day. This matches the server's own staleness 409
                  // (paper_trading.py), which says the same thing in the same words.
                  "RE-ANCHORING: the loss baselines are intact, but the daily anchor is from an earlier UTC day, so the daily-loss leg cannot be measured against today's open yet. The trailing leg is date-independent and still armed. No operator action is required -- the next cycle stamps today's anchor."
                : disarmed
                  ? // phase-36.12: the old text told the operator to wait for the next
                    // cycle to fix it -- but that cycle's silent anchor WAS the defect.
                    // It now blocks new orders and records the anchor instead.
                    "DISARMED: the loss baselines could not be restored, so neither breach leg can fire. The next cycle blocks new orders and writes an audited baseline_anchor_on_lost_history row instead of trading on unknown baselines."
                  : undefined
            }
          >
            {/* phase-36.20 + WCAG 1.4.1: the state is carried by TEXT, not by colour
                alone -- a colour-blind or monochrome operator reads the same word. */}
            {paused
              ? "PAUSED"
              : reanchoring
                ? "RE-ANCHORING"
                : disarmed
                  ? "DISARMED"
                  : "ACTIVE"}
          </span>
          {paused && state.pause_reason && (
            <span className="text-[10px] text-slate-500">({state.pause_reason})</span>
          )}
          <span className="ml-auto font-mono text-[10px] text-slate-500">
            {/* phase-36.7: an em-dash, never "0.00%", when the leg has no
                baseline -- 0.00% is a legitimate healthy reading and must not
                be shown for a leg that cannot be measured. */}
            daily{" "}
            {/* phase-36.20: branch on UNEVALUABLE (missing OR stale), not on missing
                alone. `daily_loss_pct` keeps its 0.0 initialiser whenever the leg is
                skipped, so a stale anchor was printing a fabricated "0.00%" for a leg
                that cannot fire -- erring REASSURING, the direction phase-36.7 exists
                to prevent. */}
            {dailyUnevaluable ? "—" : `${breach.daily_loss_pct.toFixed(2)}%`}{" "}
            / {state.thresholds.daily_loss_limit_pct}%
            {"   "}
            trail{" "}
            {breach.trailing_baseline_missing
              ? "—"
              : `${breach.trailing_dd_pct.toFixed(2)}%`}{" "}
            / {state.thresholds.trailing_dd_limit_pct}%
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
              // phase-36.7: mirrors the server-side 409 -- a disarmed switch
              // cannot verify "both limits read healthy", so resume is blocked.
              disabled={busy || breach.any_breached || disarmed}
              title={
                breach.any_breached
                  ? "Cannot resume while a limit is still breached"
                  : reanchoring
                    ? // phase-36.20: STILL DISABLED, and that is correct -- the server
                      // 409s on exactly this state (the staleness branch added in
                      // 36.13), so enabling the button would turn a misleading badge
                      // into a click that always fails. Only the REASON was wrong: it
                      // asserted the baselines were unrestorable, which the server's
                      // own 409 explicitly refutes.
                      "Cannot resume yet: the daily anchor is from an earlier UTC day, so the daily-loss leg cannot be verified against today's open. The baselines are intact and no operator action is required -- the next cycle stamps today's anchor, then resume will succeed."
                    : disarmed
                      ? "Cannot resume: kill switch DISARMED (loss baselines unrestorable). The next cycle blocks new orders and audits the anchor; restoring the true baselines is the fix, not resuming."
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
