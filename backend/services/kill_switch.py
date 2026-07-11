"""
kill_switch -- Production-grade pause / resume / flatten-all for paper trading.

Defaults derived from prop-trading practitioner consensus (see RESEARCH.md
Phase 4.5 step 4.5.7):

  - daily_loss_limit_pct = 4% of start-of-day NAV (modal across FTMO 5%,
    FXIFY 4%, Alpha Capital 4%, FundedNext 4%; Van Tharp 2%/trade x 2 = 4%).
  - trailing_dd_limit_pct = 10% EOD from rolling high-water mark (upper-end
    standard for long-only unlevered equity; Maven/Audacity/PropVator).

Breach semantics (FINRA Rule 15c3-5 "hard block" pattern):
  - Pause = halt new entries; existing positions kept.
  - Flatten = close every open position at market; cancel pending orders.
  - Limit breach => auto-flatten + auto-pause + audit log; explicit human
    resume required once both limits read healthy.

Audit trail (mandatory per 3forge / ESMA Supervisory Briefing 2026):
  Every state transition appends a JSON line to handoff/kill_switch_audit.jsonl
  with {timestamp, event, trigger, details}.
"""

from __future__ import annotations

import json

from backend.utils import json_io
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)


class KillSwitchState:
    """Module-level thread-safe state. Persisted across process restarts via
    the audit log: the most recent `pause` or `resume` line sets the resume
    state; if it's `pause` the system re-enters paused on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._pause_reason: Optional[str] = None
        self._sod_nav: Optional[float] = None  # start-of-day NAV snapshot
        self._sod_date: Optional[str] = None  # phase-23.2.19: UTC date of the SOD anchor
        self._peak_nav: Optional[float] = None  # trailing high-water mark
        # phase-38.1 (OPEN-10): auto-resume hysteresis. _paused_at carries
        # the UTC ISO timestamp of the most recent `pause` event; cleared
        # on resume. Persisted via audit log so restart-survivable.
        self._paused_at: Optional[str] = None
        # _auto_resume_alerted_at carries the timestamp of the T+1h pager
        # alert (one-shot per pause-cycle) so we don't spam Slack.
        self._auto_resume_alerted_at: Optional[str] = None
        self._load_from_audit()

    def _load_from_audit(self) -> None:
        if not _AUDIT_PATH.exists():
            return
        try:
            with _AUDIT_PATH.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json_io.parse_json_line(line)
                    except Exception:
                        continue
                    if row.get("event") == "pause":
                        self._paused = True
                        self._pause_reason = row.get("trigger")
                        # phase-38.1: capture the pause ts for hysteresis.
                        self._paused_at = row.get("ts")
                        self._auto_resume_alerted_at = None
                    elif row.get("event") == "resume":
                        self._paused = False
                        self._pause_reason = None
                        self._paused_at = None
                        self._auto_resume_alerted_at = None
                    elif row.get("event") == "auto_resume_alert":
                        # phase-38.1: T+1h pager alert went out -- record so
                        # we don't re-fire on the next cycle within the same
                        # pause window.
                        self._auto_resume_alerted_at = row.get("ts")
                    elif row.get("event") == "sod_snapshot":
                        self._sod_nav = float(row.get("nav") or 0.0) or None
                        # phase-23.2.19: prefer explicit `date` (rows written
                        # post-fix); fall back to parsing `ts` for legacy
                        # rows that pre-date the schema bump.
                        sod_date = row.get("date")
                        if not sod_date:
                            ts = row.get("ts")
                            if ts:
                                try:
                                    sod_date = datetime.fromisoformat(
                                        str(ts).replace("Z", "+00:00")
                                    ).astimezone(timezone.utc).date().isoformat()
                                except Exception:
                                    sod_date = None
                        self._sod_date = sod_date
                    elif row.get("event") == "peak_update":
                        self._peak_nav = float(row.get("nav") or 0.0) or None
                    elif row.get("event") == "peak_reset":
                        # phase-69.1 (audit item 2): audited peak reset (flatten /
                        # operator-resume). Restart-replayable + idempotent -- the
                        # reset value wins over prior peak_update rows in stream order.
                        self._peak_nav = float(row.get("new_peak") or 0.0) or None
        except Exception as e:
            logger.warning(f"kill_switch: audit load failed: {e}")

    @staticmethod
    def _append_audit(event: str, **fields: Any) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            with _AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"kill_switch: audit write failed: {e}")

    # ── State getters ──────────────────────────────────────────────
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def _snapshot_locked(self) -> dict:
        """phase-23.1.22: lock-free snapshot helper. Caller MUST already hold
        self._lock. Used by pause()/resume() which re-entered the same
        threading.Lock() via snapshot() and deadlocked the entire process.
        Found via faulthandler SIGUSR1 dump on a live hung backend.

        phase-23.2.19: includes sod_date so callers can decide whether to
        re-anchor SOD on a new UTC calendar day.

        phase-38.1: includes paused_at + auto_resume_alerted_at for
        hysteresis logic in check_auto_resume."""
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "sod_nav": self._sod_nav,
            "sod_date": self._sod_date,
            "peak_nav": self._peak_nav,
            "paused_at": self._paused_at,
            "auto_resume_alerted_at": self._auto_resume_alerted_at,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    # ── State transitions ──────────────────────────────────────────
    def pause(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = True
            self._pause_reason = trigger
            # phase-38.1: stamp the pause timestamp for hysteresis.
            self._paused_at = datetime.now(timezone.utc).isoformat()
            self._auto_resume_alerted_at = None
            self._append_audit("pause", trigger=trigger, details=details or {})
            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
            # re-acquiring self._lock (threading.Lock is not reentrant).
            snap = self._snapshot_locked()
        # phase-23.2.18: operator alert on auto-pause. Manual / test / bench
        # triggers stay silent. Outside the lock so the webhook path can
        # block briefly without holding kill-switch state.
        _MANUAL_TRIGGERS = {"manual", "test", "test-pre", "bench-1", "bench-2", "bench-3"}
        if trigger not in _MANUAL_TRIGGERS:
            try:
                from backend.services.observability.alerting import raise_cron_alert_sync
                raise_cron_alert_sync(
                    source="kill_switch",
                    error_type=f"auto_pause_{trigger}",
                    severity="P1",
                    title=f"Kill-switch AUTO-PAUSED trading (trigger={trigger})",
                    details={
                        "trigger": trigger,
                        **{str(k): str(v) for k, v in (details or {}).items()},
                    },
                )
            except Exception as _alert_err:
                logger.warning(f"kill_switch pause-alert dispatch failed: {_alert_err}")
        return snap

    def resume(self, trigger: str = "manual", details: Optional[dict] = None,
               nav: Optional[float] = None) -> dict:
        with self._lock:
            self._paused = False
            self._pause_reason = None
            # phase-38.1: clear pause-cycle state on resume.
            self._paused_at = None
            self._auto_resume_alerted_at = None
            self._append_audit("resume", trigger=trigger, details=details or {})
            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
            # re-acquiring self._lock (threading.Lock is not reentrant).
            snap = self._snapshot_locked()
        # phase-69.1 (audit item 2): re-anchor the trailing peak to the current
        # NAV on an operator resume, so a monotonic peak can no longer keep the
        # trailing-DD breach alive forever after a flatten (the resume endpoint
        # passes `nav`). DARK: reset_peak is a no-op until KS-PEAK-RESET: APPROVED.
        # Called OUTSIDE the with-block -- reset_peak takes self._lock, which is
        # non-reentrant. If it fired, prefer its post-reset snapshot.
        if nav is not None and nav > 0:
            rp = self.reset_peak(float(nav), trigger=f"resume:{trigger}",
                                 operator=(details or {}).get("operator"))
            if rp is not None:
                snap = rp
        return snap

    def update_sod_nav(self, nav: float, date: Optional[str] = None) -> None:
        """Record start-of-day NAV for daily-loss calculation.

        phase-23.2.19: now stamps the UTC `date` alongside `nav`. Caller
        passes today's UTC ISO date (`datetime.now(timezone.utc).date().isoformat()`)
        when re-anchoring on a new calendar day; default None falls back
        to today. The audit row gets both `nav` and `date` so a future
        boot replay can detect daily-roll boundaries without parsing `ts`.
        """
        if date is None:
            date = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            self._sod_nav = float(nav)
            self._sod_date = date
            self._append_audit("sod_snapshot", nav=self._sod_nav, date=self._sod_date)

    def update_peak(self, nav: float) -> None:
        """Ratchet the trailing high-water mark upward. Never moves down."""
        with self._lock:
            if self._peak_nav is None or nav > self._peak_nav:
                self._peak_nav = float(nav)
                self._append_audit("peak_update", nav=self._peak_nav)

    def reset_peak(self, new_peak: float, trigger: str,
                   operator: Optional[str] = None) -> Optional[dict]:
        """phase-69.1 (audit item 2): audited, restart-replayable reset of the
        trailing high-water mark.

        The monotonic `update_peak` never moves down and nothing ever reset it,
        so after a >=10% pullback that flattens the book to 100% cash the
        trailing-DD breach persists forever and BOTH resume paths refuse -- the
        engine stops earning permanently. `reset_peak` re-anchors the peak on a
        flatten-to-cash or an operator resume (callers pass the current/post-flatten
        NAV), emitting a `peak_reset` audit row that `_load_from_audit` replays, so
        the reset survives a restart and is idempotent. Thresholds (4/10/8/30) are
        byte-untouched -- this restores the documented "human resume once healthy".

        DARK by default: a no-op returning None unless the operator has recorded the
        token via `settings.kill_switch_peak_reset_enabled=True` (KS-PEAK-RESET:
        APPROVED). This is a guard-behavior change, so it must not fire until then.
        """
        try:
            from backend.config.settings import get_settings
            enabled = bool(getattr(get_settings(), "kill_switch_peak_reset_enabled", False))
        except Exception:
            enabled = False
        if not enabled:
            return None  # DARK -- no peak reset until KS-PEAK-RESET: APPROVED
        with self._lock:
            old_peak = self._peak_nav
            self._peak_nav = float(new_peak)
            self._append_audit("peak_reset", old_peak=old_peak,
                               new_peak=self._peak_nav, trigger=trigger,
                               operator=operator)
            return self._snapshot_locked()


_state = KillSwitchState()


def get_state() -> KillSwitchState:
    return _state


# ── Breach evaluation ──────────────────────────────────────────────


def evaluate_breach(
    current_nav: float,
    daily_loss_limit_pct: float,
    trailing_dd_limit_pct: float,
) -> dict:
    """
    Check both limits against the current NAV. Returns a dict with booleans
    and diagnostic context. Does NOT flip state -- callers (see PaperTrader
    below) decide whether to flatten+pause based on the returned flags.
    """
    # phase-69.1 (audit item 2, kill_switch:246): guard against invalid NAV.
    # A caller's BQ-timeout `or 0.0` fallback yields current_nav<=0, which the
    # (sod - current_nav)/sod math would render as a phantom 100% daily +
    # trailing breach -- flattening the whole book on a transient 5s timeout.
    # A funded paper book's NAV is never <=0, so treat it as no-data (fail-safe:
    # no breach) rather than a real breach. Thresholds are untouched.
    if current_nav is None or current_nav <= 0:
        return {
            "daily_loss_breached": False, "daily_loss_pct": 0.0,
            "daily_loss_limit_pct": float(daily_loss_limit_pct),
            "trailing_dd_breached": False, "trailing_dd_pct": 0.0,
            "trailing_dd_limit_pct": float(trailing_dd_limit_pct),
            "any_breached": False, "nav_invalid": True,
        }
    s = _state.snapshot()
    sod = s.get("sod_nav")
    peak = s.get("peak_nav")

    daily_loss_breached = False
    daily_loss_pct = 0.0
    if sod and sod > 0:
        daily_loss_pct = (sod - current_nav) / sod * 100.0
        daily_loss_breached = daily_loss_pct >= daily_loss_limit_pct

    trailing_dd_breached = False
    trailing_dd_pct = 0.0
    if peak and peak > 0:
        trailing_dd_pct = (peak - current_nav) / peak * 100.0
        trailing_dd_breached = trailing_dd_pct >= trailing_dd_limit_pct

    return {
        "daily_loss_breached": bool(daily_loss_breached),
        "daily_loss_pct": round(daily_loss_pct, 4),
        "daily_loss_limit_pct": float(daily_loss_limit_pct),
        "trailing_dd_breached": bool(trailing_dd_breached),
        "trailing_dd_pct": round(trailing_dd_pct, 4),
        "trailing_dd_limit_pct": float(trailing_dd_limit_pct),
        "any_breached": bool(daily_loss_breached or trailing_dd_breached),
    }


# phase-38.1 (OPEN-10): kill-switch auto-resume hysteresis.
# Operator-driven resumes created two 3.5h outage windows in 5 days
# (OPS-F10). Auto-resume after 2h of no-breach closes that gap.

AUTO_RESUME_ALERT_AT_SEC: float = 60 * 60       # T+1h: pager alert
AUTO_RESUME_TRIGGER_AT_SEC: float = 2 * 60 * 60  # T+2h: auto-resume fires


def check_auto_resume(
    current_nav: float,
    daily_loss_limit_pct: float,
    trailing_dd_limit_pct: float,
    enabled: bool = False,
) -> dict:
    # phase-38.1 (OPEN-10): evaluate hysteresis. Default-OFF; caller
    # passes `enabled=True` after operator opts in via the
    # `kill_switch_auto_resume_enabled` settings flag.
    #
    # Returns dict with:
    #   action: "no_op" | "alert" | "resume"
    #   reason: human-readable explanation
    #   seconds_paused: time since pause (or None if not paused)
    #
    # No state mutation here except via state.resume() on action="resume"
    # and audit-log append on action="alert". This keeps the function
    # ergonomic to call once per cycle.
    sentinel = {
        "action": "no_op", "reason": "auto_resume_disabled" if not enabled else "not_paused",
        "seconds_paused": None,
    }
    if not enabled:
        return sentinel
    s = _state.snapshot()
    if not s.get("paused"):
        return sentinel
    paused_at_str = s.get("paused_at")
    if not paused_at_str:
        return {"action": "no_op", "reason": "no_paused_at_timestamp", "seconds_paused": None}
    try:
        paused_at = datetime.fromisoformat(str(paused_at_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return {"action": "no_op", "reason": "no_paused_at_timestamp", "seconds_paused": None}
    now = datetime.now(timezone.utc)
    seconds_paused = (now - paused_at).total_seconds()

    # If the breach is STILL active, never auto-resume.
    breach = evaluate_breach(current_nav, daily_loss_limit_pct, trailing_dd_limit_pct)
    if breach["any_breached"]:
        return {
            "action": "no_op",
            "reason": "breach_still_active",
            "seconds_paused": round(seconds_paused, 1),
            "breach": breach,
        }

    # T+2h: auto-resume fires.
    if seconds_paused >= AUTO_RESUME_TRIGGER_AT_SEC:
        _state.resume(trigger="auto_resume_hysteresis", details={
            "seconds_paused": round(seconds_paused, 1),
            "current_nav": current_nav,
            "breach": breach,
        })
        return {
            "action": "resume",
            "reason": "no_breach_for_2h",
            "seconds_paused": round(seconds_paused, 1),
        }

    # T+1h: pager alert (one-shot per pause-cycle).
    already_alerted = s.get("auto_resume_alerted_at") is not None
    if seconds_paused >= AUTO_RESUME_ALERT_AT_SEC and not already_alerted:
        _state._append_audit(
            "auto_resume_alert",
            seconds_paused=round(seconds_paused, 1),
            current_nav=current_nav,
        )
        # In-memory state update so subsequent cycles don't re-alert.
        with _state._lock:
            _state._auto_resume_alerted_at = datetime.now(timezone.utc).isoformat()
        # Fail-open Slack dispatch.
        try:
            from backend.services.observability.alerting import raise_cron_alert_sync
            raise_cron_alert_sync(
                source="kill_switch",
                error_type="auto_resume_pending",
                severity="P2",
                title="Kill-switch auto-resume will fire in ~1h (no breach detected)",
                details={
                    "seconds_paused": str(round(seconds_paused, 1)),
                    "auto_resume_at_sec": str(AUTO_RESUME_TRIGGER_AT_SEC),
                    "current_nav": str(current_nav),
                    "breach": str(breach),
                },
            )
        except Exception as exc:
            logger.warning("kill_switch auto_resume_alert dispatch fail-open: %r", exc)
        return {
            "action": "alert",
            "reason": "no_breach_for_1h_pager_fired",
            "seconds_paused": round(seconds_paused, 1),
        }

    return {
        "action": "no_op",
        "reason": "paused_but_under_hysteresis_threshold",
        "seconds_paused": round(seconds_paused, 1),
    }
