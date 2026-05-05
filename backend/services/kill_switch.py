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
                    elif row.get("event") == "resume":
                        self._paused = False
                        self._pause_reason = None
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
        re-anchor SOD on a new UTC calendar day."""
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "sod_nav": self._sod_nav,
            "sod_date": self._sod_date,
            "peak_nav": self._peak_nav,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    # ── State transitions ──────────────────────────────────────────
    def pause(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = True
            self._pause_reason = trigger
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

    def resume(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = False
            self._pause_reason = None
            self._append_audit("resume", trigger=trigger, details=details or {})
            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
            # re-acquiring self._lock (threading.Lock is not reentrant).
            return self._snapshot_locked()

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
