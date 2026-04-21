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

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "sod_nav": self._sod_nav,
                "peak_nav": self._peak_nav,
            }

    # ── State transitions ──────────────────────────────────────────
    def pause(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = True
            self._pause_reason = trigger
            self._append_audit("pause", trigger=trigger, details=details or {})
            return self.snapshot()

    def resume(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = False
            self._pause_reason = None
            self._append_audit("resume", trigger=trigger, details=details or {})
            return self.snapshot()

    def update_sod_nav(self, nav: float) -> None:
        """Record start-of-day NAV for daily-loss calculation. Idempotent per day."""
        with self._lock:
            self._sod_nav = float(nav)
            self._append_audit("sod_snapshot", nav=self._sod_nav)

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
