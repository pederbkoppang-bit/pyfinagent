"""phase-6.7 alert dedup + Slack routing wrapper.

Research (brief 2026-04-19):
- Alertmanager defaults: group_wait=30s, repeat_interval=1h
- `for: 5m` minimum before firing prevents transient-blip noise
- Inhibition rule: critical suppresses lower-severity same-source
- in-memory dedup sufficient for single-process asyncio app

The dedup rule: an alert fires when there have been `consecutive_failure_threshold`
(default 3) occurrences of the same `(source, error_type)` within
`debounce_minutes` (default 5). After firing, the same alert will not fire
again until `repeat_hours` have passed. Critical severity bypasses all of
this and always fires.

Alerts route through the existing `backend/slack_bot/scheduler.py::
send_trading_escalation(severity, title, details)` which handles Slack +
iMessage (P0) fan-out.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque

logger = logging.getLogger(__name__)


_CRITICAL_SEVERITIES = frozenset({"P0", "critical", "CRITICAL"})


@dataclass
class _AlertState:
    occurrences: Deque[datetime] = field(default_factory=deque)
    last_fired_at: datetime | None = None


class AlertDeduper:
    """Thread-safe dedup tracker for (source, error_type) alert bursts."""

    def __init__(
        self,
        window_minutes: int = 5,
        repeat_hours: int = 1,
        consecutive_threshold: int = 3,
    ) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.repeat = timedelta(hours=repeat_hours)
        self.threshold = consecutive_threshold
        self._state: dict[tuple[str, str], _AlertState] = {}
        self._lock = threading.Lock()

    def should_fire(
        self, source: str, error_type: str, *, severity: str = "P2"
    ) -> bool:
        now = datetime.now(timezone.utc)

        # critical alerts bypass dedup entirely
        if severity in _CRITICAL_SEVERITIES:
            with self._lock:
                st = self._state.setdefault((source, error_type), _AlertState())
                st.occurrences.append(now)
                st.last_fired_at = now
            return True

        with self._lock:
            key = (source, error_type)
            st = self._state.setdefault(key, _AlertState())
            st.occurrences.append(now)
            # evict occurrences older than window
            cutoff = now - self.window
            while st.occurrences and st.occurrences[0] < cutoff:
                st.occurrences.popleft()
            if len(st.occurrences) < self.threshold:
                return False
            # respect repeat interval
            if st.last_fired_at is not None and (now - st.last_fired_at) < self.repeat:
                return False
            st.last_fired_at = now
            return True

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


_DEFAULT_DEDUPER: AlertDeduper | None = None


def _get_default_deduper() -> AlertDeduper:
    global _DEFAULT_DEDUPER
    if _DEFAULT_DEDUPER is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            _DEFAULT_DEDUPER = AlertDeduper(
                window_minutes=int(getattr(s, "alert_debounce_minutes", 5)),
                repeat_hours=int(getattr(s, "alert_repeat_hours", 1)),
                consecutive_threshold=int(
                    getattr(s, "alert_consecutive_failure_threshold", 3)
                ),
            )
        except Exception:  # pragma: no cover
            _DEFAULT_DEDUPER = AlertDeduper()
    return _DEFAULT_DEDUPER


def raise_cron_alert(
    source: str,
    error_type: str,
    severity: str,
    title: str,
    details: str,
) -> bool:
    """Dedup-aware cron alert. Returns True iff an alert was actually emitted.

    Fail-open: if Slack routing is not configured or raises, we log a
    WARNING and return False. Never raises out.
    """
    deduper = _get_default_deduper()
    if not deduper.should_fire(source, error_type, severity=severity):
        return False
    try:
        from backend.slack_bot.scheduler import send_trading_escalation

        send_trading_escalation(severity=severity, title=title, details=details)
        return True
    except Exception as exc:
        logger.warning(
            "raise_cron_alert fail-open source=%s severity=%s err=%r",
            source,
            severity,
            exc,
        )
        return False


def reset_default_deduper() -> None:
    """Test helper: drop the process-wide deduper."""
    global _DEFAULT_DEDUPER
    _DEFAULT_DEDUPER = None


__all__ = [
    "AlertDeduper",
    "raise_cron_alert",
    "reset_default_deduper",
]
