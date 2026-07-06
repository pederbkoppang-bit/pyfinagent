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

phase-23.2.18 (2026-05-05): rewrote the routing path. The previous version
called `backend.slack_bot.scheduler.send_trading_escalation` without `await`
and without the required `app: AsyncApp` argument, so every alert raised
TypeError into the fail-open `except` and was silently dropped. The
slack_bot process is also separate from the backend, so the AsyncApp
coupling was wrong by construction. We now route through
`backend.tools.slack.send_notification` (an async webhook helper). Two
public entry points:

- `raise_cron_alert(...)` -- async, awaitable. Use from async paths.
- `raise_cron_alert_sync(...)` -- sync wrapper. Schedules the
  coroutine via the running loop if one exists, else `asyncio.run`.
  Use from sync paths (kill_switch.pause, cycle_health.record_cycle_end).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque

logger = logging.getLogger(__name__)


# phase-62.7 (R-1): P1 added -- page-class severities bypass the consecutive
# threshold. The deduper's 3-in-5-min rule silently swallowed ONE-SHOT P1s
# (e.g. the kill-switch breach alert fires once per cycle), so a real breach
# paged nobody. P0/P1 = page-worthy by definition; single occurrence fires.
# phase-66 hotfix (2026-07-07, P1 page storm): "single occurrence fires"
# never meant "EVERY occurrence fires". The blanket bypass turned the
# 60s-polled freshness alarm (cycle_health._fire_freshness_alarm, whose
# docstring explicitly relies on this deduper) into ~120 pages/hour the
# moment a dashboard tab was open against a red table. Critical severities
# now bypass the consecutive THRESHOLD only; the repeat window still
# applies per (source, error_type) -- first occurrence pages instantly,
# repeats are suppressed for repeat_hours (default 1h).
_CRITICAL_SEVERITIES = frozenset({"P0", "P1", "critical", "CRITICAL"})


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

        if severity in _CRITICAL_SEVERITIES:
            with self._lock:
                st = self._state.setdefault((source, error_type), _AlertState())
                st.occurrences.append(now)
                fire = (
                    st.last_fired_at is None
                    or (now - st.last_fired_at) >= self.repeat
                )
                if fire:
                    st.last_fired_at = now
            return fire

        with self._lock:
            key = (source, error_type)
            st = self._state.setdefault(key, _AlertState())
            st.occurrences.append(now)
            cutoff = now - self.window
            while st.occurrences and st.occurrences[0] < cutoff:
                st.occurrences.popleft()
            if len(st.occurrences) < self.threshold:
                return False
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


async def _bot_token_fallback(
    source: str, severity: str, title: str, details: dict | str
) -> bool:
    """phase-62.7 (R-1): deliver a page-class alert via the bot token when the
    webhook path is unconfigured. urllib + asyncio.to_thread (no new deps;
    safe from any loop). Fail-open: returns False, never raises."""
    try:
        import json as _json
        import urllib.request as _ur

        from backend.config.settings import get_settings

        s = get_settings()
        tok = s.slack_bot_token.get_secret_value() if hasattr(
            s.slack_bot_token, "get_secret_value") else str(s.slack_bot_token)
        channel = getattr(s, "slack_channel_id", "") or "C0ANTGNNK8D"
        if not tok:
            logger.warning("alert bot-token fallback: no bot token configured")
            return False
        detail_str = (" | ".join(f"{k}={v}" for k, v in details.items())
                      if isinstance(details, dict) else str(details))
        body = _json.dumps({
            "channel": channel,
            "text": f"[{severity}] {title} -- {source}: {detail_str[:1500]}",
        }).encode()

        def _post() -> bool:
            req = _ur.Request(
                "https://slack.com/api/chat.postMessage", data=body,
                headers={"Authorization": f"Bearer {tok}",
                         "Content-Type": "application/json; charset=utf-8"})
            with _ur.urlopen(req, timeout=10) as r:
                return bool(_json.load(r).get("ok"))

        ok = await asyncio.to_thread(_post)
        logger.warning("alert bot-token fallback delivered=%s source=%s title=%r",
                       ok, source, title)
        return ok
    except Exception as exc:
        logger.warning("alert bot-token fallback fail-open: %r", exc)
        return False


async def raise_cron_alert(
    source: str,
    error_type: str,
    severity: str,
    title: str,
    details: dict | str,
) -> bool:
    """Dedup-aware cron alert. Returns True iff an alert was actually emitted.

    Routes through the webhook helper at `backend.tools.slack.send_notification`,
    not the AsyncApp-coupled `send_trading_escalation` (which lives in the
    separate slack_bot process). Fail-open: if the webhook is not configured
    or raises, log a WARNING and return False. Never raises out.

    Args:
        source: subsystem id, e.g. "autonomous_loop", "kill_switch".
        error_type: short tag for dedup, e.g. "cycle_timeout", "auto_pause".
        severity: "P0" | "P1" | "P2".
        title: short alert title.
        details: dict (preferred) or str -- key/value context lines.
    """
    deduper = _get_default_deduper()
    if not deduper.should_fire(source, error_type, severity=severity):
        return False

    try:
        from backend.config.settings import get_settings
        from backend.tools.slack import send_notification

        settings = get_settings()
        webhook = getattr(settings, "slack_webhook_url", "") or ""
        if not webhook:
            # phase-62.7 (R-1): the webhook is EMPTY on this machine, which
            # silently killed every webhook-path alert. Page-class severities
            # fall back to the bot-token chat.postMessage path -- the same
            # credential that delivers the daily digests (live-proven by the
            # 62.5 healthcheck drill). Non-page severities keep the old
            # warn-and-return-False behavior (no spam channel).
            if severity in _CRITICAL_SEVERITIES:
                return await _bot_token_fallback(source, severity, title, details)
            logger.warning(
                "raise_cron_alert: slack_webhook_url not configured "
                "(source=%s severity=%s title=%r)",
                source, severity, title,
            )
            return False

        if isinstance(details, dict):
            metadata = {str(k): str(v) for k, v in details.items()}
        else:
            metadata = {"details": str(details)}
        metadata.setdefault("source", source)
        metadata.setdefault("severity", severity)
        metadata.setdefault("error_type", error_type)

        alert_type = "error" if severity in _CRITICAL_SEVERITIES or severity == "P1" else "warning"
        message = f"[{severity}] {title}"

        await send_notification(webhook, message, metadata, alert_type=alert_type)
        logger.warning(
            "raise_cron_alert sent: source=%s severity=%s title=%r",
            source, severity, title,
        )
        return True
    except Exception as exc:
        logger.warning(
            "raise_cron_alert fail-open source=%s severity=%s err=%r",
            source,
            severity,
            exc,
        )
        return False


def raise_cron_alert_sync(
    source: str,
    error_type: str,
    severity: str,
    title: str,
    details: dict | str,
) -> bool:
    """Sync wrapper for `raise_cron_alert`. Use from sync code paths.

    If a running loop is detected, schedules the coroutine on it (returns
    True optimistically since fire-and-forget). Otherwise runs the coroutine
    to completion via `asyncio.run` and returns the actual result.

    Always fail-open: never raises out.
    """
    coro = raise_cron_alert(source, error_type, severity, title, details)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        try:
            loop.create_task(coro)
            return True
        except Exception as exc:
            logger.warning("raise_cron_alert_sync schedule failed: %r", exc)
            coro.close()
            return False

    try:
        return asyncio.run(coro)
    except Exception as exc:
        logger.warning("raise_cron_alert_sync run failed: %r", exc)
        return False


def reset_default_deduper() -> None:
    """Test helper: drop the process-wide deduper."""
    global _DEFAULT_DEDUPER
    _DEFAULT_DEDUPER = None


__all__ = [
    "AlertDeduper",
    "raise_cron_alert",
    "raise_cron_alert_sync",
    "reset_default_deduper",
]
