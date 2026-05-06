"""phase-23.2.18: regression guard against silent cycle failures.

Pre-fix forensic state from 2026-04-30 / 05-01 / 05-04 / 05-05:
- handoff/cycle_history.jsonl had no completion row for any of those dates
- handoff/.cycle_heartbeat.json was stuck at event=start
- the user got NO notification about the cycle hang or watchdog kickstart

Two-level root cause:
1. The cycle hung inside an unbounded asyncio.to_thread(); event loop alive
   (so /api/health stayed responsive and the watchdog stayed silent), cycle
   never advanced, finally never fired (or ran during cancel and the in-
   process alert was the only chance to notify the operator -- and it was
   broken).
2. backend/services/observability/alerting.py.raise_cron_alert was missing
   `await` AND the required AsyncApp argument when calling the slack_bot
   coroutine -- so every alert raised TypeError into the fail-open `except`
   and was silently dropped.

These tests assert the fix:
- raise_cron_alert is now async and routes via the webhook helper
- raise_cron_alert_sync schedules / runs the coroutine without crashing
- kill_switch.pause(trigger="auto-...") fires the alert; manual/test/bench
  triggers do NOT fire it
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from backend.services.observability import alerting
from backend.services import kill_switch


@pytest.fixture(autouse=True)
def _reset_dedup():
    alerting.reset_default_deduper()
    yield
    alerting.reset_default_deduper()


@pytest.fixture(autouse=True)
def _isolated_kill_switch_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """phase-23.2.22: redirect kill_switch._AUDIT_PATH to tmp so tests cannot
    write real pause events to production handoff/kill_switch_audit.jsonl.
    Pre-fix forensic: a 2026-05-05 pytest run wrote 7 spurious pause events
    (drawdown_breach + manual + test + test-pre + bench-1/2/3) into prod,
    creating a latent boot-paused risk for the next backend restart."""
    p = tmp_path / "kill_switch_audit.jsonl"
    monkeypatch.setattr(kill_switch, "_AUDIT_PATH", p)
    return p


@pytest.fixture
def captured_calls(monkeypatch):
    """Capture every call to backend.tools.slack.send_notification."""
    calls: list[dict[str, Any]] = []

    async def _fake_send(webhook_url, message, metadata, alert_type="info"):
        calls.append({
            "webhook_url": webhook_url,
            "message": message,
            "metadata": metadata,
            "alert_type": alert_type,
        })

    # patch the import target inside alerting.raise_cron_alert
    monkeypatch.setattr(
        "backend.tools.slack.send_notification",
        _fake_send,
    )
    # also stub a webhook URL into settings so the early-exit doesn't fire
    from backend.config.settings import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "slack_webhook_url", "https://hooks.slack.test/T0/B0/X")
    return calls


def _drive_threshold(severity: str = "P1"):
    """raise_cron_alert dedups: needs N occurrences within window. Drive
    enough occurrences past the threshold so the alert actually fires.
    P0/critical bypass dedup; P1/P2 require the threshold."""
    asyncio.run(_drive_threshold_async(severity))


async def _drive_threshold_async(severity: str):
    # Use a tiny deduper so threshold is 1.
    alerting._DEFAULT_DEDUPER = alerting.AlertDeduper(
        window_minutes=5,
        repeat_hours=1,
        consecutive_threshold=1,
    )


def test_raise_cron_alert_fires_webhook_on_cycle_error(captured_calls):
    """The autonomous_loop failure path must reach the webhook helper."""
    _drive_threshold()
    fired = asyncio.run(
        alerting.raise_cron_alert(
            source="autonomous_loop",
            error_type="cycle_error",
            severity="P1",
            title="Autonomous trading cycle error",
            details={"cycle_id": "abc12345", "error": "boom"},
        )
    )
    assert fired is True
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["webhook_url"].startswith("https://hooks.slack.test/")
    assert "[P1]" in call["message"]
    assert call["metadata"]["cycle_id"] == "abc12345"
    assert call["metadata"]["source"] == "autonomous_loop"
    assert call["metadata"]["severity"] == "P1"
    assert call["alert_type"] == "error"


def test_raise_cron_alert_fail_open_when_no_webhook(captured_calls, monkeypatch):
    """Missing webhook -> log warn, return False, never raise."""
    from backend.config.settings import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "slack_webhook_url", "")
    _drive_threshold()
    fired = asyncio.run(
        alerting.raise_cron_alert(
            source="autonomous_loop",
            error_type="cycle_error",
            severity="P1",
            title="x",
            details={},
        )
    )
    assert fired is False
    assert captured_calls == []


def test_raise_cron_alert_sync_from_no_loop_runs_to_completion(captured_calls):
    """The sync wrapper, called outside any loop, runs the coroutine."""
    _drive_threshold()
    fired = alerting.raise_cron_alert_sync(
        source="kill_switch",
        error_type="auto_pause_drawdown",
        severity="P1",
        title="Auto-paused on drawdown",
        details={"trigger": "drawdown"},
    )
    assert fired is True
    assert len(captured_calls) == 1
    assert captured_calls[0]["metadata"]["source"] == "kill_switch"


def test_kill_switch_auto_pause_fires_alert(captured_calls):
    _drive_threshold()
    state = kill_switch.KillSwitchState()
    state.pause(trigger="drawdown_breach", details={"daily_loss_pct": -2.5})
    assert len(captured_calls) == 1
    md = captured_calls[0]["metadata"]
    assert md["trigger"] == "drawdown_breach"
    assert md["source"] == "kill_switch"


def test_kill_switch_manual_pause_does_not_alert(captured_calls):
    _drive_threshold()
    state = kill_switch.KillSwitchState()
    state.pause(trigger="manual")
    state.pause(trigger="test")
    state.pause(trigger="test-pre")
    state.pause(trigger="bench-1")
    state.pause(trigger="bench-2")
    state.pause(trigger="bench-3")
    assert captured_calls == []


def test_dedup_threshold_blocks_first_occurrences(captured_calls):
    """P1/P2 alerts must respect the dedup threshold so we don't spam."""
    # default threshold is 3 within 5min window
    alerting.reset_default_deduper()
    for _ in range(2):
        asyncio.run(
            alerting.raise_cron_alert(
                source="autonomous_loop",
                error_type="cycle_error",
                severity="P2",
                title="x",
                details={},
            )
        )
    # 2 occurrences -> below default threshold -> no alert yet
    assert captured_calls == []


def test_dedup_critical_severity_bypasses_threshold(captured_calls):
    """P0 alerts must always fire."""
    alerting.reset_default_deduper()
    fired = asyncio.run(
        alerting.raise_cron_alert(
            source="autonomous_loop",
            error_type="cycle_killed",
            severity="P0",
            title="critical",
            details={},
        )
    )
    assert fired is True
    assert len(captured_calls) == 1
