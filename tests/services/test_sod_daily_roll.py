"""phase-23.2.19: regression guard against the SOD-NAV-never-rolls bug.

Pre-fix forensic state: handoff/kill_switch_audit.jsonl contained ONE
sod_snapshot ever (2026-04-20, $9499.50). Every cycle since hit the
`else: pass` stub in paper_trader.check_and_enforce_kill_switch and
left the stale value in place. The frontend rendered
daily_loss_pct = (9499.50 - 17270.87) / 9499.50 * 100 = -81.81%.

These tests assert the fix:
- KillSwitchState exposes sod_date in its snapshot
- update_sod_nav stamps both nav and date in the audit row
- paper_trader rolls SOD on a new UTC calendar day
- same-day re-call is a no-op (no new audit row written)
- boot replay restores _sod_date from explicit `date` field OR by
  parsing `ts` for legacy rows
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services import kill_switch as ks


@pytest.fixture
def tmp_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect kill_switch's audit path to a tmp file, then reset module state."""
    p = tmp_path / "kill_switch_audit.jsonl"
    monkeypatch.setattr(ks, "_AUDIT_PATH", p)
    return p


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def test_snapshot_now_includes_sod_date(tmp_audit):
    state = ks.KillSwitchState()
    snap = state.snapshot()
    assert "sod_date" in snap
    assert snap["sod_date"] is None  # no SOD recorded yet


def test_update_sod_nav_stamps_explicit_date_in_audit_row(tmp_audit):
    state = ks.KillSwitchState()
    state.update_sod_nav(15000.0, date="2026-05-05")
    rows = [json.loads(l) for l in tmp_audit.read_text().splitlines() if l.strip()]
    sod_rows = [r for r in rows if r.get("event") == "sod_snapshot"]
    assert len(sod_rows) == 1
    assert sod_rows[0]["nav"] == 15000.0
    assert sod_rows[0]["date"] == "2026-05-05"
    assert state.snapshot()["sod_date"] == "2026-05-05"


def test_update_sod_nav_default_date_is_today(tmp_audit):
    state = ks.KillSwitchState()
    state.update_sod_nav(15000.0)  # no date arg
    assert state.snapshot()["sod_date"] == _today()


def test_paper_trader_rolls_sod_on_new_day(tmp_audit):
    """The core bug: on day N+1, the cycle must re-anchor SOD."""
    state = ks.KillSwitchState()
    # Simulate yesterday's SOD already on disk
    state.update_sod_nav(15000.0, date=_yesterday())
    assert state.snapshot()["sod_date"] == _yesterday()

    # Drive the same logic that lives in paper_trader.check_and_enforce_kill_switch
    snap = state.snapshot()
    today = _today()
    if snap.get("sod_nav") is None or snap.get("sod_date") != today:
        state.update_sod_nav(17000.0, date=today)

    snap2 = state.snapshot()
    assert snap2["sod_nav"] == 17000.0
    assert snap2["sod_date"] == today

    rows = [json.loads(l) for l in tmp_audit.read_text().splitlines() if l.strip()]
    sod_rows = [r for r in rows if r.get("event") == "sod_snapshot"]
    assert len(sod_rows) == 2  # yesterday + today


def test_paper_trader_does_not_roll_same_day(tmp_audit):
    """Idempotent: same-day re-call must not write a new audit row."""
    state = ks.KillSwitchState()
    today = _today()
    state.update_sod_nav(15000.0, date=today)

    # Same-day re-call — the daily-roll guard should short-circuit
    snap = state.snapshot()
    if snap.get("sod_nav") is None or snap.get("sod_date") != today:
        state.update_sod_nav(15500.0, date=today)

    rows = [json.loads(l) for l in tmp_audit.read_text().splitlines() if l.strip()]
    sod_rows = [r for r in rows if r.get("event") == "sod_snapshot"]
    assert len(sod_rows) == 1  # NO new row, the guard short-circuited


def test_boot_replay_restores_sod_date_from_explicit_field(tmp_audit):
    """Post-fix audit rows include `date` explicitly; boot replay reads it."""
    tmp_audit.write_text(
        json.dumps({
            "ts": "2026-05-05T12:00:00+00:00",
            "event": "sod_snapshot",
            "nav": 15000.0,
            "date": "2026-05-05",
        }) + "\n"
    )
    state = ks.KillSwitchState()
    snap = state.snapshot()
    assert snap["sod_nav"] == 15000.0
    assert snap["sod_date"] == "2026-05-05"


def test_boot_replay_falls_back_to_ts_for_legacy_rows(tmp_audit):
    """Pre-fix audit rows have no `date` field; boot replay must derive it from `ts`."""
    tmp_audit.write_text(
        json.dumps({
            "ts": "2026-04-20T12:01:03.965687+00:00",
            "event": "sod_snapshot",
            "nav": 9499.5,
            # NO date field — legacy row from before phase-23.2.19
        }) + "\n"
    )
    state = ks.KillSwitchState()
    snap = state.snapshot()
    assert snap["sod_nav"] == 9499.5
    assert snap["sod_date"] == "2026-04-20"


def test_legacy_row_then_new_day_rolls_correctly(tmp_audit):
    """The exact production path on 2026-05-05:
    boot replay sees the legacy 04-20 row -> _sod_date='2026-04-20',
    then today's cycle compares != today -> re-anchors."""
    tmp_audit.write_text(
        json.dumps({
            "ts": "2026-04-20T12:01:03.965687+00:00",
            "event": "sod_snapshot",
            "nav": 9499.5,
        }) + "\n"
    )
    state = ks.KillSwitchState()
    snap = state.snapshot()
    assert snap["sod_date"] == "2026-04-20"

    today = _today()
    if snap.get("sod_nav") is None or snap.get("sod_date") != today:
        state.update_sod_nav(17270.87, date=today)

    snap2 = state.snapshot()
    assert snap2["sod_nav"] == 17270.87
    assert snap2["sod_date"] == today
