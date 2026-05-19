"""phase-30.1 tests for cycle_heartbeat_alarm.

Audit basis: handoff/archive/phase-30.0/experiment_results.md Anomaly C
(65h 34m cycle gap 2026-05-17 00:26 UTC -> 2026-05-19 18:00 UTC with no
out-of-band alert path). The alarm closes that gap by reading the
cycle_history.jsonl tail at watchdog-cron cadence and firing P1 when
no cycle has completed in >26h on a weekday.

Test plan (7 cases):
  1. Fresh cycle on weekday  -> stale=False, should_alarm=False
  2. Stale 26h on weekday    -> stale=True,  should_alarm=True
  3. Stale 26h on Saturday   -> stale=True,  should_alarm=False
  4. Stale 26h on Sunday     -> stale=True,  should_alarm=False
  5. Missing history file    -> stale=False, should_alarm=False
  6. Empty history file      -> stale=False, should_alarm=False
  7. Malformed JSON last row -> stale=False, should_alarm=False
     (graceful via fallback to next-newest valid row OR sentinel)

Mocking strategy:
- `_HISTORY_PATH` patched to a tmp_path file we control.
- `_now_utc` patched to return a deterministic datetime.
- `raise_cron_alert_sync` is NOT exercised in these unit tests; the
  pure-function `cycle_heartbeat_alarm` returns a verdict dict only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.services import cycle_health


def _write_history(path: Path, completed_at_iso: str) -> None:
    """Helper: write a single cycle_history.jsonl row with a given
    completed_at timestamp."""
    row = {
        "cycle_id": "abc12345",
        "started_at": completed_at_iso,
        "completed_at": completed_at_iso,
        "duration_ms": 300_000,
        "status": "completed",
        "n_trades": 0,
        "error_count": 0,
        "data_source_ages": {},
        "bq_ingest_lag_sec": None,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _set_now(monkeypatch: pytest.MonkeyPatch, iso_utc: str) -> None:
    """Patch cycle_health._now_utc to return a deterministic UTC datetime."""
    target = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    monkeypatch.setattr(cycle_health, "_now_utc", lambda: target)


def _set_history(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Patch the module-level _HISTORY_PATH constant."""
    monkeypatch.setattr(cycle_health, "_HISTORY_PATH", path)


# ---------------------------------------------------------------------
# 1. Fresh cycle on a weekday -> not stale, no alarm
# ---------------------------------------------------------------------
def test_fresh_cycle_on_weekday_no_alarm(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    # Last cycle completed 2 hours ago.
    _write_history(history, "2026-05-19T16:00:00+00:00")
    _set_history(monkeypatch, history)
    # Now: Tuesday 2026-05-19 18:00 UTC = 14:00 ET (weekday).
    _set_now(monkeypatch, "2026-05-19T18:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is False
    assert verdict["should_alarm"] is False
    assert verdict["is_weekday_et"] is True
    assert verdict["age_sec"] is not None
    assert verdict["age_sec"] < cycle_health._CYCLE_HEARTBEAT_STALE_SEC


# ---------------------------------------------------------------------
# 2. Stale 26h on a weekday -> alarm fires
# ---------------------------------------------------------------------
def test_stale_26h_on_weekday_alarms(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    # Last cycle completed 27h ago (> 26h threshold).
    _write_history(history, "2026-05-18T15:00:00+00:00")
    _set_history(monkeypatch, history)
    # Now: Tuesday 2026-05-19 18:00 UTC = 14:00 ET (weekday).
    _set_now(monkeypatch, "2026-05-19T18:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is True
    assert verdict["should_alarm"] is True
    assert verdict["is_weekday_et"] is True
    assert verdict["age_sec"] > cycle_health._CYCLE_HEARTBEAT_STALE_SEC


# ---------------------------------------------------------------------
# 3. Stale 26h on Saturday -> stale True, alarm suppressed (no weekend cron)
# ---------------------------------------------------------------------
def test_stale_26h_on_saturday_no_alarm(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    # Last cycle completed Friday afternoon ET ~30h before "now"
    _write_history(history, "2026-05-22T15:00:00+00:00")
    _set_history(monkeypatch, history)
    # Now: Saturday 2026-05-23 21:00 UTC = 17:00 ET (weekend, weekday()==5).
    _set_now(monkeypatch, "2026-05-23T21:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is True  # age > threshold
    assert verdict["is_weekday_et"] is False
    assert verdict["should_alarm"] is False  # suppressed on weekend


# ---------------------------------------------------------------------
# 4. Stale 30h on Sunday -> stale True, alarm suppressed
# ---------------------------------------------------------------------
def test_stale_30h_on_sunday_no_alarm(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    _write_history(history, "2026-05-23T15:00:00+00:00")
    _set_history(monkeypatch, history)
    # Now: Sunday 2026-05-24 21:00 UTC = 17:00 ET (weekday()==6).
    _set_now(monkeypatch, "2026-05-24T21:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is True
    assert verdict["is_weekday_et"] is False
    assert verdict["should_alarm"] is False


# ---------------------------------------------------------------------
# 5. Missing history file -> sentinel
# ---------------------------------------------------------------------
def test_missing_history_file_returns_sentinel(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"  # not created
    _set_history(monkeypatch, history)
    _set_now(monkeypatch, "2026-05-19T18:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is False
    assert verdict["should_alarm"] is False
    assert verdict["age_sec"] is None
    assert verdict["last_completed_at"] is None


# ---------------------------------------------------------------------
# 6. Empty history file -> sentinel
# ---------------------------------------------------------------------
def test_empty_history_file_returns_sentinel(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    history.write_text("", encoding="utf-8")
    _set_history(monkeypatch, history)
    _set_now(monkeypatch, "2026-05-19T18:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    assert verdict["stale"] is False
    assert verdict["should_alarm"] is False
    assert verdict["age_sec"] is None
    assert verdict["last_completed_at"] is None


# ---------------------------------------------------------------------
# 7. Malformed JSON last row -> falls back to next-newest valid row
# ---------------------------------------------------------------------
def test_malformed_last_row_falls_back_to_prev(tmp_path, monkeypatch):
    history = tmp_path / "cycle_history.jsonl"
    good_row = {
        "cycle_id": "good01",
        "started_at": "2026-05-19T16:00:00+00:00",
        "completed_at": "2026-05-19T16:00:00+00:00",
        "duration_ms": 300_000,
        "status": "completed",
        "n_trades": 0,
        "error_count": 0,
        "data_source_ages": {},
        "bq_ingest_lag_sec": None,
    }
    # Write 1 good row then 1 malformed line.
    history.write_text(
        json.dumps(good_row) + "\n" + "{ not valid json\n",
        encoding="utf-8",
    )
    _set_history(monkeypatch, history)
    _set_now(monkeypatch, "2026-05-19T18:00:00+00:00")

    verdict = cycle_health.cycle_heartbeat_alarm()

    # Falls back to the good row (2h old, weekday) -> fresh, no alarm.
    assert verdict["stale"] is False
    assert verdict["should_alarm"] is False
    assert verdict["last_completed_at"] == "2026-05-19T16:00:00+00:00"
