"""phase-38.2 verification: cycle_history.jsonl 'started' row + orphan audit.

Per masterplan 38.2 criteria:
  1. record_cycle_start_writes_cycle_starting_row_immediately
  2. row_persists_if_cycle_dies_mid_flight
  3. next_cycle_can_audit_orphan_rows

OPEN-11 fix: previously record_cycle_start only wrote a heartbeat (overwrite-
only single file); cycle_history.jsonl was written ONLY at record_cycle_end.
A SIGKILL / OOM / power-loss mid-cycle left no trace -- the 08:14 CEST
'lost cycle 3a' incident.

Tests use a tmp_path-isolated _HISTORY_PATH so they NEVER touch the real
handoff/cycle_history.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import cycle_health


@pytest.fixture(autouse=True)
def _isolate_history(monkeypatch, tmp_path):
    history = tmp_path / "cycle_history.jsonl"
    heartbeat = tmp_path / ".cycle_heartbeat.json"
    monkeypatch.setattr(cycle_health, "_HISTORY_PATH", history)
    monkeypatch.setattr(cycle_health, "_HEARTBEAT_PATH", heartbeat)
    yield history


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_phase_38_2_record_cycle_start_writes_started_row_immediately(_isolate_history):
    """Criterion 1: a started row exists in cycle_history.jsonl after
    record_cycle_start(), BEFORE record_cycle_end is ever called."""
    log = cycle_health.CycleHealthLog()
    log.record_cycle_start("abc123")
    rows = _read_rows(_isolate_history)
    assert len(rows) == 1
    r = rows[0]
    assert r["cycle_id"] == "abc123"
    assert r["status"] == "started"
    assert r["completed_at"] is None
    assert r["duration_ms"] is None


def test_phase_38_2_started_row_persists_if_cycle_dies_mid_flight(_isolate_history):
    """Criterion 2: simulate mid-flight death by calling start but NEVER
    end. Row must persist for the next process to audit."""
    log = cycle_health.CycleHealthLog()
    log.record_cycle_start("orphan_1")
    # Simulate process death by NOT calling record_cycle_end.
    # Reconstruct a fresh logger -- mimics process restart.
    log2 = cycle_health.CycleHealthLog()
    rows = _read_rows(_isolate_history)
    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "orphan_1"
    assert rows[0]["status"] == "started"
    # And orphan_rows() exposes it to the next cycle
    orphans = log2.orphan_rows()
    assert len(orphans) == 1
    assert orphans[0]["cycle_id"] == "orphan_1"


def test_phase_38_2_completed_cycle_is_not_an_orphan(_isolate_history):
    """Counter-test: a cycle with both start and end rows is NOT an orphan."""
    log = cycle_health.CycleHealthLog()
    started = log.record_cycle_start("ok_1")
    log.record_cycle_end("ok_1", started, "completed", n_trades=2)
    assert log.orphan_rows() == []


def test_phase_38_2_orphan_rows_distinguishes_orphan_from_completed(_isolate_history):
    """Criterion 3: orphan_rows returns ONLY the started-without-terminal
    cycles, not the completed ones. Includes mixed-status sequence."""
    log = cycle_health.CycleHealthLog()
    # Cycle 1: completed normally
    s1 = log.record_cycle_start("cycle_1")
    log.record_cycle_end("cycle_1", s1, "completed")
    # Cycle 2: dies (no end)
    log.record_cycle_start("cycle_2_orphan")
    # Cycle 3: failed (terminal but non-completed)
    s3 = log.record_cycle_start("cycle_3")
    log.record_cycle_end("cycle_3", s3, "failed", error_count=1)
    # Cycle 4: another orphan
    log.record_cycle_start("cycle_4_orphan")
    orphans = {r["cycle_id"] for r in log.orphan_rows()}
    assert orphans == {"cycle_2_orphan", "cycle_4_orphan"}


def test_phase_38_2_last_cycles_excludes_started_by_default(_isolate_history):
    """The default last_cycles call filters out started rows so existing
    UI / alarm callers see only terminal rows."""
    log = cycle_health.CycleHealthLog()
    s1 = log.record_cycle_start("c1")
    log.record_cycle_end("c1", s1, "completed")
    log.record_cycle_start("c2_orphan")  # never ends
    s3 = log.record_cycle_start("c3")
    log.record_cycle_end("c3", s3, "completed")
    visible = [r["cycle_id"] for r in log.last_cycles(n=10)]
    # Each cycle writes 2 rows; visible should be {c1, c3} terminal rows only
    assert "c2_orphan" not in visible
    assert "c1" in visible and "c3" in visible


def test_phase_38_2_last_cycles_include_started_flag_surfaces_orphans(_isolate_history):
    """include_started=True must expose the started rows to UI / audit tooling."""
    log = cycle_health.CycleHealthLog()
    log.record_cycle_start("orphan_x")
    all_rows = log.last_cycles(n=10, include_started=True)
    statuses = {r["status"] for r in all_rows}
    assert "started" in statuses


def test_phase_38_2_alarm_skips_started_rows_so_halted_cycle_triggers(_isolate_history, monkeypatch):
    """Regression: a halted cycle leaves a started row at the tail. The alarm
    MUST skip it and look at the previous terminal row -- else the alarm is
    silenced precisely when it should fire (the lost-cycle-3a failure mode)."""
    log = cycle_health.CycleHealthLog()
    # Write a stale completed cycle 48h ago
    from datetime import datetime, timezone, timedelta

    old_iso = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

    row_completed = {
        "cycle_id": "old_completed",
        "started_at": old_iso,
        "completed_at": old_iso,
        "duration_ms": 1000,
        "status": "completed",
        "n_trades": 0,
        "error_count": 0,
        "data_source_ages": {},
        "bq_ingest_lag_sec": None,
    }
    with _isolate_history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row_completed) + "\n")
    # Now a fresh start row at the tail (simulates halted cycle)
    log.record_cycle_start("halted_cycle")
    # Force is_weekday_et=True path by manually patching the helper if needed --
    # use the actual function and assert it picks the old completed row.
    verdict = cycle_health.cycle_heartbeat_alarm(threshold_sec=3600)
    # Should be stale (48h > 1h threshold) -- proves alarm did NOT short-circuit
    # on the started row's completed_at=null.
    assert verdict["last_completed_at"] == old_iso
    assert verdict["stale"] is True


def test_phase_38_2_started_row_uses_threading_lock(_isolate_history):
    """Mutation-resistance: concurrent record_cycle_start calls must produce
    well-formed JSONL lines (POSIX O_APPEND + threading.Lock belt-and-braces)."""
    import threading

    log = cycle_health.CycleHealthLog()

    def writer(cid):
        log.record_cycle_start(cid)

    threads = [threading.Thread(target=writer, args=(f"c_{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All 20 rows must be parseable JSONL -- no interleaving / truncation
    rows = _read_rows(_isolate_history)
    assert len(rows) == 20
    cids = {r["cycle_id"] for r in rows}
    assert cids == {f"c_{i}" for i in range(20)}
