"""phase-38.6 verification: restart-survivable autonomous-cycle lock.

Closes closure_roadmap.md section 3 OPEN-15. Tests cover:
  1. Happy path: acquire writes pid+cycle_id, file unlinked on exit.
  2. In-process re-acquire raises CycleLockError.
  3. Simulated kill (stale mtime + dead pid): startup cleans + acquire succeeds.
  4. Live lock is NOT cleaned by clean_stale_lock.
  5. Malformed lockfile treated as stale.
  6. No-file path: inspect_lock returns None.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
import pytest

from backend.services import cycle_lock
from backend.services.cycle_lock import (
    CycleLockError,
    acquire,
    clean_stale_lock,
    inspect_lock,
    lock_ttl_sec,  # phase-85.5: replaces the frozen _LOCK_TTL_SEC constant
)


@pytest.fixture(autouse=True)
def _isolated_lock(tmp_path, monkeypatch):
    """Redirect _LOCK_PATH to tmp_path for the test. Ensures we don't
    touch the real handoff/.autonomous_loop.lock during testing."""
    fake_lock = tmp_path / ".autonomous_loop.lock"
    monkeypatch.setattr(cycle_lock, "_LOCK_PATH", fake_lock)
    monkeypatch.setattr(cycle_lock, "_HANDOFF", tmp_path)
    yield fake_lock
    if fake_lock.exists():
        fake_lock.unlink()


def test_phase_38_6_acquire_writes_pid_and_cycle_id_then_releases(_isolated_lock):
    """Happy path: acquire writes {pid, cycle_id, started_at} and marks the
    pidfile `released` on exit.

    phase-85.5 CHANGED THIS TEST. It previously asserted the lockfile was
    UNLINKED on exit. That unlink happened BEFORE LOCK_UN, so between the two
    the process was alive and still holding the flock while the path was free
    -- a second acquirer created a new inode and both believed they held the
    cycle lock, on the normal release path. Release now rewrites the payload
    in place and never unlinks; `state == "released"` is what marks the lock
    as collectable.
    """
    fake_lock = _isolated_lock
    with acquire("test-cycle-123"):
        assert fake_lock.exists(), "lockfile must exist while held"
        data = json.loads(fake_lock.read_text())
        assert data["pid"] == os.getpid()
        assert data["cycle_id"] == "test-cycle-123"
        assert "started_at" in data
        assert data["state"] == "held"
    # After exit the file REMAINS, marked released (never unlinked).
    assert fake_lock.exists(), "phase-85.5: release must NOT unlink the lockfile"
    released = json.loads(fake_lock.read_text())
    assert released["state"] == "released"
    assert inspect_lock()["is_stale"] is True, "a released lock is collectable"


def test_phase_38_6_second_acquire_in_same_process_raises(_isolated_lock):
    """In-process re-entrancy: while the first acquire holds, a second
    must raise CycleLockError (same-process flock contention)."""
    with acquire("outer-cycle"):
        with pytest.raises(CycleLockError) as exc_info:
            with acquire("inner-cycle"):
                pytest.fail("inner acquire should not succeed while outer holds")
        assert "another live cycle holds the lock" in str(exc_info.value)


def test_phase_38_6_simulated_kill_then_startup_cleans(_isolated_lock):
    """REGRESSION for OPEN-15: simulate a mid-cycle kill (pidfile
    on disk with dead pid + stale mtime), then verify next startup's
    clean_stale_lock unlinks it + a fresh acquire succeeds."""
    fake_lock = _isolated_lock

    # Simulate the SIGKILL-residue: write a pidfile with a guaranteed-dead
    # pid + backdate mtime to >TTL ago
    dead_pid = 99_999_999  # unlikely to ever be alive
    fake_lock.write_text(json.dumps({
        "pid": dead_pid,
        "cycle_id": "killed-cycle-456",
        "started_at": "2026-05-22T00:00:00+00:00",
    }))
    old_ts = time.time() - (lock_ttl_sec() + 600)
    os.utime(fake_lock, (old_ts, old_ts))

    # inspect: should report is_stale=True -- on the DEAD PID, not the age.
    state = inspect_lock()
    assert state is not None
    assert state["is_stale"] is True
    assert state["pid_alive"] is False
    assert state["age_sec"] > lock_ttl_sec()

    # clean_stale_lock should unlink + return the cleaned state
    cleaned = clean_stale_lock(reason="test_simulated_kill")
    assert cleaned is not None
    assert cleaned["pid"] == dead_pid
    assert not fake_lock.exists()

    # Fresh acquire after recovery
    with acquire("recovery-cycle"):
        data = json.loads(fake_lock.read_text())
        assert data["cycle_id"] == "recovery-cycle"
        assert data["pid"] == os.getpid()


def test_phase_38_6_live_lock_not_cleaned(_isolated_lock):
    """Negative case: don't clean a live process's lock.
    Write a lockfile with the current pid + fresh mtime; clean_stale_lock
    must return None (no-op)."""
    fake_lock = _isolated_lock
    fake_lock.write_text(json.dumps({
        "pid": os.getpid(),  # our own pid -> alive
        "cycle_id": "live-cycle-789",
        "started_at": "2026-05-23T00:00:00+00:00",
    }))
    # Fresh mtime (just now)

    state = inspect_lock()
    assert state is not None
    assert state["pid_alive"] is True
    assert state["is_stale"] is False

    result = clean_stale_lock(reason="test_should_be_noop")
    assert result is None, "live lock must NOT be cleaned"
    assert fake_lock.exists(), "live lockfile must remain"


def test_phase_38_6_malformed_lockfile_treated_as_stale(_isolated_lock):
    """Defensive: malformed lockfile content (bad JSON) -> is_stale=True
    so the next acquire can clean + recover."""
    fake_lock = _isolated_lock
    fake_lock.write_text("not valid json {{{")

    state = inspect_lock()
    assert state is not None
    assert state["is_stale"] is True
    assert "raw_error" in state


def test_phase_38_6_no_lock_file_returns_none(_isolated_lock):
    """No lockfile -> inspect_lock returns None."""
    fake_lock = _isolated_lock
    if fake_lock.exists():
        fake_lock.unlink()
    assert inspect_lock() is None
    assert clean_stale_lock() is None


# phase-85.5 REMOVED `test_phase_38_6_ttl_constant_is_90_minutes`, which
# asserted the literal `_LOCK_TTL_SEC == 5400`. That literal IS the bug: the
# constant froze at 5400s while settings.paper_cycle_max_seconds moved to
# 7200s, leaving the TTL at 0.75x the budget so every cycle past 90 minutes
# advertised a stale lock while alive. It is replaced by a strictly stronger,
# mutation-resistant relationship test that moves the budget and asserts the
# derived TTL follows it and never falls below it:
#   backend/tests/test_phase_85_5_cycle_lock_split_brain.py
#     ::test_85_5_c3_ttl_tracks_cycle_budget_and_never_falls_below_it
# That file is named so the immutable verification command's
# `-k "cycle_lock or 85_5"` actually collects it; this module's name matches
# neither keyword, which is why the replacement does not live here.


def test_phase_38_6_lock_path_uses_handoff_dot_autonomous_loop_dot_lock():
    """Path convention per masterplan criterion #1:
    running_flag_migrates_to_handoff_dot_autonomous_loop_dot_lock."""
    # Reset to the canonical (non-test-fixture) path for this assertion
    canonical = Path(cycle_lock.__file__).resolve().parents[2] / "handoff" / ".autonomous_loop.lock"
    # We can't directly read _LOCK_PATH because the autouse fixture
    # monkey-patched it; instead, verify the source code constant
    src = Path(cycle_lock.__file__).read_text()
    assert "_HANDOFF / \".autonomous_loop.lock\"" in src, (
        "cycle_lock.py must define _LOCK_PATH at handoff/.autonomous_loop.lock"
    )
