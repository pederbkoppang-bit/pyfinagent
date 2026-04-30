"""phase-23.1.22: regression guard against the kill_switch reentrant-lock deadlock.

The bug: KillSwitchState.pause() and resume() acquired self._lock and then
called self.snapshot() which tried to re-acquire the SAME (non-reentrant)
threading.Lock — instant process-wide deadlock that hung the asyncio event
loop. Caught via faulthandler SIGUSR1 dump on a live hung backend.

The fix: extracted _snapshot_locked() helper. pause()/resume() call it
directly without re-entering the lock. The public snapshot() still takes
the lock for external callers.
"""

from __future__ import annotations

import threading
import time

import pytest

from backend.services.kill_switch import KillSwitchState


def test_pause_does_not_deadlock_on_self_lock():
    """pause() must NOT deadlock when calling its own snapshot helper."""
    state = KillSwitchState()
    completed = threading.Event()
    result_holder = {}

    def _runner():
        try:
            result_holder["value"] = state.pause(trigger="test")
        finally:
            completed.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    completed.wait(timeout=2.0)
    assert completed.is_set(), \
        "pause() deadlocked — it should return within milliseconds"
    assert result_holder["value"]["paused"] is True


def test_resume_does_not_deadlock_on_self_lock():
    """resume() must NOT deadlock when calling its own snapshot helper."""
    state = KillSwitchState()
    state.pause(trigger="test-pre")  # ensure it's paused
    completed = threading.Event()
    result_holder = {}

    def _runner():
        try:
            result_holder["value"] = state.resume(trigger="test")
        finally:
            completed.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    completed.wait(timeout=2.0)
    assert completed.is_set(), \
        "resume() deadlocked — it should return within milliseconds"
    assert result_holder["value"]["paused"] is False


def test_pause_resume_cycle_is_fast():
    """A pause -> resume -> pause cycle must complete in well under 1 second."""
    state = KillSwitchState()
    t0 = time.monotonic()
    state.pause(trigger="bench-1")
    state.resume(trigger="bench-2")
    state.pause(trigger="bench-3")
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"3-step pause/resume cycle took {elapsed:.2f}s — should be <1s"


def test_snapshot_locked_helper_present():
    """Source-level guard: pause/resume must use _snapshot_locked, not snapshot()."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "backend/services/kill_switch.py"
    text = src.read_text(encoding="utf-8")
    assert "_snapshot_locked" in text, \
        "kill_switch.py must define _snapshot_locked() helper"
    assert "phase-23.1.22" in text, \
        "kill_switch.py must carry the phase-23.1.22 marker"
    # The pause/resume bodies must call _snapshot_locked (not snapshot())
    # to avoid re-entering self._lock.
    import re
    pause_body = re.search(r"def pause\(self.*?def resume", text, re.DOTALL)
    assert pause_body and "self._snapshot_locked()" in pause_body.group(0), \
        "pause() must call self._snapshot_locked() not self.snapshot()"
    resume_body = re.search(r"def resume\(self.*?def update_sod_nav", text, re.DOTALL)
    assert resume_body and "self._snapshot_locked()" in resume_body.group(0), \
        "resume() must call self._snapshot_locked() not self.snapshot()"
