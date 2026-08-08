"""phase-85.5 (P0 SAFETY): cycle_lock split-brain via lockfile inode swap.

The defect, as proven in the audit basis and reproduced by the research gate:

1. `is_stale = (age > TTL) or (not pid_alive)` let an AGE heuristic condemn a
   lock the kernel said was live. Measured on a RUNNING cycle:
   `age_sec=5440, pid=89530, pid_alive=True, is_stale=True`.
2. `acquire()`'s stale path did `unlink -> reopen O_CREAT -> flock`, and
   unlink+recreate yields a NEW INODE whose flock contends with nobody -- so a
   second acquirer took the lock while the first still held it.
3. (found by the research gate, same class) the `finally` block unlinked
   BEFORE `LOCK_UN`, producing the same split-brain on the NORMAL release
   path with no TTL involved at all.

This module is named so the step's IMMUTABLE verification command --
`pytest backend/tests -k "cycle_lock or 85_5"` -- actually collects it. The
phase-38.6 regression suite matches neither keyword, which is why the
criterion-3 replacement test lives here rather than there.

Criteria 2 and 4 require REAL second processes, not mocks. Every contender
below is a genuine `subprocess`, and the criterion-2 fixture first
demonstrates the OLD algorithm acquiring, so the test cannot pass vacuously.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.services import cycle_lock as cl

_REPO = Path(cl.__file__).resolve().parents[2]


# ── helpers ──────────────────────────────────────────────────────────────

def _point_lock_at(monkeypatch, path: Path) -> Path:
    """Redirect the module's lock path so no test touches the real
    handoff/.autonomous_loop.lock (a live cycle uses that file)."""
    monkeypatch.setattr(cl, "_LOCK_PATH", path)
    monkeypatch.setattr(cl, "_HANDOFF", path.parent)
    return path


def _run_child(script: str, lock_path: Path, timeout: int = 30) -> str:
    """Run `script` in a REAL separate process against `lock_path`."""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        cwd=str(_REPO), capture_output=True, text=True, timeout=timeout,
    )
    return (proc.stdout or "").strip() + (
        f"\n[stderr] {proc.stderr.strip()}" if proc.returncode else ""
    )


# Contender using the CURRENT (fixed) acquire path.
_CHILD_NEW = """
import sys
from pathlib import Path
from backend.services import cycle_lock as cl
cl._LOCK_PATH = Path(sys.argv[1]); cl._HANDOFF = Path(sys.argv[1]).parent
try:
    with cl.acquire("child-cycle"):
        print("NEW_ACQUIRED")
except cl.CycleLockError:
    print("NEW_REFUSED")
"""

# Contender reproducing the OLD algorithm verbatim: on contention, unlink the
# lockfile and re-open it, which yields a new inode. This is the regression
# under test -- it MUST acquire, or the fixture is not a real live holder and
# the criterion-2 assertion would be vacuous.
_CHILD_OLD = """
import fcntl, os, sys
p = sys.argv[1]
fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("OLD_ACQUIRED_DIRECTLY_NO_CONTENTION")
except BlockingIOError:
    first = os.fstat(fd).st_ino
    os.unlink(p)                                   # <- the old stale path
    fd2 = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)   # succeeds: NEW inode
    print(f"OLD_ACQUIRED_VIA_UNLINK_RECREATE same_inode={os.fstat(fd2).st_ino == first}")
"""

# Holds the lock and sleeps, so the parent can kill it mid-hold (criterion 4).
_CHILD_HOLDER = """
import sys, time
from pathlib import Path
from backend.services import cycle_lock as cl
cl._LOCK_PATH = Path(sys.argv[1]); cl._HANDOFF = Path(sys.argv[1]).parent
with cl.acquire("holder-cycle"):
    print("HELD", flush=True)
    time.sleep(300)
"""


def _wait_for_held(path: Path, timeout: float = 20.0) -> dict:
    """Poll (never sleep-race) until the child has written a held payload."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = json.loads(path.read_text())
            if data.get("state") == "held" and data.get("pid"):
                return data
        except Exception:
            pass
        time.sleep(0.05)
    raise AssertionError(f"child never wrote a held lock at {path}")


# ── criterion 1 ──────────────────────────────────────────────────────────

def test_85_5_c1_live_holder_is_never_stale_however_old(tmp_path, monkeypatch):
    """A live process's lock is NEVER judged stale.

    Drives the EXACT measured condition from the audit basis: age > TTL with
    pid_alive True. Before the fix this returned is_stale True (age and
    liveness were OR'd); it must now be False.
    """
    lock = _point_lock_at(monkeypatch, tmp_path / ".autonomous_loop.lock")
    lock.write_text(json.dumps({
        "pid": os.getpid(),            # our own pid -> provably alive
        "cycle_id": "live-cycle",
        "started_at": "2026-08-08T00:00:00+00:00",
        "state": "held",
    }))
    old = time.time() - (cl.lock_ttl_sec() + 10_000)
    os.utime(lock, (old, old))

    state = cl.inspect_lock()
    # the measured condition really is reproduced
    assert state["age_sec"] > state["ttl_sec"], "fixture must exceed the TTL"
    assert state["pid_alive"] is True
    # age is still REPORTED -- it is useful signal ...
    assert state["age_exceeds_budget"] is True
    # ... but it may never condemn a live holder.
    assert state["is_stale"] is False, (
        "phase-85.5 regression: age alone condemned a LIVE cycle's lock"
    )


def test_85_5_c1_age_gates_nothing_destructive(tmp_path, monkeypatch):
    """The over-budget live lock must also survive clean_stale_lock()."""
    lock = _point_lock_at(monkeypatch, tmp_path / ".autonomous_loop.lock")
    lock.write_text(json.dumps({
        "pid": os.getpid(), "cycle_id": "live", "state": "held",
        "started_at": "2026-08-08T00:00:00+00:00",
    }))
    old = time.time() - (cl.lock_ttl_sec() + 10_000)
    os.utime(lock, (old, old))

    assert cl.clean_stale_lock(reason="test_over_budget_but_live") is None
    assert lock.exists(), "an over-budget but LIVE lock must not be unlinked"


# ── criterion 2 (two REAL processes; non-vacuous by construction) ─────────

def test_85_5_c2_second_real_process_refused_while_holder_alive(tmp_path, monkeypatch):
    """The unlink+recreate bypass is closed.

    Part A (non-vacuity, runs FIRST): against a genuinely-held lock, a real
    child running the OLD algorithm DOES acquire -- proving the fixture is a
    real live holder and the hazard reproduces here.

    Part B: against an identically-held lock, a real child running the CURRENT
    acquire path is REFUSED.
    """
    # ---- Part A: the OLD path acquires (so Part B cannot pass vacuously) ----
    path_a = _point_lock_at(monkeypatch, tmp_path / "a.lock")
    with cl.acquire("parent-cycle-a"):
        assert path_a.exists()
        out_old = _run_child(_CHILD_OLD, path_a)
    assert "OLD_ACQUIRED_VIA_UNLINK_RECREATE" in out_old, (
        f"fixture is not a real live holder -- old path did not contend: {out_old}"
    )
    assert "same_inode=False" in out_old, (
        "the old path must acquire a DIFFERENT inode -- that is the split brain"
    )

    # ---- Part B: the CURRENT path is refused against the same setup ----
    path_b = _point_lock_at(monkeypatch, tmp_path / "b.lock")
    with cl.acquire("parent-cycle-b"):
        assert path_b.exists()
        out_new = _run_child(_CHILD_NEW, path_b)
    assert out_new == "NEW_REFUSED", (
        f"a second REAL process acquired the cycle lock while a live holder "
        f"held it -- double-order hazard. child said: {out_new!r}"
    )


def test_85_5_c2_live_flock_holder_wins_even_when_payload_says_released(
    tmp_path, monkeypatch
):
    """The FLOCK is the authority, never the on-disk metadata.

    This is not contrived: release writes the `released` payload BEFORE
    LOCK_UN, so there is a genuine window in which the metadata reads
    collectable while the holder still owns the flock. Anything that consults
    is_stale and then unlinks+recreates would split the lock right there.

    MUTATION NOTE: this test exists because re-adding the stale
    unlink+recreate branch SURVIVED the test above -- with the fixed
    predicate a live holder is never is_stale, so that branch was unreachable
    in that scenario and the assertion could not distinguish the two. Here
    is_stale is forced True while the flock is genuinely held, which is the
    only state in which the deleted branch would fire.
    """
    lock = _point_lock_at(monkeypatch, tmp_path / ".autonomous_loop.lock")
    with cl.acquire("holder-cycle"):
        data = json.loads(lock.read_text())
        data["state"] = "released"          # forge the pre-LOCK_UN window
        lock.write_text(json.dumps(data))   # same inode: truncate, never unlink
        assert cl.inspect_lock()["is_stale"] is True, "fixture must look collectable"
        out = _run_child(_CHILD_NEW, lock)
    assert out == "NEW_REFUSED", (
        f"a contender took the lock from a LIVE flock holder because the "
        f"metadata said released -- the flock must outrank the payload. "
        f"child said: {out!r}"
    )


def test_85_5_c2_acquire_reverifies_inode_after_locking(tmp_path, monkeypatch):
    """Defence in depth (portalocker 4.0.0 / issue #115).

    If the lockfile is swapped between our open and our flock, our fd names an
    orphaned inode and our lock excludes nobody. acquire must detect that and
    retry against the current file rather than settle for the orphan.

    Driven by swapping the file from inside flock() itself, so the race is
    deterministic rather than timing-dependent.
    """
    lock = _point_lock_at(monkeypatch, tmp_path / ".autonomous_loop.lock")
    real_flock = cl.fcntl.flock
    calls = {"n": 0}

    def sneaky_flock(fd, op):
        result = real_flock(fd, op)
        calls["n"] += 1
        if calls["n"] == 1:
            # Someone replaces the lockfile the instant we lock it.
            lock.unlink(missing_ok=True)
            lock.write_text("{}")
        return result

    monkeypatch.setattr(cl.fcntl, "flock", sneaky_flock)

    with cl.acquire("verified-cycle"):
        # Our payload must be visible AT THE PATH -- if we had settled for the
        # orphan, the file here would still be the placeholder "{}".
        held = json.loads(lock.read_text())
        assert held["cycle_id"] == "verified-cycle", (
            "acquire settled for an orphaned inode -- its lock excludes nobody"
        )
    assert calls["n"] >= 2, (
        "acquire must have re-attempted after detecting the inode swap; "
        f"flock was called {calls['n']}x"
    )


# ── criterion 3 ──────────────────────────────────────────────────────────

class _FakeSettings:
    def __init__(self, budget: float) -> None:
        self.paper_cycle_max_seconds = budget


def test_85_5_c3_ttl_tracks_cycle_budget_and_never_falls_below_it(monkeypatch):
    """Replaces phase-38.6's `_LOCK_TTL_SEC == 5400` literal assertion.

    That literal IS the bug: the constant froze at 5400s while
    settings.paper_cycle_max_seconds moved to 7200s, so the TTL became 0.75x
    the budget and every cycle past 90 minutes advertised a stale lock while
    alive. Assert the RELATIONSHIP instead -- move the budget (it is
    operator-mutable at runtime, 300..21600) and the derived TTL must follow
    it and never drop below it. This fails if a future edit reintroduces a
    frozen constant or makes TTL < cycle timeout again.
    """
    import backend.config.settings as cfg

    for budget in (300.0, 1800.0, 7200.0, 21600.0):
        monkeypatch.setattr(cfg, "get_settings", lambda b=budget: _FakeSettings(b))
        assert cl.cycle_budget_sec() == budget, "TTL must READ the live budget"
        ttl = cl.lock_ttl_sec()
        assert ttl == pytest.approx(budget * cl._LOCK_TTL_MULTIPLE)
        assert ttl >= budget, (
            f"phase-85.5 regression: TTL {ttl} < cycle budget {budget}"
        )


def test_85_5_c3_no_frozen_ttl_constant_remains():
    """The frozen constant must be gone, not merely shadowed."""
    assert not hasattr(cl, "_LOCK_TTL_SEC"), (
        "_LOCK_TTL_SEC reintroduced -- the TTL must be derived at call time"
    )


# ── criterion 4 (real child killed mid-hold) ─────────────────────────────

def test_85_5_c4_dead_holder_is_still_recoverable(tmp_path, monkeypatch):
    """The fix must not trade a double-run hazard for a permanent deadlock.

    Spawns a REAL child that takes the lock, SIGKILLs it mid-hold, then
    asserts the lock is recognised as stale, cleaned, and re-acquirable.
    """
    lock = _point_lock_at(monkeypatch, tmp_path / ".autonomous_loop.lock")
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLDER, str(lock)],
        cwd=str(_REPO), stdout=subprocess.PIPE, text=True,
    )
    try:
        held = _wait_for_held(lock)
        assert held["pid"] == child.pid

        # While the child is ALIVE the lock must be respected ...
        alive_state = cl.inspect_lock()
        assert alive_state["pid_alive"] is True
        assert alive_state["is_stale"] is False
        assert cl.clean_stale_lock(reason="test_holder_alive") is None

        # ... and once it is genuinely DEAD, recovery must work.
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=15)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    dead_state = cl.inspect_lock()
    assert dead_state["pid_alive"] is False
    assert dead_state["is_stale"] is True, "a dead holder must be collectable"

    cleaned = cl.clean_stale_lock(reason="test_dead_holder_recovery")
    assert cleaned is not None
    assert not lock.exists(), "startup recovery must remove a dead holder's lock"

    # And a fresh cycle can run -- no deadlock.
    with cl.acquire("post-recovery-cycle"):
        assert json.loads(lock.read_text())["cycle_id"] == "post-recovery-cycle"


# ── criterion 5 (scope) ──────────────────────────────────────────────────

def test_85_5_c5_no_order_sizing_or_risk_logic_in_cycle_lock():
    """cycle_lock is a mutual-exclusion primitive and must stay one."""
    src = Path(cl.__file__).read_text()
    for forbidden in ("place_order", "quantity", "position_size", "risk_",
                      "submit_order", "paper_trader"):
        assert forbidden not in src, (
            f"cycle_lock.py must not contain trading logic; found {forbidden!r}"
        )
