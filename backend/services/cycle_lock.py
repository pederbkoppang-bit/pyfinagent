"""phase-38.6: restart-survivable autonomous-cycle lock.

Closes closure_roadmap.md section 3 OPEN-15: backend/services/autonomous_loop.py
has an in-process `_running` flag (line 78). When the backend is killed
mid-cycle (SIGKILL/crash/manual restart), the flag dies with the process,
leaving stale BQ state + risk of double-fire on next scheduled cron.

Fix: OS-level fcntl.flock + JSON pidfile at handoff/.autonomous_loop.lock.
- Kernel auto-releases the advisory lock on process death (all FDs closed).
- On-disk pidfile is cleaned by next startup via `clean_stale_lock`.
  (phase-85.5 changed WHEN: the "mtime > TTL" term is gone -- see below.)

Per researcher (handoff/current/research_brief_phase_38_6.md, 6 sources read
in full: flock(2) man page, Python fcntl docs, py-filelock, trbs/pid,
Improbable critique, Anthropic harness design).

phase-85.5 (P0 SAFETY) rewrote the authority model. Previously
`is_stale = (age > TTL) or (not pid_alive)` let an AGE HEURISTIC condemn a
lock the kernel said was live, and the remedy path then unlinked and
recreated the file -- yielding a new inode whose flock contended with
nobody, so two cycles could run at once on the money path. Now:

- Liveness (plus an explicitly recorded `released` state) is the sole
  authority over staleness; age may never condemn a lock.
- The stale-reacquire branch is deleted; contention simply raises.
- Acquire re-verifies (st_dev, st_ino) after locking (portalocker #115).
- Release rewrites the payload in place and NEVER unlinks.
- The TTL is derived at call time from settings.paper_cycle_max_seconds
  (7200s) and is used for OBSERVABILITY ONLY. The old constant was frozen
  at 5400s with a comment citing a 1800s budget that had long since moved.

Different timescale from cycle_health.py's 26h heartbeat alarm
(heartbeat = no-cycle-attempted; this lock = cycle-started-but-never-ended).

Stdlib-only (fcntl + os + json + pathlib); no new prod dep.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_HANDOFF = Path(__file__).resolve().parents[2] / "handoff"
_LOCK_PATH = _HANDOFF / ".autonomous_loop.lock"

# phase-85.5: the TTL is DERIVED from the live cycle budget, never frozen at
# import. The old `_LOCK_TTL_SEC = 90*60` carried the comment "1.5x
# paper_cycle_max_seconds (1800s)" while the budget in force had moved to
# 7200s -- so the TTL was 0.75x the budget and every cycle running past 90
# minutes advertised a stale lock while alive. The budget is operator-mutable
# at runtime (settings_api.py: 300 <= x <= 21600), so an import-time snapshot
# would reintroduce exactly that drift.
_LOCK_TTL_MULTIPLE = 1.5
_CYCLE_BUDGET_FALLBACK_SEC = 7200.0
_ACQUIRE_MAX_ATTEMPTS = 5


class CycleLockError(Exception):
    """Raised when the lock can't be acquired (held by a live cycle)."""


def cycle_budget_sec() -> float:
    """The wall-clock budget for one cycle, read LAZILY from settings.

    Same field autonomous_loop.py uses for its asyncio.timeout, so the lock
    and the cycle read the same number by construction -- that drift is what
    produced this bug. NB: autonomous_loop's own fallback literal is a stale
    1800.0; do not copy it.
    """
    try:
        from backend.config.settings import get_settings

        value = float(getattr(get_settings(), "paper_cycle_max_seconds",
                              _CYCLE_BUDGET_FALLBACK_SEC))
        return value if value > 0 else _CYCLE_BUDGET_FALLBACK_SEC
    except Exception:  # settings unavailable -> observability degrades, never blocks
        return _CYCLE_BUDGET_FALLBACK_SEC


def lock_ttl_sec() -> float:
    """Budget-overrun threshold, derived at call time.

    OBSERVABILITY ONLY (phase-85.5). This value gates NOTHING: it may not
    condemn a lock, unlink a file, or refuse an acquire. "This cycle has
    exceeded its budget" is a true and useful statement; it is simply not the
    same statement as "this lock is stale". Keeping a TTL that can still
    condemn a lock would rename the bug rather than fix it.
    """
    return cycle_budget_sec() * _LOCK_TTL_MULTIPLE


def _is_pid_alive(pid: int) -> bool:
    """True if pid corresponds to a running process. POSIX-only;
    PermissionError means alive-but-not-ours (e.g. root)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_payload(fd: int, payload: dict) -> None:
    """Truncate-and-rewrite the pidfile in place through our own fd.

    phase-85.5: in-place rewrite (never unlink+recreate) so the inode other
    processes coordinate on by path stays the one we hold the flock on.
    """
    os.lseek(fd, 0, 0)
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(payload).encode("utf-8"))
    os.fsync(fd)


def inspect_lock() -> Optional[dict]:
    """Return on-disk lock state + derived age_sec/is_stale/pid_alive.

    phase-85.5: also returns `state` ("held"/"released"), `ttl_sec` and
    `age_exceeds_budget`. Age is OBSERVABILITY ONLY and gates nothing.
    """
    if not _LOCK_PATH.exists():
        return None
    try:
        data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("cycle_lock: malformed lock (%r); treating as stale.", exc)
        return {"is_stale": True, "raw_error": str(exc)}
    try:
        age_sec = time.time() - _LOCK_PATH.stat().st_mtime
    except OSError:
        age_sec = float("inf")
    pid = int(data.get("pid", 0) or 0)
    pid_alive = _is_pid_alive(pid) if pid else False
    ttl_sec = lock_ttl_sec()
    data["age_sec"] = age_sec
    data["pid_alive"] = pid_alive
    data["ttl_sec"] = ttl_sec
    # phase-85.5: age is reported for observability and gates NOTHING.
    data["age_exceeds_budget"] = age_sec > ttl_sec
    # phase-85.5: liveness is the sole authority over staleness. The kernel
    # releases an advisory lock the instant the holder dies, so a live holder
    # is live no matter how old its lock is; age may never condemn it.
    # `state == "released"` is the one non-liveness term, and it is a FACT the
    # releasing process recorded, not a heuristic: the backend is long-lived,
    # so after a cycle releases, its pid is still alive and liveness alone
    # would report a finished cycle as running forever.
    data["state"] = data.get("state") or "held"
    data["is_stale"] = (data["state"] == "released") or (not pid_alive)
    return data


def clean_stale_lock(reason: str = "stale_on_startup") -> Optional[dict]:
    """Remove a collectable lockfile. The ONLY unlink site in this module.

    phase-85.5: "collectable" no longer means "mtime > TTL OR pid dead" -- age
    is not a term at all. A lock is collectable when its holder pid is dead or
    the holder explicitly recorded `state == "released"`.

    Two independent conditions must hold before anything is removed:
      1. `is_stale` says the lock is collectable, AND
      2. WE win the flock -- proving no live process holds it.

    (2) is what makes this race-free. `is_stale` alone is not sufficient: a
    live holder is briefly indistinguishable from a malformed lock while
    `_write_payload` has the file truncated, and unlinking there would destroy
    a live holder's lockfile and split the lock across two inodes.

    Returns the inspected state (so the caller can log/alert), or None when
    there is no lock, the lock is live, or a live process holds the flock.
    """
    state = inspect_lock()
    if not state or not state.get("is_stale"):
        return None
    # phase-85.5: this is the ONE remaining unlink site, and it is gated on the
    # STRONG test rather than on the is_stale heuristic alone -- only a process
    # that has just WON the flock may remove the file. Race-free by
    # construction: the winner is by definition the only one there. Previously
    # this unlinked on the predicate alone, which is how a LIVE cycle's lock
    # could be destroyed out from under it.
    try:
        fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        logger.warning("cycle_lock: open for clean failed (%r) -- leaving lock.", exc)
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "cycle_lock: NOT cleaning -- a live process holds the flock "
                "(reason=%s pid=%s).", reason, state.get("pid"),
            )
            return None
        # Re-inspect under the lock: the world may have moved between the
        # first inspect and winning the flock.
        state = inspect_lock() or state
        if not state.get("is_stale"):
            return None
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("cycle_lock: unlink failed (%r) -- proceeding anyway.", exc)
        logger.warning(
            "cycle_lock: cleaned stale lock (reason=%s pid=%s age_sec=%.0f state=%s).",
            reason, state.get("pid"), state.get("age_sec", 0.0), state.get("state"),
        )
        return state
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass


@contextmanager
def acquire(cycle_id: str) -> Iterator[None]:
    """Race-free acquire: O_RDWR|O_CREAT, flock LOCK_EX|LOCK_NB, then VERIFY.

    Yields after writing {pid, cycle_id, started_at, state: "held"}.

    phase-85.5: on exit this NO LONGER unlinks. It rewrites the payload in
    place as {state: "released"} and then releases the flock. The old order
    (unlink, then LOCK_UN) left a window in which this process was alive and
    still holding the flock while the path was free, so a second acquirer
    created a new inode and both believed they held the cycle lock.

    Contention raises CycleLockError -- there is no "stale, so take it"
    branch. A live flock holder is live by kernel guarantee, and a dead
    holder cannot be holding a flock, so that branch's premise was empty.

    After locking, the fd's (st_dev, st_ino) is compared against the path's;
    a mismatch means the file was swapped under us and our lock excludes
    nobody, so we retry (bounded by _ACQUIRE_MAX_ATTEMPTS).

    On SIGKILL / crash mid-cycle the kernel auto-releases the flock when the
    process dies. The pidfile remains with a now-dead pid; the next startup's
    `clean_stale_lock` collects it. A leftover `released` pidfile does not
    block a later acquire -- it is reopened and reused, same inode.
    """
    _HANDOFF.mkdir(parents=True, exist_ok=True)

    # phase-85.5: the stale-reacquire branch is DELETED. On BlockingIOError we
    # close and raise -- full stop. A live flock holder means a live cycle by
    # kernel guarantee, and a dead holder cannot be holding a flock, so the old
    # branch's premise ("held, but stale, so take it") was empty. Taking it
    # required unlink+reopen, which produced a NEW INODE whose flock did not
    # contend with the one the live process still held -- i.e. split-brain.
    for _attempt in range(_ACQUIRE_MAX_ATTEMPTS):
        fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = inspect_lock()
            os.close(fd)
            raise CycleLockError(
                f"another live cycle holds the lock: "
                f"pid={state and state.get('pid')} "
                f"age_sec={state and state.get('age_sec')}"
            )
        # phase-85.5: verify-after-lock (portalocker 4.0.0 / issue #115). We
        # may have locked an inode that is no longer the one at this path --
        # e.g. someone unlinked and recreated it between our open and our
        # flock. Holding a lock on an orphaned inode excludes nobody, so
        # compare identity and retry (bounded) if it moved.
        st_fd = os.fstat(fd)
        try:
            st_path = os.stat(_LOCK_PATH)
        except FileNotFoundError:
            os.close(fd)
            continue
        if (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino):
            break
        os.close(fd)
    else:
        raise CycleLockError(
            f"lockfile {_LOCK_PATH} kept being replaced under us after "
            f"{_ACQUIRE_MAX_ATTEMPTS} attempts -- refusing to run a cycle "
            f"on a lock that excludes nobody"
        )

    # NB: every failure path above raises BEFORE this try, so a failed acquire
    # can never touch a live holder's pidfile or flock. That is phase-69.1
    # audit item 4 enforced by control flow rather than by an `acquired` flag.
    try:
        _write_payload(fd, {
            "pid": os.getpid(),
            "cycle_id": cycle_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "state": "held",
        })
        yield
    finally:
        # phase-85.5: NEVER unlink here. The old code unlinked at this point
        # and released the flock AFTERWARDS, so during that window the process
        # was alive and still holding the flock while the path was free -- a
        # second acquirer created a new inode and both believed they held the
        # cycle lock, on the NORMAL release path with no TTL involved. We
        # instead record a `released` state and let the next acquirer reuse the
        # same inode, or startup recovery remove it.
        try:
            _write_payload(fd, {
                "pid": os.getpid(),
                "cycle_id": cycle_id,
                "released_at": datetime.now(timezone.utc).isoformat(),
                "state": "released",
            })
        except Exception as exc:
            logger.warning("cycle_lock: released-payload write failed (%r).", exc)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception as exc:
            logger.warning("cycle_lock: release failed (%r).", exc)
        try:
            os.close(fd)
        except Exception:
            pass
