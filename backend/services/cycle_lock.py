"""phase-38.6: restart-survivable autonomous-cycle lock.

Closes closure_roadmap.md section 3 OPEN-15: backend/services/autonomous_loop.py
has an in-process `_running` flag (line 78). When the backend is killed
mid-cycle (SIGKILL/crash/manual restart), the flag dies with the process,
leaving stale BQ state + risk of double-fire on next scheduled cron.

Fix: OS-level fcntl.flock + JSON pidfile at handoff/.autonomous_loop.lock.
- Kernel auto-releases the advisory lock on process death (all FDs closed).
- On-disk pidfile is cleaned by next startup via `clean_stale_lock` if
  mtime > TTL OR the recorded pid is dead.

Per researcher (handoff/current/research_brief_phase_38_6.md, 6 sources read
in full: flock(2) man page, Python fcntl docs, py-filelock, trbs/pid,
Improbable critique, Anthropic harness design).

TTL = 90 minutes = 1.5x settings.paper_cycle_max_seconds (1800s) with
headroom for finally-block teardown + cron jitter. Different timescale
from cycle_health.py's 26h heartbeat alarm (heartbeat = no-cycle-attempted;
this lock = cycle-started-but-never-ended).

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
_LOCK_TTL_SEC = 90 * 60  # 1.5x paper_cycle_max_seconds (1800s)


class CycleLockError(Exception):
    """Raised when the lock can't be acquired (held by a live cycle)."""


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


def inspect_lock() -> Optional[dict]:
    """Return on-disk lock state + derived age_sec/is_stale/pid_alive."""
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
    data["age_sec"] = age_sec
    data["pid_alive"] = pid_alive
    data["is_stale"] = (age_sec > _LOCK_TTL_SEC) or (not pid_alive)
    return data


def clean_stale_lock(reason: str = "stale_on_startup") -> Optional[dict]:
    """Unlink the lockfile if it's stale (mtime > TTL OR pid dead).
    Returns the inspected state (so caller can log/alert), or None if no
    lock or live."""
    state = inspect_lock()
    if not state or not state.get("is_stale"):
        return None
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("cycle_lock: unlink failed (%r) -- proceeding anyway.", exc)
    logger.warning(
        "cycle_lock: cleaned stale lock (reason=%s pid=%s age_sec=%.0f).",
        reason, state.get("pid"), state.get("age_sec", 0.0),
    )
    return state


@contextmanager
def acquire(cycle_id: str) -> Iterator[None]:
    """Race-free acquire: O_RDWR|O_CREAT, then flock LOCK_EX|LOCK_NB.

    Yields after writing {pid, cycle_id, started_at} to the pidfile.
    On exit (normal OR exception), unlinks the file + releases the flock.

    On SIGKILL / crash mid-cycle: the flock is auto-released by the kernel
    when the process dies (all FDs closed). The pidfile remains on disk
    with a now-dead pid, which the next startup's `clean_stale_lock` will
    detect and unlink.
    """
    _HANDOFF.mkdir(parents=True, exist_ok=True)
    fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False  # phase-69.1: did WE take the flock? (audit item 4)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = inspect_lock()
            if state and state.get("is_stale"):
                clean_stale_lock(reason="cycle_entry_contention_but_stale")
                os.close(fd)
                fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                os.close(fd)
                raise CycleLockError(
                    f"another live cycle holds the lock: "
                    f"pid={state and state.get('pid')} "
                    f"age_sec={state and state.get('age_sec')}"
                )
        # phase-69.1: we now hold the flock (directly or via stale-reacquire).
        # Only past this point may the finally clean up the lockfile.
        acquired = True
        payload = json.dumps({
            "pid": os.getpid(),
            "cycle_id": cycle_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        os.lseek(fd, 0, 0)
        os.ftruncate(fd, 0)
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
        yield
    finally:
        # phase-69.1 (audit item 4): a FAILED acquire (contention with a LIVE
        # cycle) must NOT unlink the live holder's pidfile or release its flock.
        # Guard cleanup on `acquired` (Python contextlib acquire-then-guard
        # pattern) so we only clean up the lock WE actually took.
        if acquired:
            try:
                _LOCK_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception as exc:
                logger.warning("cycle_lock: release failed (%r).", exc)
        try:
            os.close(fd)
        except Exception:
            pass
