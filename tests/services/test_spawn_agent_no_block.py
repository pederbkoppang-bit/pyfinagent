"""phase-23.1.21: regression guard against the asyncio-event-loop block bug.

The bug: `_spawn_real_agent` used `with ThreadPoolExecutor(max_workers=1):`
which calls `shutdown(wait=True)` on `__exit__`, blocking the asyncio caller
forever if the worker thread is stuck on a non-cancellable HTTP TCP read.

The fix: replace with `threading.Thread(target=_worker, daemon=True)` +
`thread.join(timeout=60)`. If the thread doesn't finish, abandon it (it
will be cleaned up at process exit) and raise a clear timeout exception.

These tests verify the daemon-thread pattern source-level (so they don't
take the full 60s real-time to run).
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest


def test_ticket_queue_processor_uses_daemon_thread_not_thread_pool():
    """Source-level regression guard. The fix must use threading.Thread(
    daemon=True) and join(timeout=60), NOT the old `with ThreadPoolExecutor`
    pattern that blocks on shutdown(wait=True)."""
    src = Path(__file__).resolve().parents[2] / "backend/services/ticket_queue_processor.py"
    text = src.read_text(encoding="utf-8")

    # New pattern present
    assert "threading.Thread(target=_worker, daemon=True" in text, \
        "_spawn_real_agent must use threading.Thread(daemon=True)"
    assert re.search(r"worker_thread\.join\(timeout=60\)", text), \
        "_spawn_real_agent must use thread.join(timeout=60), not future.result"
    assert "worker_thread.is_alive()" in text, \
        "_spawn_real_agent must check is_alive() to detect the timeout"
    assert "phase-23.1.21" in text, \
        "ticket_queue_processor.py must carry the phase-23.1.21 marker"

    # Old pattern absent — search _spawn_real_agent body specifically.
    # The function ends before _simulate_agent_response.
    spawn_match = re.search(
        r"def _spawn_real_agent.*?(?=\n    def _simulate_agent_response)",
        text, re.DOTALL,
    )
    assert spawn_match, "could not locate _spawn_real_agent body"
    body = spawn_match.group(0)
    # Keep the explanatory comment that mentions the old pattern; only
    # forbid the actual call (ThreadPoolExecutor( with parenthesis).
    assert "ThreadPoolExecutor(" not in body, \
        "remove ThreadPoolExecutor( call from _spawn_real_agent body (the hang bug)"


def test_daemon_thread_pattern_releases_caller_on_timeout():
    """Functional smoke test: the daemon-thread pattern must release its
    caller via `join(timeout)` even when the thread function is stuck.
    This is the principle that the fix relies on."""
    started = threading.Event()
    finished = threading.Event()

    def _stuck_worker():
        started.set()
        time.sleep(10)  # Far longer than the join timeout
        finished.set()

    thread = threading.Thread(target=_stuck_worker, daemon=True, name="stuck-test")
    thread.start()
    started.wait(2.0)

    t0 = time.monotonic()
    thread.join(timeout=0.5)  # Short timeout to keep test fast
    elapsed = time.monotonic() - t0

    # Caller is released after ~join timeout, regardless of thread state.
    assert elapsed < 1.0, f"join(timeout=0.5) took {elapsed:.2f}s — should be ~0.5s"
    assert thread.is_alive(), "thread should still be running (timeout fired)"
    assert not finished.is_set(), "stuck worker not yet finished — that's the point"
    # Thread will be cleaned up at process exit because daemon=True.


def test_main_py_registers_faulthandler_on_sigusr1():
    """phase-23.1.21 Fix C: backend/main.py lifespan must register faulthandler
    on SIGUSR1 so a future hang can be diagnosed without a kill."""
    src = Path(__file__).resolve().parents[2] / "backend/main.py"
    text = src.read_text(encoding="utf-8")
    assert "faulthandler.register" in text, \
        "main.py must register faulthandler"
    assert "_signal.SIGUSR1" in text or "signal.SIGUSR1" in text, \
        "main.py faulthandler must be on SIGUSR1"
    assert "all_threads=True" in text, \
        "faulthandler must dump all threads"
