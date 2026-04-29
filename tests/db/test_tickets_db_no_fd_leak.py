"""phase-23.1.19: regression guard against the SQLite FD leak.

`with sqlite3.connect(...) as conn:` only commits/rolls back the
transaction; it does NOT close the connection. The fix is to wrap with
`contextlib.closing()`. This test exercises common TicketsDB methods
in a tight loop and asserts the process FD count doesn't grow.
"""

from __future__ import annotations

import os
import sys

import pytest

psutil = pytest.importorskip("psutil")

from backend.db.tickets_db import (
    TicketsDB,
    TicketSource,
    TicketStatus,
    TicketPriority,
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="psutil.Process().num_fds() is UNIX-only",
)
def test_no_fd_leak_over_100_iterations(tmp_path):
    """100 round-trips on TicketsDB methods must not grow process FDs."""
    db_file = tmp_path / "leak_test.db"
    db = TicketsDB(str(db_file))
    proc = psutil.Process(os.getpid())

    # Warm-up: a handful of calls so any one-time allocation (e.g., the
    # WAL/SHM files for SQLite) lands before we sample baseline.
    for _ in range(5):
        db.get_open_tickets()
    fds_before = proc.num_fds()

    for i in range(100):
        tid = db.create_ticket(
            source=TicketSource.SLACK,
            sender_id=f"U{i:08d}",
            message_text=f"phase-23.1.19 leak test msg {i}",
            priority=TicketPriority.P2,
        )
        db.get_open_tickets()
        db.update_ticket_status(
            tid, TicketStatus.RESOLVED, response_text="ok",
        )
        db.get_ticket_stats()

    fds_after = proc.num_fds()
    delta = fds_after - fds_before

    if delta > 5:
        # Diagnostic: list paths still open against the test DB.
        leaked = [
            f.path for f in proc.open_files()
            if "leak_test" in f.path or str(tmp_path) in f.path
        ]
        pytest.fail(
            f"FD leak: {delta} net FDs after 100 iterations "
            f"(before={fds_before}, after={fds_after}). "
            f"Open paths: {leaked}"
        )
