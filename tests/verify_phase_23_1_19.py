"""phase-23.1.19 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. All 7 SQLite-touching files use closing(sqlite3.connect(...)) — zero
   bare `with sqlite3.connect(...)` patterns left.
2. tickets_db.py imports from contextlib import closing.
3. backend/main.py logs RLIMIT_NOFILE at lifespan startup.
4. New FD-leak regression test exists and passes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


SQLITE_FILES = [
    "backend/db/tickets_db.py",
    "backend/services/ticket_queue_processor.py",
    "backend/services/sla_monitor.py",
    "backend/services/response_delivery.py",
    "backend/services/stuck_task_reaper.py",
    "backend/slack_bot/commands.py",
    "backend/slack_bot/direct_responder.py",
]


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. No bare `with sqlite3.connect(` patterns remain across the 7 files.
    for rel in SQLITE_FILES:
        src = (repo / rel).read_text(encoding="utf-8")
        # Bare pattern is the leak; closing()-wrapped is fine.
        # Use a regex that matches `with sqlite3.connect(` NOT preceded by
        # `closing(`.
        bare_hits = re.findall(
            r"(?<!closing\()with\s+sqlite3\.connect\(",
            src,
        )
        assert not bare_hits, \
            f"{rel} still has bare `with sqlite3.connect(` (leak): {len(bare_hits)} occurrence(s)"
        # And there must be at least one closing()-wrapped occurrence per file.
        assert "with closing(sqlite3.connect" in src, \
            f"{rel} expected at least one closing(sqlite3.connect(...)) wrap"

    # 2. tickets_db.py imports closing.
    tdb = (repo / "backend/db/tickets_db.py").read_text(encoding="utf-8")
    assert "from contextlib import closing" in tdb, \
        "tickets_db.py must import closing from contextlib"

    # 3. main.py logs RLIMIT_NOFILE.
    main_src = (repo / "backend/main.py").read_text(encoding="utf-8")
    assert "RLIMIT_NOFILE" in main_src, \
        "backend/main.py must log RLIMIT_NOFILE at startup"
    assert "phase-23.1.19" in main_src, \
        "backend/main.py must carry the phase-23.1.19 marker comment"
    assert "soft=4096" in main_src or "_soft < 4096" in main_src, \
        "backend/main.py must WARN when soft limit < 4096"

    # 4. New regression test exists and passes.
    test_path = repo / "tests/db/test_tickets_db_no_fd_leak.py"
    assert test_path.exists(), \
        "tests/db/test_tickets_db_no_fd_leak.py must exist"
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/db/test_tickets_db_no_fd_leak.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"FD-leak test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert " passed" in result.stdout, \
        f"unexpected output: {result.stdout}"

    print("ok 23 sqlite3.connect sites wrapped with closing() across 7 files + "
          "tickets_db imports closing + main.py logs RLIMIT_NOFILE + "
          "FD-leak regression test passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
