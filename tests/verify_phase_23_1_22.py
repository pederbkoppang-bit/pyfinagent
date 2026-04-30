"""phase-23.1.22 immutable verification — covers consolidated 23.1.20+21+22.

Asserts:
1. kill_switch.py has _snapshot_locked helper; pause/resume use it.
2. ticket_queue_processor uses daemon thread (phase-23.1.21).
3. main.py registers faulthandler (phase-23.1.21).
4. paper_trading.py has asyncio.timeout(5) on resume + kill-switch (phase-23.1.20).
5. bigquery_client.get_paper_portfolio enforces 30s timeout (phase-23.1.20).
6. Watchdog script + plist exist (phase-23.1.21).
7. All 10 new tests pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Deadlock fix (phase-23.1.22)
    ks = (repo / "backend/services/kill_switch.py").read_text(encoding="utf-8")
    assert "phase-23.1.22" in ks, "kill_switch.py missing phase-23.1.22 marker"
    assert "_snapshot_locked" in ks, "_snapshot_locked helper missing"
    pause_body = re.search(r"def pause\(self.*?def resume", ks, re.DOTALL)
    resume_body = re.search(r"def resume\(self.*?def update_sod_nav", ks, re.DOTALL)
    assert pause_body and "self._snapshot_locked()" in pause_body.group(0), \
        "pause() must call _snapshot_locked, not snapshot()"
    assert resume_body and "self._snapshot_locked()" in resume_body.group(0), \
        "resume() must call _snapshot_locked, not snapshot()"

    # 2. Daemon thread pattern (phase-23.1.21)
    pt = (repo / "backend/services/ticket_queue_processor.py").read_text(encoding="utf-8")
    assert "threading.Thread(target=_worker, daemon=True" in pt, \
        "_spawn_real_agent must use daemon Thread"
    assert "worker_thread.join(timeout=60)" in pt

    # 3. faulthandler (phase-23.1.21)
    main_src = (repo / "backend/main.py").read_text(encoding="utf-8")
    assert "faulthandler.register" in main_src and "all_threads=True" in main_src

    # 4. asyncio.timeout(5) (phase-23.1.20)
    api_src = (repo / "backend/api/paper_trading.py").read_text(encoding="utf-8")
    assert "phase-23.1.20" in api_src
    assert api_src.count("async with asyncio.timeout(5)") >= 2, \
        "paper_trading.py must wrap resume + kill-switch BQ calls in asyncio.timeout(5)"
    assert 'Retry-After' in api_src

    # 5. BQ 30s timeout (phase-23.1.20)
    bq_src = (repo / "backend/db/bigquery_client.py").read_text(encoding="utf-8")
    assert "result(timeout=30)" in bq_src, \
        "get_paper_portfolio must enforce 30s BQ timeout"

    # 6. Watchdog
    assert (repo / "scripts/launchd/backend_watchdog.sh").exists()
    assert (repo / "scripts/launchd/com.pyfinagent.backend-watchdog.plist").exists()

    # 7. Tests pass
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_kill_switch_no_deadlock.py",
         "tests/services/test_spawn_agent_no_block.py",
         "tests/api/test_pause_resume_timeout.py",
         "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    print("ok kill_switch deadlock fix (_snapshot_locked) + daemon-thread spawn + "
          "faulthandler SIGUSR1 + asyncio.timeout(5) + BQ result(timeout=30) + "
          "watchdog plist + 10 new tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
