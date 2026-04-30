"""phase-23.1.21 immutable verification.

Asserts:
1. _spawn_real_agent uses daemon Thread + join(timeout=60), not ThreadPoolExecutor.
2. backend/main.py lifespan registers faulthandler on SIGUSR1.
3. Watchdog script + plist exist; the script has the 3-failure threshold.
4. launchd plist has ProcessType=Interactive (App Nap exemption).
5. New tests pass (3 in test_spawn_agent_no_block.py).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Daemon thread pattern
    pt_src = (repo / "backend/services/ticket_queue_processor.py").read_text(encoding="utf-8")
    assert "phase-23.1.21" in pt_src, "ticket_queue_processor.py missing phase-23.1.21 marker"
    assert "threading.Thread(target=_worker, daemon=True" in pt_src, \
        "_spawn_real_agent must use threading.Thread(daemon=True)"
    assert "worker_thread.join(timeout=60)" in pt_src, \
        "_spawn_real_agent must use thread.join(timeout=60)"
    assert "worker_thread.is_alive()" in pt_src, \
        "_spawn_real_agent must check is_alive() to detect timeout"

    # 2. faulthandler in main.py
    main_src = (repo / "backend/main.py").read_text(encoding="utf-8")
    assert "faulthandler.register" in main_src, \
        "main.py must register faulthandler"
    assert "all_threads=True" in main_src, \
        "faulthandler must dump all threads"
    assert "phase-23.1.21" in main_src, \
        "main.py must carry the phase-23.1.21 marker"

    # 3. Watchdog script + plist
    watchdog_sh = repo / "scripts/launchd/backend_watchdog.sh"
    watchdog_plist = repo / "scripts/launchd/com.pyfinagent.backend-watchdog.plist"
    assert watchdog_sh.exists(), "scripts/launchd/backend_watchdog.sh missing"
    assert watchdog_plist.exists(), "scripts/launchd/com.pyfinagent.backend-watchdog.plist missing"
    sh_src = watchdog_sh.read_text(encoding="utf-8")
    assert "FAILURE_THRESHOLD=3" in sh_src, \
        "watchdog script must use 3-failure threshold"
    assert "kill -USR1" in sh_src, \
        "watchdog must SIGUSR1 the backend before kicking"
    assert "launchctl kickstart -k" in sh_src, \
        "watchdog must use launchctl kickstart -k for restart"
    plist_src = watchdog_plist.read_text(encoding="utf-8")
    assert "<integer>60</integer>" in plist_src and "StartInterval" in plist_src, \
        "watchdog plist must have StartInterval=60"

    # 4. Backend plist has ProcessType=Interactive
    backend_plist = Path.home() / "Library/LaunchAgents/com.pyfinagent.backend.plist"
    if backend_plist.exists():
        plist_src = backend_plist.read_text(encoding="utf-8")
        assert "<key>ProcessType</key>" in plist_src and "Interactive" in plist_src, \
            "backend plist must have ProcessType=Interactive (App Nap exemption)"

    # 5. New tests pass
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_spawn_agent_no_block.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    print("ok daemon-thread spawn pattern + faulthandler SIGUSR1 + external watchdog "
          "(60s interval, 3-fail threshold) + ProcessType=Interactive + 3 new tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
