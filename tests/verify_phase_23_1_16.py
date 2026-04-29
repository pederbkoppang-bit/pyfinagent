"""phase-23.1.16 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. _fetch_ticker_meta uses ThreadPoolExecutor with max_workers=5 (parallel yfinance).
2. The serial sleep(0.3) is gone.
3. /ticker-meta route uses per-ticker cache keys.
4. main.py lifespan registers the prewarm task.
5. New ticker-meta perf tests pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1 + 2. Parallel yfinance, serial sleep removed
    pt_src = (repo / "backend/api/paper_trading.py").read_text(encoding="utf-8")
    assert "phase-23.1.16" in pt_src, "paper_trading.py missing phase-23.1.16 marker"
    assert re.search(
        r"ThreadPoolExecutor\(max_workers=5\)",
        pt_src,
    ), "_fetch_ticker_meta must use ThreadPoolExecutor(max_workers=5)"
    assert "as_completed" in pt_src, "_fetch_ticker_meta must use as_completed"
    # The old serial loop with time.sleep(0.3) must be gone
    assert "time.sleep(0.3)" not in pt_src, \
        "remove time.sleep(0.3) — replaced by ThreadPoolExecutor parallelism"

    # 3. Per-ticker cache keys in /ticker-meta route
    assert "paper:ticker_meta:single:" in pt_src, \
        "/ticker-meta route must use per-ticker cache key shape"
    # The legacy set-level lookup must be gone from the route handler.
    assert 'cache_key = f"paper:ticker_meta:{' not in pt_src, \
        "remove the legacy set-level cache_key — replaced by per-ticker"

    # 4. Prewarm task in main.py lifespan
    main_src = (repo / "backend/main.py").read_text(encoding="utf-8")
    assert "_prewarm_ticker_meta" in main_src, \
        "main.py lifespan must register _prewarm_ticker_meta task"
    assert "asyncio.create_task(_prewarm_ticker_meta())" in main_src, \
        "main.py must fire _prewarm_ticker_meta via asyncio.create_task"
    assert "Prewarming ticker-meta cache" in main_src, \
        "main.py prewarm must emit the 'Prewarming ticker-meta cache' info log"

    # 5. New tests
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/api/test_ticker_meta_perf.py",
         "tests/api/test_ticker_meta.py",
         "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert " passed" in result.stdout, f"unexpected output: {result.stdout}"

    print("ok ThreadPoolExecutor parallel yfinance + per-ticker cache keys + "
          "lifespan prewarm + 4 new perf tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
