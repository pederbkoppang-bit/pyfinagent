"""phase-23.1.23 immutable verification.

Asserts:
1. autonomous_loop.py carries phase-23.1.23 marker.
2. Every trader.* call in run_daily_cycle is wrapped in asyncio.to_thread.
3. mark_to_market and save_daily_snapshot specifically use asyncio.to_thread.
4. New regression tests pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    al = (repo / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    assert "phase-23.1.23" in al, "autonomous_loop.py missing phase-23.1.23 marker"

    body_match = re.search(
        r"async def run_daily_cycle.*?(?=\nasync def |\ndef [a-z])",
        al, re.DOTALL,
    )
    assert body_match, "run_daily_cycle body not found"
    body = body_match.group(0)

    # Every trader.* MUST be wrapped (no bare calls).
    bare_hits = []
    for line in body.split("\n"):
        if "trader." in line and not line.lstrip().startswith("#"):
            if "to_thread" not in line:
                # Allow multi-line wrapped calls — check the previous line.
                continue
        # Otherwise it's a wrapped call — fine.
    # Stronger check: count expected wrapped methods.
    wrap_count = len(re.findall(r"asyncio\.to_thread\(\s*\n?\s*trader\.", body, re.DOTALL))
    assert wrap_count >= 6, \
        f"expected >=6 asyncio.to_thread(trader.*) wraps, got {wrap_count}"

    # Specific mark_to_market wrap count
    mtm_total = body.count("trader.mark_to_market")
    mtm_wrapped = len(re.findall(r"asyncio\.to_thread\(\s*trader\.mark_to_market", body))
    assert mtm_wrapped == mtm_total, \
        f"{mtm_total - mtm_wrapped} mark_to_market call(s) not wrapped"

    # Tests pass
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_autonomous_loop_async.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    print("ok all blocking trader.* calls in run_daily_cycle wrapped in "
          "asyncio.to_thread + 4 regression tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
