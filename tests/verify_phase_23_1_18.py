"""phase-23.1.18 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. save_paper_snapshot uses MERGE on snapshot_date (Fix A).
2. red-line query uses MAX(total_nav), not ANY_VALUE (Fix C).
3. cleanup_phase_23_1_18.py exists with --dry-run + --apply modes (Fix B).
4. New snapshot upsert tests pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. save_paper_snapshot MERGE
    bq_src = (repo / "backend/db/bigquery_client.py").read_text(encoding="utf-8")
    assert re.search(
        r"def save_paper_snapshot\(self.*?MERGE\s+`",
        bq_src, re.DOTALL,
    ), "save_paper_snapshot must issue a MERGE statement"
    assert "ON T.snapshot_date = S.snapshot_date" in bq_src, \
        "save_paper_snapshot MERGE must key on snapshot_date"
    assert "requires 'snapshot_date' field for MERGE key" in bq_src, \
        "save_paper_snapshot must guard on snapshot_date presence"

    # 2. Red-line query uses MAX(total_nav)
    sov_src = (repo / "backend/api/sovereign_api.py").read_text(encoding="utf-8")
    assert "MAX(total_nav) AS nav" in sov_src, \
        "sovereign_api _fetch_snapshots must aggregate with MAX(total_nav)"
    assert "ANY_VALUE(total_nav)" not in sov_src, \
        "remove the ANY_VALUE legacy aggregation"
    assert "phase-23.1.18" in sov_src, \
        "sovereign_api must carry the phase-23.1.18 marker comment"

    # 3. Cleanup script
    cleanup_path = repo / "scripts/cleanup_phase_23_1_18.py"
    assert cleanup_path.exists(), "scripts/cleanup_phase_23_1_18.py missing"
    cleanup_src = cleanup_path.read_text(encoding="utf-8")
    assert "--apply" in cleanup_src and "args.apply" in cleanup_src, \
        "cleanup script must require explicit --apply"
    assert "ROW_NUMBER() OVER" in cleanup_src and \
           "PARTITION BY snapshot_date" in cleanup_src, \
        "cleanup script must use ROW_NUMBER() PARTITION BY snapshot_date"
    assert "ORDER BY total_nav DESC" in cleanup_src, \
        "cleanup script tie-breaker must be total_nav DESC"

    # 4. Tests
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_snapshot_upsert.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert " passed" in result.stdout, f"unexpected output: {result.stdout}"

    print("ok save_paper_snapshot MERGE upsert + red-line MAX(total_nav) query + "
          "cleanup script (dry-run/apply with ROW_NUMBER PARTITION BY) + 3 new tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
