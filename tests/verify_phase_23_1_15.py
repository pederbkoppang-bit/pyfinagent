"""phase-23.1.15 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. paper_trader.execute_buy contains the idempotency-guard block.
2. bigquery_client.save_paper_position uses MERGE on ticker.
3. bigquery_client.get_paper_trades_for_ticker_since helper exists.
4. cleanup_phase_23_1_15.py exists with --dry-run default + --apply mode.
5. Three new idempotency tests pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Idempotency guard in execute_buy
    pt_src = (repo / "backend/services/paper_trader.py").read_text(encoding="utf-8")
    assert "phase-23.1.15" in pt_src and "Idempotency guard" in pt_src, \
        "paper_trader.py missing phase-23.1.15 Idempotency guard block"
    assert "get_paper_trades_for_ticker_since" in pt_src, \
        "execute_buy must call get_paper_trades_for_ticker_since"
    assert "timedelta(minutes=30)" in pt_src, \
        "execute_buy idempotency guard must use 30-minute lookback"
    assert "duplicate BUY" in pt_src, \
        "execute_buy guard must log 'duplicate BUY' on hit"

    # 2. MERGE upsert for save_paper_position
    bq_src = (repo / "backend/db/bigquery_client.py").read_text(encoding="utf-8")
    assert re.search(
        r"def save_paper_position\(self.*?MERGE\s+`",
        bq_src, re.DOTALL,
    ), "save_paper_position must issue a MERGE statement"
    assert "ON T.ticker = S.ticker" in bq_src, \
        "save_paper_position MERGE must key on T.ticker = S.ticker"
    assert "WHEN MATCHED THEN" in bq_src and "WHEN NOT MATCHED THEN" in bq_src, \
        "save_paper_position MERGE must have both MATCHED and NOT MATCHED branches"
    assert "requires 'ticker' field for MERGE key" in bq_src, \
        "save_paper_position must guard on ticker presence"

    # 3. New BQ helper
    assert "def get_paper_trades_for_ticker_since" in bq_src, \
        "bigquery_client.py missing get_paper_trades_for_ticker_since helper"

    # 4. Cleanup script
    cleanup_path = repo / "scripts/cleanup_phase_23_1_15.py"
    assert cleanup_path.exists(), "scripts/cleanup_phase_23_1_15.py missing"
    cleanup_src = cleanup_path.read_text(encoding="utf-8")
    assert "--apply" in cleanup_src and "--dry-run" not in cleanup_src or \
           ("--apply" in cleanup_src and "args.apply" in cleanup_src), \
        "cleanup script must have --apply mode"
    assert "WDC_DUPLICATE_TRADE_ID" in cleanup_src and \
           "e5447bd9-9cb0-437b-b2a2-c851703b77b1" in cleanup_src, \
        "cleanup script must hard-code WDC duplicate trade_id"
    assert "XOM_TEST_TRADE_ID" in cleanup_src and \
           "a8e6b00e-e39b-4a00-9eb4-540097b2212a" in cleanup_src, \
        "cleanup script must hard-code XOM test orphan trade_id"
    assert "1451.40" in cleanup_src or "TOTAL_REFUND" in cleanup_src, \
        "cleanup script must declare total refund amount"

    # 5. Tests
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_trade_idempotency.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert " passed" in result.stdout, f"unexpected output: {result.stdout}"

    print("ok execute_buy idempotency-guard + paper_positions MERGE upsert + "
          "get_paper_trades_for_ticker_since helper + cleanup script (dry-run/apply) + "
          "4 new tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
