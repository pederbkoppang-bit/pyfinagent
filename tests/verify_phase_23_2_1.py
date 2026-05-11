"""phase-23.2.1 verifier.

Replays the immutable verification from `.claude/masterplan.json::23.2.1`:

    bq SELECT DATE(snapshot_date), COUNT(*) FROM paper_portfolio_snapshots
    WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(),
    INTERVAL 9 DAY) GROUP BY 1 ORDER BY 1; expect ~9 rows, no gaps

Exit semantics: exit 0 if the BQ query runs successfully. Exit non-zero
ONLY if the query itself fails (auth, network, permissions). The row
count and gap analysis are reported in stdout but do not gate the exit
code -- the row count is the finding, not a verifier failure mode.

Run via:
    source .venv/bin/activate && python tests/verify_phase_23_2_1.py
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

PROJECT = os.environ.get("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"
TABLE = "paper_portfolio_snapshots"
LOCATION = "us-central1"  # MCP defaults to US; this table is in us-central1.

SQL = f"""
SELECT DATE(PARSE_DATE('%Y-%m-%d', snapshot_date)) AS day,
       COUNT(*) AS n
FROM `{PROJECT}.{DATASET}.{TABLE}`
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(),
                                                        INTERVAL 9 DAY)
GROUP BY 1
ORDER BY 1
"""


def main() -> int:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        print(f"FAIL: google-cloud-bigquery not installed: {exc}", file=sys.stderr)
        return 2

    try:
        client = bigquery.Client(project=PROJECT, location=LOCATION)
        rows = list(client.query(SQL).result(timeout=30))
    except Exception as exc:
        print(f"FAIL: BQ query errored: {exc}", file=sys.stderr)
        return 3

    print("=== phase-23.2.1 — paper_portfolio_snapshots 9-day window ===")
    print(f"{'day':<12} | {'n':>3}")
    print("-" * 20)
    for r in rows:
        print(f"{str(r['day']):<12} | {r['n']:>3}")
    print(f"-- TOTAL ROWS: {len(rows)} --")

    today = _dt.date.today()
    expected = [today - _dt.timedelta(days=d) for d in range(0, 10)]
    present = {r["day"] for r in rows}
    missing = sorted(set(expected) - present)
    print()
    print(f"Window: {today - _dt.timedelta(days=9)} -> {today} (10 calendar days)")
    print(f"Distinct days present: {len(present)}")
    print(f"Missing dates: {missing if missing else 'none'}")
    print()

    if len(rows) >= 9 and not missing:
        print("RESULT: criterion met (~9 rows, no gaps).")
    else:
        print("RESULT: criterion NOT met (HYPOTHESIS_DISCONFIRMED).")
        print("  This is a system finding, not a verifier defect.")
        print("  See handoff/current/experiment_results.md for analysis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
