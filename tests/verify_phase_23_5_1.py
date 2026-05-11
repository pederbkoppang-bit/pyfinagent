"""phase-23.5.1 verifier — paper_trading_daily liveness.

Replays the immutable verification from `.claude/masterplan.json::23.5.1`:

  Probe `/api/jobs/all` and assert that the `paper_trading_daily`
  entry has `status != "manifest"` and `next_run is not None`.

Per the research brief (researcher a60d76678e12b724f), this is a
structurally-guaranteed liveness signal for `main_apscheduler`
jobs: the dashboard derives `status = "scheduled" if nrt is not
None else "paused"` (`backend/api/cron_dashboard_api.py:174`) — the
literal value `"manifest"` is never produced for this source.

Exit 0 on PASS; non-zero with a label on the failed assert.

Run via:
    python tests/verify_phase_23_5_1.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "paper_trading_daily"
URL = "http://localhost:8000/api/jobs/all"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL: backend unreachable at {URL}: {exc}", file=sys.stderr)
        return 2

    job = next((j for j in payload.get("jobs", []) if j.get("id") == JOB_ID), None)
    if job is None:
        print(f"FAIL: job {JOB_ID!r} missing from /api/jobs/all", file=sys.stderr)
        return 3

    if job.get("status") == "manifest":
        print(f"FAIL: status still 'manifest': {job}", file=sys.stderr)
        return 4

    if job.get("next_run") is None:
        print(f"FAIL: next_run is null: {job}", file=sys.stderr)
        return 5

    print(f"OK {job['id']} status={job['status']} next_run={job['next_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
