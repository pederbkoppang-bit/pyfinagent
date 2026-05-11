"""phase-23.5.2 verifier — ticket_queue_process_batch liveness.

Replays the immutable verification from `.claude/masterplan.json::23.5.2`:

  Probe `/api/jobs/all` and assert that the
  `ticket_queue_process_batch` entry has `status != "manifest"` and
  `next_run is not None`.

Per the research brief (researcher a258e82e537f932f1), the criterion
is structurally guaranteed for `main_apscheduler` IntervalTrigger
jobs without an `end_date`:
- `cron_dashboard_api.py:174` derives `status = "scheduled" if nrt
  is not None else "paused"` — `"manifest"` is never produced for
  this source class.
- APScheduler 3.x `IntervalTrigger.get_next_fire_time()` only
  returns `None` when `end_date` is exceeded; with no `end_date`
  configured (`backend/main.py:220-225`), `next_run is not None` is
  tautological. The real liveness gate is the `j is not None`
  existence check, also covered by this verifier.

Exit 0 on PASS; non-zero with a label on the failed assert.

Run via:
    python tests/verify_phase_23_5_2.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "ticket_queue_process_batch"
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
