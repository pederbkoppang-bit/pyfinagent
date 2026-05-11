"""phase-23.5.5 verifier — watchdog_health_check liveness.

Replays the immutable verification from `.claude/masterplan.json::23.5.5`:

  Probe `/api/jobs/all` and assert that the `watchdog_health_check`
  entry has `status != "manifest"` and `next_run is not None`.

After the phase-23.5.2.6 spam fix and the phase-23.5.2.5 bridge,
`status` should advance to `"ok"` after the first fire (15 min after
daemon restart). `next_run` advances every 15 min (interval=15).

Exit 0 on PASS; non-zero with a label on the failed assert.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "watchdog_health_check"
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
