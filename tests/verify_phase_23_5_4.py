"""phase-23.5.4 verifier — evening_digest liveness.

Replays the immutable verification from `.claude/masterplan.json::23.5.4`:

  Probe `/api/jobs/all` and assert that the `evening_digest` entry
  has `status != "manifest"` and `next_run is not None`.

Unlike phase-23.5.3, this verifier is NOT a structural false positive:
the phase-23.5.3.1 Docker-alias fix repointed _send_evening_digest's
httpx calls to localhost, so the `EVENT_JOB_EXECUTED` heartbeat now
reflects a real successful Slack post.

Exit 0 on PASS; non-zero with a label on the failed assert.

Run via:
    python tests/verify_phase_23_5_4.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "evening_digest"
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
