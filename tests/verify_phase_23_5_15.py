"""phase-23.5.15 verifier — com.pyfinagent.backend liveness (post-23.5.13.3 amended criterion)."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "com.pyfinagent.backend"
URL = "http://localhost:8000/api/jobs/all"
VALID = {"running", "ok", "failed", "not_loaded", "unknown"}


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
        print(f"FAIL: status='manifest' (bridge regression): {job}", file=sys.stderr)
        return 4

    if job.get("status") not in VALID:
        print(f"FAIL: status {job.get('status')!r} not in {VALID}: {job}", file=sys.stderr)
        return 5

    print(f"OK {job['id']} status={job['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
