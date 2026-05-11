"""phase-23.5.12 verifier — weekly_data_integrity liveness."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "weekly_data_integrity"
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
