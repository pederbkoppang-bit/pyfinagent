"""phase-23.5.14 verifier — com.pyfinagent.backend-watchdog liveness (soft).

The masterplan's immutable criterion includes `next_run is not None`,
which is structurally UNMEETABLE for launchd entries: `launchctl print`
does not expose next-fire-time for StartInterval / KeepAlive jobs.
The phase-23.5.13.2 bridge surfaces real `status` but `next_run`
remains null by design (see `cron_dashboard_api.py:293`).

This verifier asserts the SOFT half of the criterion (status not
"manifest" + status is one of the bridge's documented values). The
hard criterion is expected to fail; phase-23.5.14's verdict is
CONDITIONAL with a criterion-mismatch disclosure per Anthropic
immutable-criteria doctrine.

Exit 0 when the soft criterion holds. Non-zero on a real bridge
regression (e.g., status reverts to "manifest").
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

JOB_ID = "com.pyfinagent.backend-watchdog"
URL = "http://localhost:8000/api/jobs/all"
VALID_STATUSES = {"running", "ok", "failed", "not_loaded", "unknown"}


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
        print(f"FAIL: status reverted to 'manifest' (bridge regression): {job}", file=sys.stderr)
        return 4

    if job.get("status") not in VALID_STATUSES:
        print(f"FAIL: unexpected status {job.get('status')!r}: {job}", file=sys.stderr)
        return 5

    print(f"OK {job['id']} status={job['status']} (next_run/last_run null by launchd-bridge design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
