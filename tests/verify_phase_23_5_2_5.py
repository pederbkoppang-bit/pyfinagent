"""phase-23.5.2.5 verifier — slack-bot heartbeat-bridge for /api/jobs/all.

Replays the immutable verification from `.claude/masterplan.json::23.5.2.5`:

  - All 11 slack_bot jobs surface in /api/jobs/all.
  - At least 6 of them have status != "manifest" (registry-merged rows).
  - All 11 have next_run populated (seeded by startup state-push).

Exit 0 on PASS; non-zero with a label on the failed assert.

Run via:
    python tests/verify_phase_23_5_2_5.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8000/api/jobs/all"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL: backend unreachable at {URL}: {exc}", file=sys.stderr)
        return 2

    slack = [j for j in payload.get("jobs", []) if j.get("source") == "slack_bot"]
    if len(slack) != 11:
        print(f"FAIL: want 11 slack_bot, got {len(slack)}", file=sys.stderr)
        return 3

    non_manifest = [j for j in slack if j.get("status") != "manifest"]
    if len(non_manifest) < 6:
        bad = [(j["id"], j.get("status")) for j in slack]
        print(f"FAIL: expect >=6 non-manifest, got {len(non_manifest)}: {bad}", file=sys.stderr)
        return 4

    with_nr = [j for j in slack if j.get("next_run")]
    if len(with_nr) != 11:
        miss = [j["id"] for j in slack if not j.get("next_run")]
        print(f"FAIL: all 11 must have next_run; missing: {miss}", file=sys.stderr)
        return 5

    print(f"OK 11 slack_bot; {len(non_manifest)} non-manifest; {len(with_nr)} with next_run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
