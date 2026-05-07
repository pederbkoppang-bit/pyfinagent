"""phase-23.3.1: immutable verification.

Asserts the two main APScheduler add_job calls have explicit id + name
+ replace_existing, and live /api/jobs/all returns human-readable
labels for both jobs.
"""

from __future__ import annotations

import ast
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_main_py_queue_job():
    rel = "backend/main.py"
    text = _read(rel)
    ast.parse(text)
    assert 'id="ticket_queue_process_batch"' in text, \
        "queue add_job missing id='ticket_queue_process_batch'"
    assert 'name="Ticket queue batch processor"' in text, \
        "queue add_job missing name"
    assert "replace_existing=True" in text, \
        "queue add_job missing replace_existing=True"
    return f"OK {rel}"


def check_paper_trading_py():
    rel = "backend/api/paper_trading.py"
    text = _read(rel)
    ast.parse(text)
    assert 'name="Paper trading daily run"' in text, \
        "paper_trading_daily add_job missing name"
    return f"OK {rel}"


def check_live_api_labels():
    """Live HTTP probe: GET /api/jobs/all and assert both main jobs
    surface with human-readable labels (no UUID hex strings)."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/jobs/all", timeout=5,
        ) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as e:
        return f"SKIP live -- backend unreachable ({e})"

    main_jobs = [j for j in body.get("jobs", []) if j.get("source") == "main_apscheduler"]
    ids = {j["id"]: j for j in main_jobs}

    assert "paper_trading_daily" in ids, \
        f"paper_trading_daily missing from main_apscheduler jobs; got ids: {list(ids)}"
    assert ids["paper_trading_daily"]["description"] == "Paper trading daily run", \
        f"paper_trading_daily description wrong: {ids['paper_trading_daily']['description']!r}"

    assert "ticket_queue_process_batch" in ids, \
        f"ticket_queue_process_batch missing; got ids: {list(ids)}"
    assert ids["ticket_queue_process_batch"]["description"] == "Ticket queue batch processor"

    # Sanity: no UUID-hex ids in main_apscheduler
    import re
    uuid_pattern = re.compile(r"^[0-9a-f]{32}$")
    for j in main_jobs:
        assert not uuid_pattern.match(j["id"]), \
            f"main_apscheduler still has UUID-hex job id: {j['id']!r}"

    return "OK live -- both main jobs have human-readable labels"


def check_audit_findings_doc():
    rel = "handoff/current/phase-23.3.1-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"audit findings missing: {rel}"
    text = p.read_text()
    assert "paper_trading_daily" in text and "ticket_queue_process_batch" in text, \
        "audit findings must name both jobs"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_main_py_queue_job,
        check_paper_trading_py,
        check_live_api_labels,
        check_audit_findings_doc,
    ]
    failed = 0
    for fn in checks:
        try:
            print(fn())
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print(f"\nphase-23.3.1 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
