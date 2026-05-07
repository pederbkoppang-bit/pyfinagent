"""phase-23.3.6: end-to-end /cron page verification.

Final consolidation check after 6 prior phase-23.3.x audit/fix steps.
Asserts the live /api/jobs/all + /api/logs/tail endpoints match the
expected post-audit shape: 19 jobs across 3 sources, 9 allowlisted
log keys, path-traversal still blocked, frontend type-checks clean.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _curl_json(path: str):
    """Fetch JSON from the live backend; raise SkipTest equivalent if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:8000{path}", timeout=5,
        ) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise AssertionError(f"backend unreachable: {e}") from e


def check_jobs_all_shape():
    body = _curl_json("/api/jobs/all")
    assert body.get("n_total") == 19, \
        f"expected 19 jobs, got {body.get('n_total')}"
    by_source: dict[str, int] = {}
    for j in body["jobs"]:
        by_source[j["source"]] = by_source.get(j["source"], 0) + 1
    assert by_source.get("main_apscheduler") == 2, \
        f"expected 2 main_apscheduler, got {by_source.get('main_apscheduler')}"
    assert by_source.get("slack_bot") == 11, \
        f"expected 11 slack_bot, got {by_source.get('slack_bot')}"
    assert by_source.get("launchd") == 6, \
        f"expected 6 launchd, got {by_source.get('launchd')}"
    return f"OK /api/jobs/all -- 19 jobs (2+11+6)"


def check_main_apscheduler_named():
    """Phase-23.3.1: queue process_batch must NOT have a UUID hex id."""
    body = _curl_json("/api/jobs/all")
    main_ids = {j["id"] for j in body["jobs"] if j["source"] == "main_apscheduler"}
    assert "paper_trading_daily" in main_ids
    assert "ticket_queue_process_batch" in main_ids, \
        f"queue job must have human-readable id; got {main_ids}"
    return "OK main_apscheduler ids are human-readable"


def check_launchd_includes_5_new_services():
    """Phase-23.3.4: 5 user-local services added beyond backend-watchdog."""
    body = _curl_json("/api/jobs/all")
    launchd_ids = {j["id"] for j in body["jobs"] if j["source"] == "launchd"}
    expected = {
        "com.pyfinagent.backend-watchdog",
        "com.pyfinagent.backend",
        "com.pyfinagent.frontend",
        "com.pyfinagent.mas-harness",
        "com.pyfinagent.ablation",
        "com.pyfinagent.autoresearch",
    }
    missing = expected - launchd_ids
    assert not missing, f"missing launchd ids: {missing}"
    return "OK launchd has all 6 expected services"


def check_logs_tail_9_keys_all_resolvable():
    """Phase-23.3.5: 9 allowlisted log keys all return without error."""
    keys = (
        "backend", "watchdog", "restart", "harness", "mas_harness_launchd",
        "autoresearch", "autoresearch_launchd", "ablation", "ablation_launchd",
    )
    failures = []
    for key in keys:
        try:
            body = _curl_json(f"/api/logs/tail?log={key}&lines=2")
            assert "lines" in body
            assert "exists" in body
        except Exception as e:
            failures.append(f"{key}: {e}")
    assert not failures, f"some keys failed: {failures}"
    return f"OK all 9 allowlisted log keys responded"


def check_path_traversal_still_blocked():
    """Phase-23.2.23 contract: unknown keys must 400, never echo paths."""
    try:
        urllib.request.urlopen(
            "http://127.0.0.1:8000/api/logs/tail?log=etc/passwd&lines=1",
            timeout=5,
        )
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"path traversal probe should return 400, got {e.code}"
        return "OK /api/logs/tail?log=etc/passwd returns 400"
    raise AssertionError("path traversal probe unexpectedly succeeded")


def check_frontend_static_clean():
    """tsc --noEmit + eslint . both exit 0 (errors-only)."""
    fe = ROOT / "frontend"
    proc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=fe, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        out = (proc.stdout + proc.stderr).strip().splitlines()[-10:]
        raise AssertionError("tsc failed:\n" + "\n".join(out))

    proc = subprocess.run(
        ["npx", "--no-install", "eslint", "."],
        cwd=fe, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        out = (proc.stdout + proc.stderr).strip().splitlines()[-10:]
        raise AssertionError("eslint failed:\n" + "\n".join(out))
    return "OK frontend tsc + eslint clean"


def check_audit_findings_consolidation():
    rel = "handoff/current/phase-23.3.6-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    text = p.read_text()
    # Must reference each prior sub-phase
    for sub in ("23.3.0", "23.3.1", "23.3.2", "23.3.3", "23.3.4", "23.3.5"):
        assert sub in text, f"consolidation missing reference to {sub}"
    # Must surface the OPERATOR-ACTION items
    assert "OPERATOR" in text and ".env" in text, \
        "consolidation must surface the .env operator action"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_jobs_all_shape,
        check_main_apscheduler_named,
        check_launchd_includes_5_new_services,
        check_logs_tail_9_keys_all_resolvable,
        check_path_traversal_still_blocked,
        check_frontend_static_clean,
        check_audit_findings_consolidation,
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
    print(f"\nphase-23.3.6 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
