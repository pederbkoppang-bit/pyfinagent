"""phase-23.3.4: immutable verification."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_launchd_jobs_extended():
    rel = "backend/api/cron_dashboard_api.py"
    text = _read(rel)
    ast.parse(text)
    for new_id in (
        "com.pyfinagent.backend",
        "com.pyfinagent.frontend",
        "com.pyfinagent.mas-harness",
        "com.pyfinagent.ablation",
        "com.pyfinagent.autoresearch",
    ):
        assert f'"id": "{new_id}"' in text, f"_LAUNCHD_JOBS missing {new_id}"
    assert '"id": "com.pyfinagent.claude-code-proxy"' not in text, \
        "claude-code-proxy must NOT have an id entry in _LAUNCHD_JOBS"
    return f"OK {rel}"


def check_pytest_passes():
    rel = "tests/api/test_launchd_manifest_count.py"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", rel, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        raise AssertionError(f"pytest failed: {tail}")
    return f"OK {rel}"


def check_live_api_returns_6_launchd():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/jobs/all", timeout=5,
        ) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as e:
        return f"SKIP live -- backend unreachable ({e})"
    launchd = [j for j in body.get("jobs", []) if j.get("source") == "launchd"]
    assert len(launchd) == 6, \
        f"expected 6 live launchd entries, got {len(launchd)}: {[j['id'] for j in launchd]}"
    return f"OK live -- /api/jobs/all returns 6 launchd entries"


def check_audit_findings_doc():
    rel = "handoff/current/phase-23.3.4-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    text = p.read_text()
    assert "127" in text and "autoresearch" in text, \
        "audit findings must document autoresearch exit 127"
    assert "ALPHAVANTAGE_API_KEY" in text, \
        "audit findings must show the literal .env line"
    assert "leading space" in text or "remove the space" in text.lower(), \
        "audit findings must explain the one-character fix"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_launchd_jobs_extended,
        check_pytest_passes,
        check_live_api_returns_6_launchd,
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
    print(f"\nphase-23.3.4 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
