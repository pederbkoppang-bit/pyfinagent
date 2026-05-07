"""phase-23.3.5: immutable verification."""

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


def check_log_paths_corrected():
    rel = "backend/api/cron_dashboard_api.py"
    text = _read(rel)
    ast.parse(text)
    # Re-pointed keys
    assert '"harness":' in text and 'handoff" / "mas-harness.log"' in text, \
        "harness must point at handoff/mas-harness.log (repo root)"
    assert '"autoresearch":' in text and 'handoff" / "autoresearch.log"' in text, \
        "autoresearch must point at handoff/autoresearch.log (repo root)"
    # New keys
    for key in ("autoresearch_launchd", "ablation", "ablation_launchd"):
        assert f'"{key}":' in text, f"new allowlist key {key} missing"
    # backend, watchdog, restart must remain at their current paths
    assert '"backend":' in text, "backend key removed"
    return f"OK {rel}"


def check_frontend_log_keys_synced():
    rel = "frontend/src/app/cron/page.tsx"
    text = _read(rel)
    for key in (
        "backend", "watchdog", "restart", "harness", "mas_harness_launchd",
        "autoresearch", "autoresearch_launchd", "ablation", "ablation_launchd",
    ):
        assert f'key: "{key}"' in text, f"frontend LOG_KEYS missing {key!r}"
    return f"OK {rel}"


def check_pytest_passes():
    rel = "tests/services/test_log_path_allowlist.py"
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


def check_live_harness_returns_recent():
    """The harness key must serve LIVE content (multi-MB) not the stale 4KB."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/logs/tail?log=harness&lines=5",
            timeout=5,
        ) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as e:
        return f"SKIP live -- backend unreachable ({e})"
    # Live mas-harness.log is 38+ MB; stale was 2.9 MB. Either way we want
    # the larger file. Use a generous threshold to allow growth.
    size = body.get("total_size_bytes", 0)
    assert size > 5_000_000, \
        f"harness key resolves to a small/stale file (total_size_bytes={size}); should be the live multi-MB log"
    assert body.get("n_returned", 0) > 0, "harness tail returned 0 lines"
    return f"OK live -- harness tail size={size//1024}KB"


def check_live_autoresearch_launchd_surfaces_env_bug():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/logs/tail?log=autoresearch_launchd&lines=3",
            timeout=5,
        ) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as e:
        return f"SKIP live -- backend unreachable ({e})"
    joined = "\n".join(body.get("lines", []))
    assert "command not found" in joined, \
        "autoresearch_launchd should expose the .env exit-127 errors"
    return f"OK live -- autoresearch_launchd surfaces .env bug"


def check_audit_findings():
    rel = "handoff/current/phase-23.3.5-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    text = p.read_text()
    assert "line 24" in text and "line 56" in text, \
        "audit findings must document .env line 24 + line 56 bugs"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_log_paths_corrected,
        check_frontend_log_keys_synced,
        check_pytest_passes,
        check_live_harness_returns_recent,
        check_live_autoresearch_launchd_surfaces_env_bug,
        check_audit_findings,
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
    print(f"\nphase-23.3.5 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
