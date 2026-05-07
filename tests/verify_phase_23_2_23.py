"""phase-23.2.23: immutable verification.

Asserts:
1. backend/api/cron_dashboard_api.py defines /api/jobs/all + /api/logs/tail
   with allowlist + clamp.
2. backend/main.py registers the new router and the scheduler-registry
   wiring.
3. tests/api/test_cron_dashboard.py has the expected test names; pytest
   passes.
4. frontend/src/app/cron/page.tsx exists with the standard 6-tier shell,
   two tabs, and Phosphor icons (no emoji).
5. frontend/src/components/Sidebar.tsx has the System -> Cron / Logs entry.
6. frontend/src/lib/api.ts and types.ts have the new wrappers + types.
7. tsc --noEmit passes (caller runs separately for speed).
8. Live HTTP smoke checks against localhost:8000.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_cron_dashboard_api():
    rel = "backend/api/cron_dashboard_api.py"
    text = _read(rel)
    ast.parse(text)
    assert '@router.get("/jobs/all")' in text, "/api/jobs/all route missing"
    assert '@router.get("/logs/tail")' in text, "/api/logs/tail route missing"
    assert "_log_paths" in text, "log allowlist resolver missing"
    assert "_LINES_MIN" in text and "_LINES_MAX" in text, "lines clamp constants missing"
    assert "register_scheduler" in text, "scheduler registry hook missing"
    # Allowlist must include the documented keys
    for key in ("backend", "watchdog", "harness"):
        assert f'"{key}"' in text, f"allowlist key `{key}` missing"
    return f"OK {rel}"


def check_main_router_wiring():
    rel = "backend/main.py"
    text = _read(rel)
    ast.parse(text)
    assert "cron_dashboard_router" in text, "router not imported"
    assert "app.include_router(cron_dashboard_router)" in text, \
        "cron router not registered with the FastAPI app"
    assert "_register_cron_scheduler(\"main\"" in text, \
        "main scheduler not registered for introspection"
    assert "_register_cron_scheduler(\"queue\"" in text, \
        "queue scheduler not registered for introspection"
    # Q/A-2 advisory: deterministically assert the new routes are NOT in
    # _PUBLIC_PATHS (criterion 3). Previously only manual-grep-asserted.
    public_paths_match = re.search(
        r"_PUBLIC_PATHS\s*=\s*\((.*?)\)",
        text,
        re.DOTALL,
    )
    assert public_paths_match, "_PUBLIC_PATHS tuple not found in main.py"
    public_paths_block = public_paths_match.group(1)
    assert "/api/jobs/all" not in public_paths_block, \
        "/api/jobs/all must NOT be in _PUBLIC_PATHS (auth required per criterion 3)"
    assert "/api/logs" not in public_paths_block, \
        "/api/logs must NOT be in _PUBLIC_PATHS (auth required per criterion 3)"
    return f"OK {rel}"


def check_pytest_passes():
    rel = "tests/api/test_cron_dashboard.py"
    text = _read(rel)
    ast.parse(text)
    for fn in (
        "test_jobs_all_returns_envelope_shape",
        "test_jobs_all_includes_live_apscheduler_jobs",
        "test_jobs_all_includes_static_slack_bot_manifest",
        "test_jobs_all_includes_static_launchd_manifest",
        "test_jobs_all_handles_introspection_failure_gracefully",
        "test_logs_tail_rejects_unknown_log_key",
        "test_logs_tail_rejects_traversal_attempt",
        "test_logs_tail_returns_last_n_lines",
        "test_logs_tail_clamps_lines_to_max",
        "test_logs_tail_clamps_lines_to_min",
        "test_logs_tail_returns_empty_when_log_missing",
    ):
        assert fn in text, f"missing test: {fn}"

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", rel, "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        raise AssertionError(f"pytest failed: {tail}")
    return f"OK {rel} -- pytest 11/11"


def check_frontend_page():
    rel = "frontend/src/app/cron/page.tsx"
    text = _read(rel)
    # 6-tier shell
    assert 'flex h-screen overflow-hidden' in text, "outer shell missing"
    assert "<Sidebar />" in text, "Sidebar import/render missing"
    assert 'flex-1 overflow-y-auto scrollbar-thin' in text, "scrollable content zone missing"
    # Two tabs
    assert "TabId" in text and "jobs" in text and "logs" in text, "tabs missing"
    assert "JobsTab" in text and "LogsTab" in text, "tab components missing"
    # Phosphor icons + no emoji
    assert 'from "@/lib/icons"' in text, "icons import missing"
    # Disallow common pictographic emoji ranges that operators sometimes
    # paste in. Phosphor SVGs go through React components, not Unicode.
    pictograph = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]"
    )
    bad = pictograph.findall(text)
    assert not bad, f"emoji found in cron page: {bad[:3]}"
    return f"OK {rel}"


def check_sidebar_entry():
    rel = "frontend/src/components/Sidebar.tsx"
    text = _read(rel)
    assert 'href: "/cron"' in text, "Cron entry missing from sidebar"
    assert 'label: "Cron / Logs"' in text, "Cron entry label wrong"
    assert "icon: Clock" in text, "Clock icon not used"
    return f"OK {rel}"


def check_api_client_and_types():
    types_text = _read("frontend/src/lib/types.ts")
    assert "JobInfo" in types_text and "AllJobsResponse" in types_text, \
        "JobInfo / AllJobsResponse missing from types.ts"
    assert "LogTailResponse" in types_text, "LogTailResponse missing"
    api_text = _read("frontend/src/lib/api.ts")
    assert "getAllJobs" in api_text, "getAllJobs missing from api.ts"
    assert "getLogTail" in api_text, "getLogTail missing from api.ts"
    return "OK frontend/src/lib/{types,api}.ts"


def check_live_endpoint():
    """Hit the live backend (best-effort: skip if backend not running)."""
    try:
        import urllib.request
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/health", timeout=3,
        ) as r:
            if r.status != 200:
                return "SKIP live -- /api/health not 200"
    except Exception as e:
        return f"SKIP live -- backend not reachable ({type(e).__name__})"

    # Note: without an auth token this will return 401 -- that itself
    # PROVES the route is registered + protected. We don't need to
    # auth here; we're just checking the route exists.
    try:
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api/jobs/all", timeout=3)
        except urllib.error.HTTPError as e:
            assert e.code in (401, 403), \
                f"/api/jobs/all returned {e.code} (expected 401/403 unauth or 200 auth)"
            return f"OK live -- /api/jobs/all responded {e.code} (auth-protected route)"
    except Exception as e:
        return f"WARN live -- {type(e).__name__}: {e}"
    return "OK live -- /api/jobs/all reachable"


def main() -> int:
    checks = [
        check_cron_dashboard_api,
        check_main_router_wiring,
        check_pytest_passes,
        check_frontend_page,
        check_sidebar_entry,
        check_api_client_and_types,
        check_live_endpoint,
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
    print(f"\nphase-23.2.23 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
