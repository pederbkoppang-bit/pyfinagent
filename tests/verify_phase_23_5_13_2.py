"""phase-23.5.13.2 verifier — launchctl-print bridge for /api/jobs/all.

  1. `_launchctl_state` symbol present in cron_dashboard_api.
  2. The new launchd-bridge unit tests pass.
  3. The launchd merge block is wired in `get_all_jobs()` (replaces the
     prior `_static_to_dict(entry, source="launchd")` call).
  4. Live `/api/jobs/all` returns 6 launchd entries with >=4
     non-manifest status values.

Exit 0 only when all 4 checks pass.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CDA_PY = REPO / "backend" / "api" / "cron_dashboard_api.py"
TEST_FILE = REPO / "tests" / "api" / "test_cron_dashboard_launchd_bridge.py"
URL = "http://localhost:8000/api/jobs/all"


def check_helper_present() -> tuple[bool, str]:
    src = CDA_PY.read_text(encoding="utf-8")
    if "def _launchctl_state(" not in src:
        return False, "_launchctl_state helper missing"
    if "def _probe_launchctl(" not in src:
        return False, "_probe_launchctl helper missing"
    if "def _classify_launchctl_state(" not in src:
        return False, "_classify_launchctl_state helper missing"
    return True, "all 3 helpers present"


def check_merge_wired() -> tuple[bool, str]:
    src = CDA_PY.read_text(encoding="utf-8")
    # The launchd loop body must call `_launchctl_state(entry["id"])` AND must
    # not call `_static_to_dict(..., source="launchd")` anymore.
    if not re.search(r"_launchctl_state\(\s*entry\[\"id\"\]\s*\)", src):
        return False, "_launchctl_state(entry['id']) not called in get_all_jobs"
    if re.search(r'_static_to_dict\(\s*entry\s*,\s*source\s*=\s*"launchd"\s*\)', src):
        return False, "legacy _static_to_dict(..., source='launchd') still present"
    return True, "launchd merge block wired"


def check_unit_tests_pass() -> tuple[bool, str]:
    if not TEST_FILE.exists():
        return False, f"test file missing: {TEST_FILE}"
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    p = subprocess.run(
        [bin_str, "-m", "pytest", str(TEST_FILE), "-q"],
        capture_output=True, text=True, timeout=120, cwd=REPO,
    )
    if p.returncode != 0:
        return False, f"pytest failed (exit {p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-500:]}"
    return True, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "pytest OK"


def check_live_api() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        return False, f"backend unreachable: {exc}"
    ld = [j for j in payload.get("jobs", []) if j.get("source") == "launchd"]
    if len(ld) != 6:
        return False, f"want 6 launchd, got {len(ld)}"
    non_manifest = [j for j in ld if j.get("status") != "manifest"]
    if len(non_manifest) < 4:
        bad = [(j["id"], j.get("status")) for j in ld]
        return False, f"want >=4 non-manifest, got {len(non_manifest)}: {bad}"
    return True, f"6 launchd, {len(non_manifest)} non-manifest"


def main() -> int:
    checks = [
        ("helper symbols present", check_helper_present),
        ("merge block wired",      check_merge_wired),
        ("unit tests pass",        check_unit_tests_pass),
        ("live API >=4 non-manifest", check_live_api),
    ]
    print("=== phase-23.5.13.2 verifier ===")
    failed = []
    for label, fn in checks:
        ok, info = fn()
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label}: {info}")
        if not ok:
            failed.append(label)
    print()
    if failed:
        print(f"FAIL ({len(failed)}/{len(checks)}): {failed}")
        return 1
    print(f"PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
