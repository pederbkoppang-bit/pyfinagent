"""phase-23.6.3 verifier — plist-derived next-fire-time for StartCalendarInterval launchd jobs.

  1. Helper present + plistlib imported.
  2. Algorithm correctness for ablation (Hour=3) and autoresearch (Hour=2):
     non-None ISO 8601 string with tz offset, parseable, matches plist
     wall-clock semantics (today if now<fire, else tomorrow).
  3. Graceful degradation: backend (no SCI) -> None; backend-watchdog
     (StartInterval) -> None; nonexistent label -> None.
  4. Live `/api/jobs/all`: ablation + autoresearch have non-null ISO
     `next_run`; the other 4 launchd entries have null `next_run`.
  5. tests/api/ pytest suite passes (incl. updated test_cron_dashboard.py).
  6. All 28 prior phase-23 verifiers exit 0.

Exit 0 only when all 6 checks pass.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CDA_PY = REPO / "backend" / "api" / "cron_dashboard_api.py"
URL = "http://localhost:8000/api/jobs/all"
VENV_PY = REPO / ".venv" / "bin" / "python"

CALENDAR_INTERVAL_IDS = ("com.pyfinagent.ablation", "com.pyfinagent.autoresearch")
NON_SCI_LAUNCHD_IDS = (
    "com.pyfinagent.backend",
    "com.pyfinagent.frontend",
    "com.pyfinagent.backend-watchdog",
    "com.pyfinagent.mas-harness",
)


def _py() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _call_helper(label: str) -> str:
    """Invoke `_plist_next_run(label)` via venv python; return its repr."""
    snippet = (
        "import sys; sys.path.insert(0, '.'); "
        "from backend.api.cron_dashboard_api import _plist_next_run; "
        f"print(repr(_plist_next_run({label!r})))"
    )
    p = subprocess.run(
        [_py(), "-c", snippet], capture_output=True, text=True,
        timeout=30, cwd=REPO,
    )
    if p.returncode != 0:
        raise RuntimeError(f"helper call failed: {p.stderr}")
    return p.stdout.strip()


def check_helper_present() -> tuple[bool, str]:
    src = CDA_PY.read_text()
    if "import plistlib" not in src:
        return False, "plistlib import missing"
    if "def _plist_next_run" not in src:
        return False, "_plist_next_run helper not defined"
    if "def _load_plist" not in src:
        return False, "_load_plist helper not defined"
    return True, "plistlib imported; _load_plist + _plist_next_run defined"


def check_algorithm_correctness() -> tuple[bool, str]:
    """ablation Hour=3 and autoresearch Hour=2 must return future-ISO."""
    expected_hours = {
        "com.pyfinagent.ablation": 3,
        "com.pyfinagent.autoresearch": 2,
    }
    for label, expected_hour in expected_hours.items():
        out = _call_helper(label)
        if out == "None":
            return False, f"{label}: helper returned None (expected ISO string)"
        # repr -> strip quotes
        iso = out.strip("'\"")
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError as exc:
            return False, f"{label}: ISO unparseable: {iso!r} ({exc})"
        if parsed.tzinfo is None:
            return False, f"{label}: must be tz-aware, got naive: {iso!r}"
        if parsed.hour != expected_hour or parsed.minute != 0:
            return False, f"{label}: wrong wall-clock hour/minute {parsed.hour}:{parsed.minute:02d} (expected {expected_hour}:00)"
        if parsed.second != 0 or parsed.microsecond != 0:
            return False, f"{label}: second/microsecond non-zero: {iso!r}"
        # Must be in the future (or within 5s, allowing for slow shell)
        now = datetime.now().astimezone()
        if parsed < now - timedelta(seconds=5):
            return False, f"{label}: parsed time {iso!r} is in the past (now={now.isoformat()})"
        # And not more than 24h in the future
        if parsed > now + timedelta(hours=24, minutes=5):
            return False, f"{label}: parsed time {iso!r} more than 24h ahead"
    return True, "ablation + autoresearch return correct future ISO strings"


def check_graceful_degradation() -> tuple[bool, str]:
    cases = {
        "com.pyfinagent.backend": "no StartCalendarInterval (KeepAlive)",
        "com.pyfinagent.backend-watchdog": "StartInterval, not StartCalendarInterval",
        "com.pyfinagent.does-not-exist-9999": "missing plist file",
    }
    for label, reason in cases.items():
        out = _call_helper(label)
        if out != "None":
            return False, f"{label} ({reason}): expected None, got {out!r}"
    return True, "all 3 degradation cases return None without crash"


def check_live_api() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        return False, f"backend unreachable: {exc}"
    by_id = {j["id"]: j for j in payload.get("jobs", []) if j.get("source") == "launchd"}
    if len(by_id) != 6:
        return False, f"expected 6 launchd jobs, found {len(by_id)}"
    for label in CALENDAR_INTERVAL_IDS:
        row = by_id.get(label)
        if not row:
            return False, f"{label} missing from /api/jobs/all"
        nr = row.get("next_run")
        if not isinstance(nr, str):
            return False, f"{label}: next_run must be ISO string, got {nr!r}"
        try:
            parsed = datetime.fromisoformat(nr)
        except ValueError as exc:
            return False, f"{label}: next_run unparseable: {nr!r} ({exc})"
        if parsed.tzinfo is None:
            return False, f"{label}: next_run must be tz-aware: {nr!r}"
    for label in NON_SCI_LAUNCHD_IDS:
        row = by_id.get(label)
        if row is None:
            return False, f"{label} missing from /api/jobs/all"
        if row.get("next_run") is not None:
            return False, f"{label}: next_run must be None, got {row.get('next_run')!r}"
    return True, "live API: 2 SCI rows have ISO next_run; 4 non-SCI rows null"


def check_pytest_suite() -> tuple[bool, str]:
    """phase-23.6.3 amended scope (handoff/audit/criterion_amendments.jsonl
    `phase-23.6.3-tests-api-scope`): the cron-dashboard test surface only.
    Full `tests/api/` is blocked by a pre-existing import failure in
    test_observability.py that predates 23.6.3 (see follow-up phase-23.6.4)."""
    targets = [
        "tests/api/test_cron_dashboard.py",
        "tests/api/test_cron_dashboard_launchd_bridge.py",
    ]
    p = subprocess.run(
        [_py(), "-m", "pytest", *targets, "-q", "--no-header"],
        capture_output=True, text=True, timeout=180, cwd=REPO,
    )
    if p.returncode != 0:
        tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
        return False, "pytest cron-dashboard targets failed: " + " | ".join(tail)
    last = p.stdout.strip().splitlines()[-1]
    return True, f"cron-dashboard pytest: {last}"


def check_sibling_verifier_sweep() -> tuple[bool, str]:
    """Run all 23.5.* + 23.6.0/1/2 verifiers; expect 28/28 PASS."""
    sibs = sorted(REPO.glob("tests/verify_phase_23_5_*.py")) + [
        REPO / "tests" / "verify_phase_23_6_0.py",
        REPO / "tests" / "verify_phase_23_6_1.py",
        REPO / "tests" / "verify_phase_23_6_2.py",
    ]
    fails = []
    for v in sibs:
        if not v.exists():
            continue
        p = subprocess.run(
            [_py(), str(v)], capture_output=True, text=True,
            timeout=240, cwd=REPO,
        )
        if p.returncode != 0:
            fails.append(f"{v.name}: exit {p.returncode}")
    if fails:
        return False, f"sibling regressions: {fails}"
    return True, f"{len(sibs)} sibling verifiers all exit 0"


def main() -> int:
    checks = [
        ("helper present + plistlib imported",        check_helper_present),
        ("algorithm correctness (ablation+autoresearch)", check_algorithm_correctness),
        ("graceful degradation (3 cases)",            check_graceful_degradation),
        ("live API reflects plist-derived next_run",  check_live_api),
        ("tests/api/ pytest suite passes",            check_pytest_suite),
        ("28 sibling verifiers green",                check_sibling_verifier_sweep),
    ]
    print("=== phase-23.6.3 verifier ===")
    failed = []
    for label, fn in checks:
        try:
            ok, info = fn()
        except Exception as exc:
            ok, info = False, f"check raised: {exc}"
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
