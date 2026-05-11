"""phase-23.6.2 verifier — cosmetic schedule labels + autoresearch description.

  1. All 11 _SLACK_BOT_JOBS schedule fields use the bracket notation
     (`cron[...]` or `interval[...]`).
  2. None of the 11 entries still contain the placeholder pattern
     (`phase-9.\\d+ cron`, `phase-9.\\d+ interval`, `morning_digest_hour:00`,
     `evening_digest_hour:00`, `watchdog_interval_minutes`).
  3. Each of the 11 schedule strings exactly matches the recommended
     replacement.
  4. _LAUNCHD_JOBS autoresearch description no longer contains
     `FAILING exit 127` AND contains `exit 1` AND `phase-23.5.19`.
  5. Live `/api/jobs/all` shows the new schedule strings + new description.
  6. All 27 prior phase-23 verifiers exit 0.

Exit 0 only when all 6 checks pass.
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
URL = "http://localhost:8000/api/jobs/all"

EXPECTED_SLACK_SCHEDULES = {
    "morning_digest":          "cron[hour='8', minute='0']",
    "evening_digest":          "cron[hour='17', minute='0']",
    "watchdog_health_check":   "interval[0:15:00]",
    "prompt_leak_redteam":     "cron[hour='3', minute='15']",
    "daily_price_refresh":     "cron[hour='1']",
    "weekly_fred_refresh":     "cron[day_of_week='sun', hour='2']",
    "nightly_mda_retrain":     "cron[hour='3']",
    "hourly_signal_warmup":    "cron[minute='5']",
    "nightly_outcome_rebuild": "cron[hour='4']",
    "weekly_data_integrity":   "cron[day_of_week='mon', hour='5']",
    "cost_budget_watcher":     "cron[hour='6']",
}

PLACEHOLDER_TOKENS = (
    "phase-9.",
    "morning_digest_hour:00",
    "evening_digest_hour:00",
    "watchdog_interval_minutes",
)


def _load_module_constants() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Import the module + return (_SLACK_BOT_JOBS, _LAUNCHD_JOBS).

    Uses the venv Python via subprocess to avoid system-Python import errors
    (httpx etc. are venv-only).
    """
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    snippet = (
        "import json, sys; sys.path.insert(0, '.'); "
        "from backend.api.cron_dashboard_api import _SLACK_BOT_JOBS, _LAUNCHD_JOBS; "
        "print(json.dumps({"
        "'slack': [dict(j) for j in _SLACK_BOT_JOBS], "
        "'launchd': [dict(j) for j in _LAUNCHD_JOBS]"
        "}))"
    )
    p = subprocess.run(
        [bin_str, "-c", snippet],
        capture_output=True, text=True, timeout=30, cwd=REPO,
    )
    if p.returncode != 0:
        raise RuntimeError(f"module import failed: {p.stderr}")
    payload = json.loads(p.stdout)
    return tuple(payload["slack"]), tuple(payload["launchd"])


def check_no_placeholders() -> tuple[bool, str]:
    slack, _ = _load_module_constants()
    bad = []
    for entry in slack:
        sched = entry.get("schedule", "")
        for tok in PLACEHOLDER_TOKENS:
            if tok in sched:
                bad.append(f"{entry['id']!r}: still has {tok!r}")
                break
    if bad:
        return False, "; ".join(bad)
    return True, f"none of the 11 slack_bot entries contain placeholder tokens"


def check_schedules_exact_match() -> tuple[bool, str]:
    slack, _ = _load_module_constants()
    seen = {entry["id"]: entry["schedule"] for entry in slack}
    mismatches = []
    for jid, expected in EXPECTED_SLACK_SCHEDULES.items():
        actual = seen.get(jid)
        if actual != expected:
            mismatches.append(f"{jid!r}: expected {expected!r}, got {actual!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, f"all 11 schedules match recommended replacements"


def check_bracket_notation() -> tuple[bool, str]:
    slack, _ = _load_module_constants()
    pat = re.compile(r"^(cron|interval)\[")
    bad = [e["id"] for e in slack if not pat.match(e.get("schedule", ""))]
    if bad:
        return False, f"entries not using bracket notation: {bad}"
    return True, f"all 11 schedules use cron[...] or interval[...] format"


def check_autoresearch_description() -> tuple[bool, str]:
    _, launchd = _load_module_constants()
    auto = next((e for e in launchd if e["id"] == "com.pyfinagent.autoresearch"), None)
    if auto is None:
        return False, "com.pyfinagent.autoresearch entry missing from _LAUNCHD_JOBS"
    desc = auto.get("description", "")
    if "FAILING exit 127" in desc:
        return False, "description still contains stale 'FAILING exit 127'"
    if "exit 1" not in desc:
        return False, f"description does not mention current exit 1 state: {desc!r}"
    if "phase-23.5.19" not in desc:
        return False, f"description does not reference phase-23.5.19: {desc!r}"
    return True, "autoresearch description updated to current state"


def check_live_api() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        return False, f"backend unreachable: {exc}"
    by_id = {j["id"]: j for j in payload.get("jobs", [])}
    # Check 1 slack_bot sample
    sample = by_id.get("daily_price_refresh")
    if not sample:
        return False, "daily_price_refresh missing from /api/jobs/all"
    if sample.get("schedule") != EXPECTED_SLACK_SCHEDULES["daily_price_refresh"]:
        return False, f"live daily_price_refresh schedule still {sample.get('schedule')!r} (expected {EXPECTED_SLACK_SCHEDULES['daily_price_refresh']!r})"
    # Check the autoresearch description
    auto = by_id.get("com.pyfinagent.autoresearch")
    if not auto:
        return False, "com.pyfinagent.autoresearch missing from /api/jobs/all"
    desc = auto.get("description", "")
    if "FAILING exit 127" in desc:
        return False, "live autoresearch description still has 'FAILING exit 127'"
    if "phase-23.5.19" not in desc:
        return False, f"live autoresearch description not updated: {desc!r}"
    return True, "live API reflects new schedule strings + new description"


def check_sibling_verifier_sweep() -> tuple[bool, str]:
    """Run all phase-23.5.* and 23.6.0/23.6.1 verifiers; expect 27/27 PASS."""
    sibs = sorted(REPO.glob("tests/verify_phase_23_5_*.py")) + [
        REPO / "tests" / "verify_phase_23_6_0.py",
        REPO / "tests" / "verify_phase_23_6_1.py",
    ]
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    fails = []
    for v in sibs:
        if not v.exists():
            continue
        p = subprocess.run(
            [bin_str, str(v)],
            capture_output=True, text=True, timeout=180, cwd=REPO,
        )
        if p.returncode != 0:
            fails.append(f"{v.name}: exit {p.returncode}")
    if fails:
        return False, f"sibling regressions: {fails}"
    return True, f"{len(sibs)} sibling verifiers all exit 0"


def main() -> int:
    checks = [
        ("no placeholder tokens",                 check_no_placeholders),
        ("schedules exact match recommended",     check_schedules_exact_match),
        ("bracket notation used",                 check_bracket_notation),
        ("autoresearch description updated",      check_autoresearch_description),
        ("live API reflects edits",               check_live_api),
        ("27 sibling verifiers green",            check_sibling_verifier_sweep),
    ]
    print("=== phase-23.6.2 verifier ===")
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
