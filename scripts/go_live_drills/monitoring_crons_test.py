"""
Go-Live drill test: all monitoring crons operational (Phase 4.4.3.4).

Standalone, stdlib-only drill. Inspects the scheduler, settings, and
formatters source code via AST parsing to verify that all three required
cron registrations exist:

  1. morning_digest  — cron trigger
  2. evening_digest  — cron trigger
  3. watchdog_health_check — interval trigger

Run from the repo root:

    python scripts/go_live_drills/monitoring_crons_test.py

Exit code 0 on PASS, exit 1 on any failure.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "backend" / "slack_bot" / "scheduler.py"
SETTINGS_PATH = REPO_ROOT / "backend" / "config" / "settings.py"
FORMATTERS_PATH = REPO_ROOT / "backend" / "slack_bot" / "formatters.py"

PASS_COUNT = 0
FAIL_COUNT = 0


def _report(name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        print("PASS " + name + (": " + detail if detail else ""))
    else:
        FAIL_COUNT += 1
        print("FAIL " + name + (": " + detail if detail else ""))


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_add_job_ids(source: str) -> dict[str, str]:
    """Parse scheduler source and extract {job_id: trigger_type} from add_job calls."""
    tree = ast.parse(source)
    jobs = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_job"):
            continue
        job_id = None
        trigger_type = None
        for kw in node.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                job_id = kw.value.value
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            trigger_type = node.args[1].value
        if job_id and trigger_type:
            jobs[job_id] = trigger_type
    return jobs


def scenario_0_files_exist():
    for path, label in [
        (SCHEDULER_PATH, "scheduler.py"),
        (SETTINGS_PATH, "settings.py"),
        (FORMATTERS_PATH, "formatters.py"),
    ]:
        _report("S0 " + label + " exists", path.exists(), str(path))


def scenario_1_three_jobs_registered():
    source = _read_source(SCHEDULER_PATH)
    jobs = _find_add_job_ids(source)
    required = {"morning_digest", "evening_digest", "watchdog_health_check"}
    found = set(jobs.keys()) & required
    _report(
        "S1 all 3 cron jobs registered in scheduler",
        found == required,
        "found=" + repr(sorted(found)) + " required=" + repr(sorted(required)),
    )


def scenario_2_trigger_types():
    source = _read_source(SCHEDULER_PATH)
    jobs = _find_add_job_ids(source)
    _report(
        "S2 morning_digest uses cron trigger",
        jobs.get("morning_digest") == "cron",
        "trigger=" + repr(jobs.get("morning_digest")),
    )
    _report(
        "S2 evening_digest uses cron trigger",
        jobs.get("evening_digest") == "cron",
        "trigger=" + repr(jobs.get("evening_digest")),
    )
    _report(
        "S2 watchdog_health_check uses interval trigger",
        jobs.get("watchdog_health_check") == "interval",
        "trigger=" + repr(jobs.get("watchdog_health_check")),
    )


def scenario_3_settings_fields():
    source = _read_source(SETTINGS_PATH)
    fields = ["morning_digest_hour", "evening_digest_hour", "watchdog_interval_minutes"]
    for field in fields:
        _report(
            "S3 settings has " + field,
            field in source,
        )


def scenario_4_evening_formatter():
    source = _read_source(FORMATTERS_PATH)
    _report(
        "S4 format_evening_digest exists in formatters",
        "def format_evening_digest" in source,
    )


def scenario_5_scheduler_imports_evening_formatter():
    source = _read_source(SCHEDULER_PATH)
    _report(
        "S5 scheduler imports format_evening_digest",
        "format_evening_digest" in source,
    )


def scenario_6_watchdog_hits_health():
    source = _read_source(SCHEDULER_PATH)
    _report(
        "S6 watchdog checks /api/health endpoint",
        "/api/health" in source,
    )


def main():
    scenario_0_files_exist()
    scenario_1_three_jobs_registered()
    scenario_2_trigger_types()
    scenario_3_settings_fields()
    scenario_4_evening_formatter()
    scenario_5_scheduler_imports_evening_formatter()
    scenario_6_watchdog_hits_health()

    total = PASS_COUNT + FAIL_COUNT
    if FAIL_COUNT:
        print("DRILL FAIL: " + str(PASS_COUNT) + "/" + str(total) + " scenarios passed")
        return 1
    print(
        "DRILL PASS: " + str(PASS_COUNT) + "/" + str(total)
        + " monitoring cron scenarios verified against scheduler.py, "
        + "settings.py, and formatters.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
