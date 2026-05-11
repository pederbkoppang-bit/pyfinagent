"""phase-23.6.1 verifier — phase-9 production fn wiring.

  1. `backend/slack_bot/jobs/_production_fns.py` exists with 8 factories.
  2. `register_phase9_jobs()` accepts `app` + `loop` kwargs (signature check).
  3. `start_scheduler` captures `asyncio.get_running_loop()` and passes
     `app, loop` to `register_phase9_jobs`.
  4. New unit tests in `test_phase9_production_wiring.py` pass (14 tests).
  5. All existing slack_bot tests still pass (no regression).
  6. Live `/api/jobs/all` shows all 11 slack_bot jobs still non-manifest.

Exit 0 only when all 6 checks pass.
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROD_FNS_PY = REPO / "backend" / "slack_bot" / "jobs" / "_production_fns.py"
SCHEDULER_PY = REPO / "backend" / "slack_bot" / "scheduler.py"
WIRING_TESTS = REPO / "tests" / "slack_bot" / "test_phase9_production_wiring.py"
SLACK_BOT_TESTS_DIR = REPO / "tests" / "slack_bot"
URL = "http://localhost:8000/api/jobs/all"

EXPECTED_FACTORIES = (
    "make_price_fetch_fn",
    "make_price_write_fn",
    "make_fred_fetch_fn",
    "make_fred_write_fn",
    "make_ledger_fetch_fn",
    "make_outcome_write_fn",
    "make_alert_fn_for_budget",
    "make_alert_fn_for_integrity",
)


def check_factories_present() -> tuple[bool, str]:
    if not PROD_FNS_PY.exists():
        return False, f"{PROD_FNS_PY} missing"
    src = PROD_FNS_PY.read_text(encoding="utf-8")
    missing = [f for f in EXPECTED_FACTORIES if f"def {f}(" not in src]
    if missing:
        return False, f"missing factories: {missing}"
    # Confirm lazy imports: yfinance / fredapi / google.cloud.bigquery should
    # NOT appear at module-top (only inside closure bodies). Check that the
    # `import yfinance` token appears AFTER `def make_price_fetch_fn(`.
    yf_pos = src.find("import yfinance")
    factory_pos = src.find("def make_price_fetch_fn(")
    if yf_pos > 0 and yf_pos < factory_pos:
        return False, "yfinance imported eagerly at module top — must be lazy"
    return True, f"all {len(EXPECTED_FACTORIES)} factories present + lazy imports"


def check_register_signature() -> tuple[bool, str]:
    """Verify register_phase9_jobs accepts app+loop kwargs.

    Uses the venv Python (which has httpx + Slack Bolt) when available; falls
    back to a regex check on the source if the import path fails (e.g., the
    verifier was invoked with the system python3).
    """
    pytest_bin = REPO / ".venv" / "bin" / "python"
    if pytest_bin.exists():
        # Use venv python for the import-based check
        snippet = (
            "import sys, inspect; sys.path.insert(0, '.'); "
            "from backend.slack_bot.scheduler import register_phase9_jobs as f; "
            "sig = inspect.signature(f); "
            "assert 'app' in sig.parameters, 'missing app'; "
            "assert 'loop' in sig.parameters, 'missing loop'; "
            "assert sig.parameters['app'].default is None, 'app default not None'; "
            "assert sig.parameters['loop'].default is None, 'loop default not None'; "
            "print('OK')"
        )
        p = subprocess.run(
            [str(pytest_bin), "-c", snippet],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        )
        if p.returncode == 0 and "OK" in p.stdout:
            return True, "signature accepts app+loop with None defaults (venv-import check)"
        return False, f"venv import check failed: {p.stdout}{p.stderr}"
    # Fallback: regex source check
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    if not re.search(r"def register_phase9_jobs\([^)]*app\s*:[^,)=]*=\s*None", src, re.DOTALL):
        return False, "regex did not find app=None kwarg"
    if not re.search(r"def register_phase9_jobs\([^)]*loop\s*:[^,)=]*=\s*None", src, re.DOTALL):
        return False, "regex did not find loop=None kwarg"
    return True, "signature has app+loop kwargs (regex fallback)"


def check_start_scheduler_passes_loop() -> tuple[bool, str]:
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    if "asyncio.get_running_loop()" not in src:
        return False, "start_scheduler does not call asyncio.get_running_loop()"
    if not re.search(r"register_phase9_jobs\([^)]*app=app[^)]*loop=", src):
        return False, "start_scheduler does not pass app+loop to register_phase9_jobs"
    return True, "start_scheduler captures loop and passes app+loop"


def _run_pytest(target: Path) -> tuple[bool, str]:
    if not target.exists():
        return False, f"missing: {target}"
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    p = subprocess.run(
        [bin_str, "-m", "pytest", str(target), "-q"],
        capture_output=True, text=True, timeout=180, cwd=REPO,
    )
    if p.returncode != 0:
        return False, f"pytest failed (exit {p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-500:]}"
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "pytest OK"
    return True, last


def check_wiring_tests_pass() -> tuple[bool, str]:
    return _run_pytest(WIRING_TESTS)


def check_all_slack_bot_tests_pass() -> tuple[bool, str]:
    return _run_pytest(SLACK_BOT_TESTS_DIR)


def check_live_jobs_all_no_regression() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        return False, f"backend unreachable: {exc}"
    slack = [j for j in payload.get("jobs", []) if j.get("source") == "slack_bot"]
    if len(slack) != 11:
        return False, f"want 11 slack_bot, got {len(slack)}"
    bad = [j["id"] for j in slack if j.get("status") == "manifest"]
    if bad:
        return False, f"jobs reverted to manifest: {bad}"
    return True, f"11/11 slack_bot non-manifest"


def main() -> int:
    checks = [
        ("factories present + lazy imports", check_factories_present),
        ("register_phase9_jobs signature",   check_register_signature),
        ("start_scheduler passes loop",      check_start_scheduler_passes_loop),
        ("wiring unit tests pass",           check_wiring_tests_pass),
        ("all slack_bot tests pass",         check_all_slack_bot_tests_pass),
        ("live /api/jobs/all unchanged",     check_live_jobs_all_no_regression),
    ]
    print("=== phase-23.6.1 verifier ===")
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
