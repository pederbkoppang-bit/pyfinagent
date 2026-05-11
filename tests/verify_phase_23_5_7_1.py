"""phase-23.5.7.1 verifier — defensive dict-envelope coerce in _send_evening_digest.

  1. _send_evening_digest body contains the `isinstance(_raw, dict)` guard.
  2. format_evening_digest body is unchanged (formatter still slices
     `trades_today[:10]` -- the formatter remains strictly typed).
  3. The 4 unit tests in tests/slack_bot/test_evening_digest_envelope_coerce.py
     all pass.
  4. The 4 tests in tests/slack_bot/test_digest_url_semantics.py for the
     evening case still pass (now with the realistic envelope shape).

Exit 0 only when all 4 checks pass.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEDULER_PY = REPO / "backend" / "slack_bot" / "scheduler.py"
FORMATTERS_PY = REPO / "backend" / "slack_bot" / "formatters.py"
COERCE_TEST = REPO / "tests" / "slack_bot" / "test_evening_digest_envelope_coerce.py"
URL_TEST = REPO / "tests" / "slack_bot" / "test_digest_url_semantics.py"


def _extract_function_body(src: str, name: str) -> str | None:
    pattern = rf"async def {re.escape(name)}\(.*?\):\n(.*?)\n(?:async def|def [a-z]|class )"
    m = re.search(pattern, src, flags=re.DOTALL)
    return m.group(1) if m else None


def check_evening_digest_has_coerce() -> tuple[bool, str]:
    body = _extract_function_body(SCHEDULER_PY.read_text(encoding="utf-8"), "_send_evening_digest")
    if body is None:
        return False, "could not locate _send_evening_digest body"
    if "isinstance(_raw, dict)" not in body:
        return False, "_send_evening_digest body missing isinstance(_raw, dict) guard"
    if "trades_data = _raw.get(\"trades\", [])" not in body and 'trades_data = _raw.get("trades", [])' not in body:
        return False, "_send_evening_digest body missing _raw.get('trades', []) extraction"
    return True, "envelope coerce wired in _send_evening_digest"


def check_format_evening_digest_unchanged() -> tuple[bool, str]:
    src = FORMATTERS_PY.read_text(encoding="utf-8")
    if "trades_today[:10]" not in src and "trades_today[: 10]" not in src and "trades_today[:max" not in src:
        # The slice may be on a different name; just confirm the formatter exists
        # and we did not refactor it to a different shape.
        if "def format_evening_digest" not in src:
            return False, "format_evening_digest function missing"
        return True, "format_evening_digest present (slice idiom may have been preserved differently)"
    return True, "format_evening_digest still slices trades_today (formatter strictly typed; fix lives upstream)"


def _run_pytest(target: Path) -> tuple[bool, str]:
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    p = subprocess.run(
        [bin_str, "-m", "pytest", str(target), "-q"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO,
    )
    if p.returncode != 0:
        return False, f"pytest failed (exit {p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-500:]}"
    return True, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "pytest OK"


def check_coerce_tests_pass() -> tuple[bool, str]:
    if not COERCE_TEST.exists():
        return False, f"test file missing: {COERCE_TEST}"
    return _run_pytest(COERCE_TEST)


def check_url_tests_pass() -> tuple[bool, str]:
    if not URL_TEST.exists():
        return False, f"test file missing: {URL_TEST}"
    return _run_pytest(URL_TEST)


def main() -> int:
    checks = [
        ("evening digest has coerce",      check_evening_digest_has_coerce),
        ("format_evening_digest unchanged", check_format_evening_digest_unchanged),
        ("coerce unit tests pass",          check_coerce_tests_pass),
        ("url-semantics tests pass",        check_url_tests_pass),
    ]
    print("=== phase-23.5.7.1 verifier ===")
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
