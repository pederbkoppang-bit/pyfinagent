"""phase-23.2.20: immutable verification.

Asserts:
1. backend/services/cycle_health.py: SAFE.TIMESTAMP(MAX(...)) wrapper present;
   except clause logs at logger.warning (not logger.debug).
2. tests/services/test_freshness_query_shape.py: 5 expected test names present.
3. AST passes for the modified .py file.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_cycle_health():
    rel = "backend/services/cycle_health.py"
    text = _read(rel)
    ast.parse(text)
    assert "SAFE.TIMESTAMP(MAX(" in text, \
        "SAFE.TIMESTAMP(MAX(...)) wrapper missing — STRING columns will reject"
    # Regex: the except clause for _bq_max_event_age must use logger.warning,
    # not logger.debug. We assert at least one logger.warning appears in the
    # function body (the except is the only logger call there).
    fn_match = re.search(
        r"def _bq_max_event_age\([^)]*\)[^:]*:.*?(?=\n(?:def |class |\Z))",
        text,
        re.DOTALL,
    )
    assert fn_match is not None, "_bq_max_event_age function not found"
    fn_body = fn_match.group(0)
    assert "logger.warning(" in fn_body, \
        "_bq_max_event_age except clause must log at WARNING level"
    assert "logger.debug(" not in fn_body, \
        "_bq_max_event_age must NOT use logger.debug (regression guard)"
    return f"OK {rel}"


def check_test_exists():
    rel = "tests/services/test_freshness_query_shape.py"
    text = _read(rel)
    ast.parse(text)
    for fn in (
        "test_sql_uses_safe_timestamp_wrapper",
        "test_returns_age_on_successful_query",
        "test_returns_none_on_empty_result",
        "test_returns_none_when_age_is_null",
        "test_failed_query_logs_at_warning_not_debug",
    ):
        assert fn in text, f"missing test: {fn}"
    return f"OK {rel}"


def main() -> int:
    checks = [check_cycle_health, check_test_exists]
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
    print("\nphase-23.2.20 verification: ALL PASS (2/2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
