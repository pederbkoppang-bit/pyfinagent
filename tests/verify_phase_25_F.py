"""verify_phase_25_F -- Byte-identical regression tests for aliasing detection.

Runs pytest -k filter on the two new regression tests added to
`tests/services/test_signal_attribution.py` and asserts both PASS.

Verifies:
  1. `test_lite_path_byte_identical_flagged` test name exists in the module.
  2. `test_full_path_distinct_rationale` test name exists in the module.
  3. pytest invocation exits 0 with both tests in the PASSED output.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_FILE = REPO / "tests/services/test_signal_attribution.py"
VENV_PY = REPO / ".venv/bin/python"

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1+2: test names present in the module ────────────────────────
src = TEST_FILE.read_text(encoding="utf-8")
has_test_1 = bool(re.search(r"def test_lite_path_byte_identical_flagged\b", src))
has_test_2 = bool(re.search(r"def test_full_path_distinct_rationale\b", src))
claim(
    "1. test_lite_path_byte_identical_flagged_defined",
    has_test_1,
    "test name found" if has_test_1 else "test name missing",
)
claim(
    "2. test_full_path_distinct_rationale_defined",
    has_test_2,
    "test name found" if has_test_2 else "test name missing",
)


# ── Claim 3+4: pytest exit code 0 with both tests PASSED ───────────────
if has_test_1 and has_test_2:
    proc = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "pytest",
            "tests/services/test_signal_attribution.py",
            "-k",
            "test_lite_path_byte_identical_flagged or test_full_path_distinct_rationale",
            "-v",
            "--no-header",
            "--tb=short",
        ],
        cwd=str(REPO),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr
    exit_ok = proc.returncode == 0
    test1_passed = "test_lite_path_byte_identical_flagged PASSED" in out
    test2_passed = "test_full_path_distinct_rationale PASSED" in out
    claim(
        "3. pytest_test_lite_path_byte_identical_flagged_passes",
        exit_ok and test1_passed,
        f"exit={proc.returncode} matched={test1_passed}",
    )
    claim(
        "4. pytest_test_full_path_distinct_rationale_passes",
        exit_ok and test2_passed,
        f"exit={proc.returncode} matched={test2_passed}",
    )
else:
    claim("3. pytest_test_lite_path_byte_identical_flagged_passes", False, "test not defined; skipped invocation")
    claim("4. pytest_test_full_path_distinct_rationale_passes", False, "test not defined; skipped invocation")


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.F verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
