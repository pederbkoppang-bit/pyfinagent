"""phase-23.6.0 verifier — dotenv validator + operator-fix doc + pre-commit hook.

  1. `scripts/validators/check_dotenv_syntax.py` exists; runs against
     synthetic clean fixture (exit 0) AND synthetic dirty fixture (exit 1).
  2. `tests/services/test_dotenv_syntax.py` test suite passes.
  3. `handoff/runbooks/dotenv-leading-space-fix.md` exists and contains
     the verbatim sed sequence + verification command.
  4. `.git/hooks/pre-commit` (or `.pre-commit-config.yaml`) invokes the
     validator on staged .env files.

Exit 0 only when all 4 checks pass.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VALIDATOR = REPO / "scripts" / "validators" / "check_dotenv_syntax.py"
TEST_FILE = REPO / "tests" / "services" / "test_dotenv_syntax.py"
RUNBOOK = REPO / "handoff" / "runbooks" / "dotenv-leading-space-fix.md"
PRE_COMMIT = REPO / ".git" / "hooks" / "pre-commit"
PRE_COMMIT_CONFIG = REPO / ".pre-commit-config.yaml"


def check_validator_runs() -> tuple[bool, str]:
    if not VALIDATOR.exists():
        return False, f"validator missing: {VALIDATOR}"
    # Run against synthetic clean fixture
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("KEY=value\nOTHER=42\n")
        clean_path = f.name
    p_clean = subprocess.run(
        [sys.executable, str(VALIDATOR), clean_path],
        capture_output=True, text=True, timeout=20,
    )
    if p_clean.returncode != 0:
        return False, f"validator exit {p_clean.returncode} on clean fixture: {p_clean.stdout}{p_clean.stderr}"
    # Run against synthetic dirty fixture
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("BAD= value\n")
        dirty_path = f.name
    p_dirty = subprocess.run(
        [sys.executable, str(VALIDATOR), dirty_path],
        capture_output=True, text=True, timeout=20,
    )
    if p_dirty.returncode != 1:
        return False, f"validator exit {p_dirty.returncode} on dirty fixture (want 1): {p_dirty.stdout}{p_dirty.stderr}"
    if "CRITICAL" not in p_dirty.stdout:
        return False, f"validator did not report CRITICAL on dirty fixture: {p_dirty.stdout}"
    return True, "validator clean=0 dirty=1 with CRITICAL surfaced"


def check_pytest_passes() -> tuple[bool, str]:
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


def check_runbook() -> tuple[bool, str]:
    if not RUNBOOK.exists():
        return False, f"runbook missing: {RUNBOOK}"
    text = RUNBOOK.read_text(encoding="utf-8")
    required_tokens = (
        "sed -i ''",
        "[A-Z_][A-Z0-9_]*",
        "launchctl bootstrap",
        "check_dotenv_syntax.py",
    )
    missing = [t for t in required_tokens if t not in text]
    if missing:
        return False, f"runbook missing tokens: {missing}"
    return True, f"runbook contains all {len(required_tokens)} required tokens"


def check_pre_commit_hook() -> tuple[bool, str]:
    if PRE_COMMIT.exists():
        text = PRE_COMMIT.read_text(encoding="utf-8")
        if "check_dotenv_syntax.py" in text:
            return True, ".git/hooks/pre-commit invokes the validator"
    if PRE_COMMIT_CONFIG.exists():
        text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
        if "check_dotenv_syntax" in text:
            return True, ".pre-commit-config.yaml references the validator"
    return False, f"neither {PRE_COMMIT} nor {PRE_COMMIT_CONFIG} invokes the validator"


def main() -> int:
    checks = [
        ("validator runs (clean=0, dirty=1)", check_validator_runs),
        ("pytest test_dotenv_syntax passes",  check_pytest_passes),
        ("runbook contains required tokens",  check_runbook),
        ("pre-commit hook invokes validator", check_pre_commit_hook),
    ]
    print("=== phase-23.6.0 verifier ===")
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
