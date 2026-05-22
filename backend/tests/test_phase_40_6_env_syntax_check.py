"""phase-40.6 .env syntax-guard script tests.

Tests scripts/qa/env_syntax_check.py (stdlib-only AST-style validator).
Each test injects a synthetic .env into tmp_path + runs the script via
subprocess + asserts the violation report matches expectations.

Tests cover:
  1. script exists + executable
  2. clean file -> exit 0
  3. KeyWithoutValue rule
  4. IncorrectDelimiter rule (colon instead of equals)
  5. LeadingCharacter rule (starts with digit)
  6. QuoteCharacter rule (unbalanced quote)
  7. DuplicatedKey rule
  8. LowercaseKey rule (warning only -- doesn't fail)
  9. backend/.env.example today exits clean (canonical template invariant)
  10. pre-commit hook exists + executable
  11. CI workflow exists with continue-on-error
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qa" / "env_syntax_check.py"
HOOK = REPO_ROOT / ".claude" / "hooks" / "pre-commit-env-check.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "env-syntax-lint.yml"
CANONICAL_TEMPLATE = REPO_ROOT / "backend" / ".env.example"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_phase_40_6_script_exists_and_executable():
    assert SCRIPT.exists(), f"script missing: {SCRIPT}"
    # On unix, the file must have the executable bit set
    assert SCRIPT.stat().st_mode & 0o111, (
        f"{SCRIPT} must be executable (chmod +x)"
    )


def test_phase_40_6_clean_env_exits_zero(tmp_path: Path):
    """Canonical clean .env -> exit 0 + zero violations."""
    p = tmp_path / "clean.env"
    p.write_text(
        "# A comment\n"
        "ALPHAVANTAGE_API_KEY=abc123\n"
        "DEEP_THINK_MODEL=gemini-2.5-pro\n"
        "MAX_POSITIONS=14\n",
        encoding="utf-8",
    )
    result = _run([str(p)])
    assert result.returncode == 0, (
        f"clean .env should exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_phase_40_6_key_without_value_detected(tmp_path: Path):
    """A line with no '=' should trip KeyWithoutValue."""
    p = tmp_path / "bad.env"
    p.write_text("ORPHAN_KEY no equals here\n", encoding="utf-8")
    result = _run([str(p)])
    assert result.returncode == 1
    assert "KeyWithoutValue" in result.stdout


def test_phase_40_6_incorrect_delimiter_detected(tmp_path: Path):
    """A colon instead of '=' should trip IncorrectDelimiter."""
    p = tmp_path / "bad.env"
    p.write_text("WRONG_DELIM: value\n", encoding="utf-8")
    result = _run([str(p)])
    assert result.returncode == 1
    assert "IncorrectDelimiter" in result.stdout


def test_phase_40_6_leading_character_detected(tmp_path: Path):
    """A key starting with a digit should trip LeadingCharacter."""
    p = tmp_path / "bad.env"
    p.write_text("1BAD_KEY=value\n", encoding="utf-8")
    result = _run([str(p)])
    assert result.returncode == 1
    assert "LeadingCharacter" in result.stdout


def test_phase_40_6_quote_character_detected(tmp_path: Path):
    """Unbalanced quote should trip QuoteCharacter."""
    p = tmp_path / "bad.env"
    p.write_text('UNBALANCED="missing close\n', encoding="utf-8")
    result = _run([str(p)])
    assert result.returncode == 1
    assert "QuoteCharacter" in result.stdout


def test_phase_40_6_duplicated_key_detected(tmp_path: Path):
    """Same KEY defined twice -> DuplicatedKey on the second occurrence."""
    p = tmp_path / "bad.env"
    p.write_text("FOO=first\nFOO=second\n", encoding="utf-8")
    result = _run([str(p)])
    assert result.returncode == 1
    assert "DuplicatedKey" in result.stdout


def test_phase_40_6_lowercase_key_is_warning_not_error(tmp_path: Path):
    """Lowercase keys are valid Python identifiers but violate convention.
    Should trip LowercaseKey at severity=warning -- but NOT fail the run
    (warnings only fail if no other errors)."""
    p = tmp_path / "warn.env"
    p.write_text("lowercase_key=value\n", encoding="utf-8")
    result = _run([str(p)])
    # warnings alone don't trip exit 1
    assert result.returncode == 0, (
        f"lowercase-only file should exit 0 (warnings don't fail); got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # But the warning should still be reported
    assert "LowercaseKey" in result.stdout


def test_phase_40_6_backend_env_example_is_clean():
    """Canonical template `backend/.env.example` must validate clean.
    Catches regression if a future template edit introduces malformed lines."""
    assert CANONICAL_TEMPLATE.exists(), (
        f"backend/.env.example missing: {CANONICAL_TEMPLATE}"
    )
    result = _run([str(CANONICAL_TEMPLATE)])
    assert result.returncode == 0, (
        f"backend/.env.example must exit 0 (clean); got {result.returncode}\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr}"
    )


def test_phase_40_6_pre_commit_hook_exists_and_executable():
    """Criterion 2 verbatim from masterplan: pre_commit_hook_invokes_it."""
    assert HOOK.exists(), f"pre-commit hook missing: {HOOK}"
    assert HOOK.stat().st_mode & 0o111, (
        f"{HOOK} must be executable (chmod +x)"
    )
    # Verify the hook invokes the script
    content = HOOK.read_text(encoding="utf-8")
    assert "scripts/qa/env_syntax_check.py" in content, (
        "pre-commit hook must call scripts/qa/env_syntax_check.py"
    )


def test_phase_40_6_ci_workflow_exists_and_invokes_script():
    """Criterion 3 verbatim from masterplan: ci_lane_runs_it."""
    assert WORKFLOW.exists(), f"CI workflow missing: {WORKFLOW}"
    content = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/qa/env_syntax_check.py" in content, (
        "CI workflow must invoke scripts/qa/env_syntax_check.py"
    )
    # phase-40.6 soft-launch with continue-on-error
    assert "continue-on-error: true" in content, (
        "phase-40.6 workflow should soft-launch with continue-on-error: true"
    )


def test_phase_40_6_script_no_args_returns_usage_error():
    """Script with no args returns usage info + exit 2 (per documented contract)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"no-args should exit 2 (usage error); got {result.returncode}"
    )
    assert "Usage:" in result.stderr
