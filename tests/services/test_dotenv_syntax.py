"""Pytest for the dotenv syntax validator (phase-23.6.0).

Tests scan against synthetic in-memory fixtures. Does NOT touch the real
`backend/.env` (sandbox-blocked from the harness; would also make tests
non-portable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the repo root to sys.path so we can import the validator module.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.validators.check_dotenv_syntax import scan_text, main  # noqa: E402


# ── scan_text: rule coverage ──────────────────────────────────────


def test_clean_file_returns_no_findings():
    text = "KEY=value\nOTHER=42\n"
    assert scan_text(text) == []


def test_blank_lines_and_comments_skipped():
    text = "\n# a comment\n\nKEY=value\n  # indented comment\n"
    assert scan_text(text) == []


def test_leading_space_after_eq_is_critical():
    """The exact bug pattern: KEY= value -> bash exit 127."""
    text = "ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X\n"
    findings = scan_text(text)
    assert len(findings) == 1
    severity, label, lineno, raw, _desc = findings[0]
    assert severity == "CRITICAL"
    assert label == "leading_space_after_eq"
    assert lineno == 1
    assert raw == "ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X"


def test_leading_whitespace_before_key_is_critical():
    text = " KEY=value\n"
    findings = scan_text(text)
    assert any(f[1] == "leading_space_before_key" for f in findings)


def test_trailing_whitespace_unquoted_is_warning():
    text = "KEY=value   \n"
    findings = scan_text(text)
    severities = {(f[0], f[1]) for f in findings}
    assert ("WARNING", "trailing_whitespace_unquoted") in severities


def test_inline_comment_unquoted_is_warning():
    text = "KEY=value # this comment becomes part of the value\n"
    findings = scan_text(text)
    severities = {(f[0], f[1]) for f in findings}
    assert ("WARNING", "inline_comment_unquoted") in severities


def test_quoted_value_with_inline_comment_is_clean():
    """Inside a quoted value, the inline-comment rule should NOT fire."""
    text = 'KEY="value # this is fine inside quotes"\n'
    findings = scan_text(text)
    # The inline-comment rule is skipped for quoted values; trailing-
    # whitespace rule is also skipped.
    assert not any(f[1] in {"trailing_whitespace_unquoted", "inline_comment_unquoted"} for f in findings)


def test_missing_trailing_newline_is_info():
    text = "KEY=value"  # no \n
    findings = scan_text(text)
    severities = {(f[0], f[1]) for f in findings}
    assert ("INFO", "missing_trailing_newline") in severities


def test_present_trailing_newline_no_info():
    text = "KEY=value\n"
    findings = scan_text(text)
    assert not any(f[1] == "missing_trailing_newline" for f in findings)


def test_realistic_dirty_env_file_finds_three_critical():
    """Mirror the historical phase-23.3.5 pattern: 3 leading-space lines
    (24, 25, 56). Synthetic file places them at lines 1, 2, 3 for clarity.
    """
    text = (
        "ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X\n"
        "FRED_API_KEY= c0379d038fc49bb50e3a8c0cd4d1eb0a\n"
        "ANTHROPIC_API_KEY= sk-ant-api03-FOO\n"
    )
    findings = scan_text(text)
    crit = [f for f in findings if f[0] == "CRITICAL"]
    assert len(crit) == 3
    assert [f[2] for f in crit] == [1, 2, 3]


def test_idempotency_clean_file_run_twice():
    """Running the validator twice on the same input must yield identical findings."""
    text = "KEY=value\n"
    assert scan_text(text) == scan_text(text)


def test_idempotency_dirty_file_run_twice():
    text = "BAD= value\n"
    f1 = scan_text(text)
    f2 = scan_text(text)
    assert f1 == f2


# ── main: exit-code semantics ──────────────────────────────────────


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(content, encoding="utf-8")
    return p


def test_main_clean_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = _write(tmp_path, "KEY=value\n")
    assert main([str(p)]) == 0


def test_main_critical_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = _write(tmp_path, "BAD= value\n")
    assert main([str(p)]) == 1


def test_main_warning_default_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Without --strict, WARNING findings do not gate exit code."""
    p = _write(tmp_path, "KEY=value   \n")
    assert main([str(p)]) == 0


def test_main_warning_strict_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = _write(tmp_path, "KEY=value   \n")
    assert main(["--strict", str(p)]) == 1


def test_main_missing_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "does-not-exist.env"
    assert main([str(p)]) == 2
