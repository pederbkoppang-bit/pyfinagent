"""phase-38.5 ASCII-only logger audit script tests.

Tests the new scripts/qa/ascii_logger_check.py helper. The script itself
is stdlib-only; these tests inject synthetic Python source via tmp_path
and verify the audit catches the right patterns + ignores the right ones.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qa" / "ascii_logger_check.py"


@pytest.fixture
def synth_root(tmp_path: Path) -> Path:
    """A clean tmp directory with no violations by default."""
    (tmp_path / "module.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.info('clean ASCII message %s', 'with-arg')\n",
        encoding="utf-8",
    )
    return tmp_path


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_phase_38_5_script_exists_and_executable():
    assert SCRIPT.exists(), f"script missing: {SCRIPT}"


def test_phase_38_5_clean_codebase_exits_zero(synth_root: Path):
    result = _run(["--roots", str(synth_root)])
    assert result.returncode == 0, (
        f"clean codebase should exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_phase_38_5_em_dash_in_logger_info_is_violation(tmp_path: Path):
    """Canonical case: em-dash (U+2014) inside logger.info() string literal."""
    (tmp_path / "bad.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.info('Step 1 -- result: %s', 'ok')\n"  # this is ASCII -- (two hyphens)
        "logger.error('FAIL " + chr(0x2014) + " bad em-dash')\n",  # actual em-dash
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path)])
    assert result.returncode == 1, "should exit 1 on em-dash violation"
    assert "U+2014" in result.stdout, f"should report U+2014; got: {result.stdout}"


def test_phase_38_5_arrow_unicode_caught(tmp_path: Path):
    """Right-arrow (U+2192) -- frequent escapee."""
    (tmp_path / "arrow.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.info('a " + chr(0x2192) + " b')\n",
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path)])
    assert result.returncode == 1
    assert "U+2192" in result.stdout


def test_phase_38_5_fstring_literal_part_is_checked(tmp_path: Path):
    """An f-string can have a literal part (Constant) + an interpolation
    (FormattedValue). Only the literal part should be audited; runtime
    interpolation can be anything (we can't know at lint time)."""
    bad = chr(0x2014)  # em-dash in the LITERAL part
    (tmp_path / "fs.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        f"x = 'irrelevant'\nlogger.info(f'literal{bad}part {{x}}')\n",
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path)])
    assert result.returncode == 1
    assert "U+2014" in result.stdout


def test_phase_38_5_non_logger_attribute_call_ignored(tmp_path: Path):
    """A non-logger attribute call like `print()` or `requests.get()` with a
    non-ASCII string MUST be ignored. Otherwise we trip on every i18n test."""
    (tmp_path / "ignored.py").write_text(
        "x = 'irrelevant " + chr(0x2014) + "'\n"  # bare assignment
        "print('hello " + chr(0x2014) + "')\n"  # non-logger call
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.info('clean')\n",  # logger call with clean string
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path)])
    assert result.returncode == 0, (
        f"non-logger non-ASCII should be ignored; got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_phase_38_5_syntax_error_file_warns_not_crashes(tmp_path: Path, capsys):
    """A file with a SyntaxError should produce a stderr WARN line and be
    skipped (not crash the whole run)."""
    (tmp_path / "bad_syntax.py").write_text(
        "def broken(\n",  # truncated; will SyntaxError on parse
        encoding="utf-8",
    )
    (tmp_path / "good.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.info('clean')\n",
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path)])
    # Should not crash + should still exit 0 (no violations in good.py)
    assert result.returncode == 0
    # The skip warning should be on stderr
    assert "SyntaxError" in result.stderr or "skipped" in result.stderr


def test_phase_38_5_json_output_format(tmp_path: Path):
    """--json emits one JSON record per violation (machine-parseable for CI)."""
    import json as _json
    (tmp_path / "j.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "logger.error('bad " + chr(0x2014) + "')\n",
        encoding="utf-8",
    )
    result = _run(["--roots", str(tmp_path), "--json"])
    assert result.returncode == 1
    # First non-empty stdout line should be valid JSON
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, f"no JSON output; got: {result.stdout}"
    record = _json.loads(lines[0])
    assert record["method"] == "error"
    assert record["codepoint"] == "U+2014"
    assert "line" in record and "col" in record and "path" in record


def test_phase_38_5_real_codebase_clean_post_sweep():
    """Post phase-38.5.1 sweep: ascii_logger_check.py must exit 0 against
    the real backend/ + scripts/ tree. Cycle 21 found 151 violations;
    cycle 42 swept them (22 files + 4 files, 126 lines edited). This test
    locks the new CLEAN invariant; any future regression trips it."""
    result = _run(["--roots", str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")])
    assert result.returncode == 0, (
        f"real codebase must be CLEAN of non-ASCII logger violations "
        f"post phase-38.5.1 sweep; got exit {result.returncode}\n"
        f"stdout: {result.stdout[:2000]}"
    )
