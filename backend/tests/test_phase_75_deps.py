"""phase-75.13: Python dependency integrity -- lockfile, undeclared runtime
deps, dead + implicit declarations.

Six immutable criteria (verbatim from .claude/masterplan.json step 75.13),
one test group per criterion. These tests deliberately go BEYOND the
step's own verification command, which is a text-`assert` chain with
measured weaknesses (research_brief_75.13.md "Vacuous-guard analysis"):
the command's `==`-count includes comment lines, its `'<name>' in r`
checks are whole-file substrings satisfied by a mere comment mention, and
it never asserts pytest/the loud-fail seam/the lock header/freeze
equality at all. Every test below parses real requirement LINES (comment
stripped) so a name mentioned only in a comment does not satisfy it --
see test_requirements_txt_pyyaml_comment_only_does_not_satisfy_parsed_test
for the documented command-vs-test delta (mutation M6).

1. backend/requirements.lock: real ==-pinned lines (not comment `==`),
   >=150 count, exchange* prefix, header comment (regen command + sync
   commands) as the first non-empty content.
2. backend/requirements.txt: six new real declaration lines
   (exchange-calendars HYPHEN form, numpy, PyYAML CAPS, pytest,
   python-dateutil, google-cloud-storage) + xlrd's enhanced comment +
   fpdf2 fully gone (parsed names AND whole-file substring).
3. .github/workflows/pip-audit.yml: yaml-parsed run step actually
   invokes --requirement backend/requirements.lock, and the lock is in
   the push/pull_request paths filters.
4. scripts/autoresearch/run_memo.py: the new find_spec("gpt_researcher")
   guard fails loudly (rc=1) and runs BEFORE _embedding_preflight()
   (behavioral -- monkeypatched find_spec + a preflight stub that raises
   if ever called).
5. scripts/autoresearch/requirements-autoresearch.txt: gpt-researcher
   pinned + its embedding closure.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_TXT = REPO_ROOT / "backend" / "requirements.txt"
REQUIREMENTS_LOCK = REPO_ROOT / "backend" / "requirements.lock"
PIP_AUDIT_YML = REPO_ROOT / ".github" / "workflows" / "pip-audit.yml"
AUTORESEARCH_REQS = REPO_ROOT / "scripts" / "autoresearch" / "requirements-autoresearch.txt"
RUN_MEMO_PATH = REPO_ROOT / "scripts" / "autoresearch" / "run_memo.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_REQ_LINE_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*(==|>=)\s*([0-9A-Za-z.\-+_]+)\s*$"
)


def _parse_requirements(text: str) -> dict[str, tuple[str, str]]:
    """Parse REAL requirement lines only: strip trailing comments, skip
    blanks, skip anything that isn't a bare `name(==|>=)version` line.
    Returns {normalized_name: (operator, version)}. A package named only
    inside a comment produces NO entry -- this is what makes this a
    parsed-line test rather than the immutable command's whole-file
    substring check."""
    pins: dict[str, tuple[str, str]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_LINE_RE.match(line)
        if not m:
            continue
        name, op, version = m.group(1), m.group(2), m.group(3)
        norm = re.sub(r"[-_.]+", "-", name).lower()
        pins[norm] = (op, version)
    return pins


# ─────────────────────────────────────────────────────────────────────
# Criterion 2 -- backend/requirements.txt real declaration lines
# ─────────────────────────────────────────────────────────────────────

def test_requirements_txt_six_new_pins_are_real_lines_not_comments():
    pins = _parse_requirements(_read(REQUIREMENTS_TXT))
    expected_exact = {
        "exchange-calendars": "4.13.2",
        "numpy": "2.4.4",
        "pyyaml": "6.0.3",
        "pytest": "9.0.3",
        "python-dateutil": "2.9.0.post0",
        "google-cloud-storage": "3.10.1",
    }
    for norm_name, version in expected_exact.items():
        assert norm_name in pins, (
            f"{norm_name} missing as a real requirement line "
            f"(a comment mention would not satisfy this parsed-line test)"
        )
        op, ver = pins[norm_name]
        assert op == "==", f"{norm_name} must be an exact pin (==), got {op!r}"
        assert ver == version, f"{norm_name} version mismatch: {ver!r} != {version!r}"


def test_requirements_txt_pyyaml_exact_case():
    text = _read(REQUIREMENTS_TXT)
    # The immutable verification command's assert ('PyYAML' in r) is
    # case-sensitive; confirm the real line (not a comment) carries it.
    assert re.search(r"(?m)^PyYAML==6\.0\.3\b", text), (
        "PyYAML must appear with exact CAPS case as a real requirement line"
    )
    assert not re.search(r"(?m)^pyyaml==", text), "lowercase 'pyyaml==' line would be a distinct (wrong) declaration"


def test_requirements_txt_exchange_calendars_hyphen_form_only():
    text = _read(REQUIREMENTS_TXT)
    # pip freeze emits the underscore form; the requirements.txt line must
    # be the hyphen (canonical PyPI) spelling per the research-gate gotcha.
    assert re.search(r"(?m)^exchange-calendars==4\.13\.2\b", text)
    assert "exchange_calendars" not in text, "underscore form must not appear in requirements.txt"


def test_requirements_txt_xlrd_comment_enhanced_and_still_parses():
    text = _read(REQUIREMENTS_TXT)
    line = next((ln for ln in text.splitlines() if ln.strip().startswith("xlrd")), None)
    assert line is not None, "xlrd declaration missing"
    assert "read_excel" in line.lower() or "pandas" in line.lower(), (
        "xlrd comment must name the pandas .xls engine role"
    )
    assert "macro_regime.py:" in line and "154" in line, "xlrd comment must anchor the GPR cache consumer"
    pins = _parse_requirements(text)
    assert "xlrd" in pins


def test_requirements_txt_fpdf2_fully_removed():
    text = _read(REQUIREMENTS_TXT)
    # Whole-file substring check (mirrors the immutable command's own
    # 'fpdf2 not in r' assert) AND the parsed-names check.
    assert "fpdf2" not in text.lower(), "fpdf2 residue found (comment or declaration)"
    pins = _parse_requirements(text)
    assert "fpdf2" not in pins


# ─────────────────────────────────────────────────────────────────────
# Criterion 1 -- backend/requirements.lock
# ─────────────────────────────────────────────────────────────────────

def test_requirements_lock_real_pin_count_and_exchange_prefix():
    text = _read(REQUIREMENTS_LOCK)
    real_pin_lines = [
        ln for ln in text.splitlines()
        if "==" in ln and not ln.strip().startswith("#")
    ]
    # Stricter than the immutable command: excludes comment-`==` lines,
    # so a padded/malformed lock with comment noise would not inflate
    # this count (research_brief_75.13.md "measure-don't-assert" weakness).
    assert len(real_pin_lines) >= 150, f"lock has too few real pinned lines: {len(real_pin_lines)}"
    assert any(ln.lower().startswith("exchange") for ln in real_pin_lines), (
        "no exchange* pin found in the lock"
    )


def test_requirements_lock_opens_with_header_comment_containing_regen_command():
    lines = _read(REQUIREMENTS_LOCK).splitlines()
    first_nonempty = next(ln for ln in lines if ln.strip())
    assert first_nonempty.startswith("#"), "lock must open with a header comment, not a pin"
    header = "\n".join(lines[:30])
    assert "pip freeze > backend/requirements.lock" in header, (
        "header missing the exact regeneration command"
    )
    assert (
        "pip install -r backend/requirements.lock" in header
        or "uv pip sync backend/requirements.lock" in header
    ), "header missing a documented sync command"


# ─────────────────────────────────────────────────────────────────────
# Criterion 3 -- .github/workflows/pip-audit.yml (yaml-parsed, not grep)
# ─────────────────────────────────────────────────────────────────────

def test_pip_audit_yml_run_step_targets_the_lock():
    doc = yaml.safe_load(_read(PIP_AUDIT_YML))
    steps = doc["jobs"]["audit"]["steps"]
    run_cmds = [s["run"] for s in steps if "run" in s]
    assert any("--requirement backend/requirements.lock" in cmd for cmd in run_cmds), (
        "no run step invokes pip-audit against backend/requirements.lock "
        "(a comment mention would not satisfy this yaml-parsed test)"
    )


def test_pip_audit_yml_paths_filters_include_the_lock():
    doc = yaml.safe_load(_read(PIP_AUDIT_YML))
    # PyYAML's YAML-1.1 resolver parses the bare `on:` key as the boolean
    # True, not the string "on" -- handle both forms defensively.
    on_block = doc.get("on", doc.get(True))
    assert on_block is not None, "workflow missing an on: trigger block"
    push_paths = on_block["push"]["paths"]
    pr_paths = on_block["pull_request"]["paths"]
    assert "backend/requirements.lock" in push_paths
    assert "backend/requirements.lock" in pr_paths


# ─────────────────────────────────────────────────────────────────────
# Criterion 5 -- scripts/autoresearch/requirements-autoresearch.txt
# ─────────────────────────────────────────────────────────────────────

def test_autoresearch_requirements_manifest_pins_gpt_researcher():
    assert AUTORESEARCH_REQS.exists(), "scripts/autoresearch/requirements-autoresearch.txt missing"
    pins = _parse_requirements(_read(AUTORESEARCH_REQS))
    assert "gpt-researcher" in pins
    op, ver = pins["gpt-researcher"]
    assert op == "==" and ver == "0.14.8"


# ─────────────────────────────────────────────────────────────────────
# Criterion 4 -- run_memo.py find_spec guard (behavioral)
# ─────────────────────────────────────────────────────────────────────

def _load_run_memo():
    """Load run_memo.py as a fresh module object via its file path (it is
    a standalone script, not part of an importable package)."""
    spec = importlib.util.spec_from_file_location("run_memo_under_test_75_13", RUN_MEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_memo_guard_fails_loud_and_precedes_embedding_preflight(monkeypatch):
    module = _load_run_memo()
    monkeypatch.setattr(sys, "argv", ["run_memo.py"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name == "gpt_researcher":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    def _preflight_must_not_be_called(*a, **kw):
        raise AssertionError(
            "_embedding_preflight must NOT be called when gpt_researcher "
            "is missing -- the guard must short-circuit first"
        )

    monkeypatch.setattr(module, "_embedding_preflight", _preflight_must_not_be_called)

    rc = module.main()
    assert rc == 1, f"expected rc=1 on missing gpt_researcher, got {rc}"


def test_run_memo_guard_passes_through_when_gpt_researcher_present(monkeypatch):
    """Sanity counterpart: when find_spec finds gpt_researcher, the guard
    must NOT short-circuit -- control reaches _embedding_preflight (here
    stubbed to itself return a skip message so we don't run GPTResearcher
    or need network access in the test)."""
    module = _load_run_memo()
    monkeypatch.setattr(sys, "argv", ["run_memo.py"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name == "gpt_researcher":
            return object()  # any non-None spec-like sentinel
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    calls = []

    def _preflight_stub():
        calls.append(True)
        return "skip-message"

    monkeypatch.setattr(module, "_embedding_preflight", _preflight_stub)

    rc = module.main()
    assert calls, "_embedding_preflight should have been reached when gpt_researcher IS importable"
    assert rc == 0, f"expected rc=0 (preflight skip path), got {rc}"
