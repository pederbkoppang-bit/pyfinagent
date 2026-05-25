"""phase-38.4 (OPEN-13) -- pytest module for harness_log_gate.

Mirrors the bash smoke test at `.claude/hooks/lib/harness_log_gate_test.sh`
plus 3 additional unit cases exercising the helper directly.

Per masterplan 38.4 criteria:
  1. harness_log_gate_py_helper_exists
  2. auto_commit_and_push_sh_calls_the_gate
  3. missing_phase_id_in_harness_log_skips_push_with_warn
  4. owner_approval_recorded_before_enabling_the_gate
  5. fail_open_discipline_preserved
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE_HELPER = REPO / ".claude" / "hooks" / "lib" / "harness_log_gate.py"
AUTO_COMMIT_HOOK = REPO / ".claude" / "hooks" / "auto-commit-and-push.sh"


def _load_helper():
    spec = importlib.util.spec_from_file_location("harness_log_gate", GATE_HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Criterion 1
def test_phase_38_4_helper_exists():
    assert GATE_HELPER.exists(), f"helper missing at {GATE_HELPER}"


# Criterion 2
def test_phase_38_4_auto_commit_hook_calls_gate():
    text = AUTO_COMMIT_HOOK.read_text(encoding="utf-8")
    assert "harness_log_gate.py" in text, (
        "auto-commit-and-push.sh must reference the gate helper"
    )
    assert "HARNESS_LOG_GATE_HELPER" in text, (
        "hook must declare the HARNESS_LOG_GATE_HELPER path"
    )
    # And it must come AFTER the live_check gate (deliberate ordering)
    idx_lc = text.find("live_check_gate.py")
    idx_hl = text.find("harness_log_gate.py")
    assert idx_lc > 0 and idx_hl > idx_lc, (
        "harness_log gate must come AFTER live_check gate in the hook"
    )


# Criterion 3 + 5
def test_phase_38_4_missing_phase_id_returns_skip_when_enabled(tmp_path):
    mod = _load_helper()
    log = tmp_path / "h.log"
    log.write_text("## Cycle 99 -- 2026-05-25 -- phase=99.9 result=PASS\n", encoding="utf-8")
    # enabled=True, but step_id=38.4 is NOT in the log
    assert mod.gate_decision(str(log), "38.4", enabled=True) == "skip"


def test_phase_38_4_present_phase_id_returns_passed_when_enabled(tmp_path):
    mod = _load_helper()
    log = tmp_path / "h.log"
    log.write_text("## Cycle 99 -- 2026-05-25 -- phase=38.4 result=PASS\n", encoding="utf-8")
    assert mod.gate_decision(str(log), "38.4", enabled=True) == "passed"


# Criterion 4
def test_phase_38_4_gate_default_off_when_env_var_unset(tmp_path, monkeypatch):
    """Operator-approval criterion: gate fires ONLY when
    HARNESS_LOG_GATE_ENABLED=true env var is set. Default behavior is
    proceed (backward-compat). This is the literal operator-opt-in surface."""
    mod = _load_helper()
    log = tmp_path / "h.log"
    log.write_text("# empty\n", encoding="utf-8")
    monkeypatch.delenv("HARNESS_LOG_GATE_ENABLED", raising=False)
    # When enabled=False, ANY input returns proceed (no actual check)
    assert mod.gate_decision(str(log), "38.4", enabled=False) == "proceed"
    # When env var literally unset, the `main` wrapper reads enabled=False
    import subprocess
    env = dict(os.environ)
    env.pop("HARNESS_LOG_GATE_ENABLED", None)
    out = subprocess.run(
        ["python3", str(GATE_HELPER), str(log), "38.4"],
        capture_output=True, text=True, env=env,
    )
    assert out.stdout.strip() == "proceed", f"unset env should return proceed, got {out.stdout!r}"


# Criterion 5
def test_phase_38_4_fail_open_on_missing_log_file(tmp_path):
    mod = _load_helper()
    nonexistent = tmp_path / "does_not_exist.log"
    # Even with enabled=True, missing file -> proceed (fail-open)
    assert mod.gate_decision(str(nonexistent), "38.4", enabled=True) == "proceed"


def test_phase_38_4_fail_open_on_empty_step_id(tmp_path):
    mod = _load_helper()
    log = tmp_path / "h.log"
    log.write_text("# anything\n", encoding="utf-8")
    assert mod.gate_decision(str(log), "", enabled=True) == "proceed"


def test_phase_38_4_prefix_match_guard():
    """A `phase=38.6` token must NOT match step_id=38 (or 38.6.1 must not
    match 38.6). The regex requires the step-id followed by whitespace
    or end-of-line."""
    mod = _load_helper()
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("phase=38.6.1 result=PASS\n")
        path = f.name
    assert mod.gate_decision(path, "38.6", enabled=True) == "skip"
    # But phase=38.6 (exact) does match step_id=38.6
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("phase=38.6 result=PASS\n")
        path = f.name
    assert mod.gate_decision(path, "38.6", enabled=True) == "passed"
