"""phase-23.2.22: immutable verification.

Asserts:
1. tests/services/test_cycle_failure_alerts.py has an autouse tmp_audit
   fixture that monkeypatches kill_switch._AUDIT_PATH.
2. tests/services/test_kill_switch_no_deadlock.py has the same.
3. backend/services/portfolio_manager.py has the diagnostic log line at
   the position-cap break point.
4. tests/services/test_position_cap_logging.py exists with the expected
   test names.
5. handoff/kill_switch_audit.jsonl ends with a cleanup row + a resume row
   (phase=23.2.22) so boot replay restores the unpaused state.
6. Running pytest on the two formerly-polluting test files does NOT grow
   the production audit log (size identical before/after).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_test_cycle_failure_alerts_isolated():
    rel = "tests/services/test_cycle_failure_alerts.py"
    text = _read(rel)
    ast.parse(text)
    assert "_isolated_kill_switch_audit" in text, \
        "tmp_audit autouse fixture missing"
    assert 'monkeypatch.setattr(kill_switch, "_AUDIT_PATH"' in text, \
        "fixture must monkeypatch kill_switch._AUDIT_PATH"
    assert "@pytest.fixture(autouse=True)" in text, \
        "fixture must be autouse=True"
    return f"OK {rel}"


def check_test_kill_switch_no_deadlock_isolated():
    rel = "tests/services/test_kill_switch_no_deadlock.py"
    text = _read(rel)
    ast.parse(text)
    assert "_isolated_kill_switch_audit" in text, \
        "tmp_audit autouse fixture missing"
    assert 'monkeypatch.setattr(kill_switch, "_AUDIT_PATH"' in text, \
        "fixture must monkeypatch kill_switch._AUDIT_PATH"
    assert "@pytest.fixture(autouse=True)" in text, \
        "fixture must be autouse=True"
    return f"OK {rel}"


def check_portfolio_manager_log():
    rel = "backend/services/portfolio_manager.py"
    text = _read(rel)
    ast.parse(text)
    assert "Position cap reached" in text, \
        "position-cap diagnostic log line missing"
    # Ensure the log lives BEFORE the break (regex: matches the diagnostic
    # call, then any whitespace, then `break`).
    pattern = re.compile(
        r'logger\.info\(\s*\n?\s*"Position cap reached.*?break',
        re.DOTALL,
    )
    assert pattern.search(text), \
        "diagnostic log must precede the position-cap break"
    return f"OK {rel}"


def check_test_position_cap_logging():
    rel = "tests/services/test_position_cap_logging.py"
    text = _read(rel)
    ast.parse(text)
    for fn in (
        "test_position_cap_emits_diagnostic_log_when_full",
        "test_position_cap_does_not_log_when_room_remains",
    ):
        assert fn in text, f"missing test: {fn}"
    return f"OK {rel}"


def check_audit_cleanup_marker():
    """Verify (a) the cleanup row is present AND (b) the actual boot-replay
    state from the live module yields paused=False. Q/A-2 advisory: a revert
    of just the v2 resume would silently re-introduce the latent risk if we
    only checked for "any resume after cleanup". The boot-replay invariant
    is the right thing to assert -- if KillSwitchState().snapshot() says
    paused=True after a revert, that's a real regression."""
    rel = "handoff/kill_switch_audit.jsonl"
    text = _read(rel).strip().splitlines()
    cleanup_idx = None
    for i, line in enumerate(text):
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") == "cleanup" and row.get("trigger") == "phase-23.2.22":
            cleanup_idx = i
            break
    assert cleanup_idx is not None, \
        "cleanup row tagged phase-23.2.22 missing from kill_switch_audit.jsonl"
    # The real invariant: boot replay must yield paused=False. This catches
    # any future pause-after-cleanup that was not followed by a resume.
    sys.path.insert(0, str(ROOT))
    from backend.services.kill_switch import KillSwitchState  # noqa: E402
    snap = KillSwitchState().snapshot()
    assert snap["paused"] is False, \
        f"boot replay yields paused=True after cleanup -- audit log has unmatched pause-after-cleanup. snap={snap}"
    return f"OK {rel} -- cleanup row present + boot replay paused=False"


def check_pytest_does_not_grow_audit_log():
    """Run pytest on the THREE formerly-polluting test files and assert the
    production audit log size does not change. test_pause_resume_timeout was
    the 3rd polluting file -- it invokes the real /pause endpoint via
    pause_trading() which writes through the module-level _state singleton."""
    rel = "handoff/kill_switch_audit.jsonl"
    audit = ROOT / rel
    size_before = audit.stat().st_size
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/services/test_cycle_failure_alerts.py",
            "tests/services/test_kill_switch_no_deadlock.py",
            "tests/api/test_pause_resume_timeout.py",
            "-q", "--no-header",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        raise AssertionError(
            f"pytest on the formerly-polluting files failed (exit {proc.returncode}):\n{tail}"
        )
    size_after = audit.stat().st_size
    assert size_after == size_before, (
        f"pytest grew handoff/kill_switch_audit.jsonl by {size_after - size_before} bytes "
        f"(was {size_before}, now {size_after}) -- the autouse tmp_audit fixture is not isolating writes"
    )
    return f"OK {rel} -- pytest run did not grow the production audit log"


def check_test_pause_resume_timeout_isolated():
    rel = "tests/api/test_pause_resume_timeout.py"
    text = _read(rel)
    ast.parse(text)
    assert "_isolated_kill_switch_audit" in text, \
        "tmp_audit autouse fixture missing — pause_trading() calls in this file leak to prod"
    assert 'monkeypatch.setattr(kill_switch, "_AUDIT_PATH"' in text, \
        "fixture must monkeypatch kill_switch._AUDIT_PATH"
    assert "@pytest.fixture(autouse=True)" in text, \
        "fixture must be autouse=True"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_test_cycle_failure_alerts_isolated,
        check_test_kill_switch_no_deadlock_isolated,
        check_test_pause_resume_timeout_isolated,
        check_portfolio_manager_log,
        check_test_position_cap_logging,
        check_audit_cleanup_marker,
        check_pytest_does_not_grow_audit_log,
    ]
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
    print(f"\nphase-23.2.22 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
