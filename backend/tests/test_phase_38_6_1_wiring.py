"""phase-38.6.1 verification: cycle_lock wired into autonomous_loop + main.py.

Per masterplan 38.6.1 criteria:
  1. autonomous_loop imports cycle_lock.acquire
  2. _running guard at line 142 replaced with acquire context manager
  3. main.py lifespan calls clean_stale_lock at startup
  4. existing test_phase_38_6_restart_survivable still passes

ZERO new behavior: the cycle_lock module already exists from phase-38.6
(cycle 43). This step only wires it in. Tests verify the wiring is
structurally correct.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTONOMOUS_LOOP = REPO_ROOT / "backend" / "services" / "autonomous_loop.py"
MAIN_PY = REPO_ROOT / "backend" / "main.py"


def test_phase_38_6_1_autonomous_loop_imports_cycle_lock():
    """Criterion 1: autonomous_loop.py imports cycle_lock.acquire."""
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    assert "from backend.services.cycle_lock import" in text, (
        "autonomous_loop.py must import from backend.services.cycle_lock"
    )
    # And specifically the acquire + CycleLockError symbols
    assert "acquire" in text and "CycleLockError" in text, (
        "autonomous_loop.py must import both acquire and CycleLockError"
    )


def test_phase_38_6_1_running_guard_uses_acquire_context_manager():
    """Criterion 2: the _running guard is replaced with a cycle_lock
    acquire + CycleLockError catch."""
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    # The phase-38.6.1 comment must be present
    assert "phase-38.6.1" in text, "autonomous_loop.py must reference phase-38.6.1"
    # The acquire call site
    assert "_cycle_lock_acquire(" in text, (
        "autonomous_loop.py must call _cycle_lock_acquire(...)"
    )
    # The CycleLockError catch with already_running_file_lock reason
    assert "already_running_file_lock" in text, (
        "autonomous_loop.py must return reason='already_running_file_lock' on lock contention"
    )


def test_phase_38_6_1_release_in_finally_block():
    """The lock context manager must be released in the finally block
    (idempotent: NameError caught if dry-run path didn't set _lock_cm)."""
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    # The finally block must include _lock_cm.__exit__ with NameError guard
    pattern = re.compile(
        r"_lock_cm\.__exit__\(None,\s*None,\s*None\)[^\n]*\n[^\n]*except\s*\(NameError,\s*AttributeError\)",
        re.DOTALL,
    )
    assert pattern.search(text), (
        "autonomous_loop.py finally block must call _lock_cm.__exit__(...) + "
        "catch NameError/AttributeError (idempotent for dry-run path)"
    )


def test_phase_38_6_1_main_py_lifespan_calls_clean_stale_lock():
    """Criterion 3: main.py lifespan calls clean_stale_lock at startup."""
    text = MAIN_PY.read_text(encoding="utf-8")
    assert "from backend.services.cycle_lock import clean_stale_lock" in text, (
        "main.py must import clean_stale_lock from cycle_lock module"
    )
    assert "_clean_stale_lock(reason=\"startup_recovery\")" in text, (
        "main.py must call clean_stale_lock(reason='startup_recovery') at startup"
    )
    # Must reference phase-38.6.1
    assert "phase-38.6.1" in text, "main.py must reference phase-38.6.1"


def test_phase_38_6_1_main_py_recovery_is_fail_open():
    """main.py recovery hook must be wrapped in try/except (fail-open
    per existing convention)."""
    text = MAIN_PY.read_text(encoding="utf-8")
    # Locate the phase-38.6.1 block
    start = text.find("phase-38.6.1: cycle-lock stale-recovery hook")
    end = text.find("phase-38.6.1: cycle_lock recovery hook failed (fail-open)")
    assert start > 0 and end > start, (
        "main.py must have phase-38.6.1 recovery block AND its fail-open except handler"
    )
    block = text[start:end + 100]
    assert "try:" in block and "except Exception:" in block, (
        "phase-38.6.1 block must be wrapped in try/except"
    )


def test_phase_38_6_1_running_flag_still_set_for_ui_status():
    """Honest scope: the in-process _running flag is KEPT (for UI/api
    status surface), but the LOCK is the source of truth. Verify
    _running = True/False still appears in the function body."""
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    assert "_running = True" in text, "in-process _running flag still set for UI status"
    assert "_running = False" in text, "in-process _running flag still cleared in finally"


def test_phase_38_6_1_acquire_imported_at_function_scope():
    """The cycle_lock import is at function scope (lazy) to avoid
    circular-import risk + match the existing import pattern at
    autonomous_loop.py for backend.services.* deps."""
    text = AUTONOMOUS_LOOP.read_text(encoding="utf-8")
    # The import must be inside run_daily_cycle, after the function def
    func_start = text.find("def run_daily_cycle")
    import_idx = text.find("from backend.services.cycle_lock import", func_start)
    next_def = text.find("\ndef ", func_start + 1)
    if next_def < 0:
        next_def = len(text)
    assert func_start < import_idx < next_def, (
        "cycle_lock import must be inside run_daily_cycle function scope (lazy)"
    )
