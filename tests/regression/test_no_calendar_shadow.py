"""phase-16.37 (#52) regression: prevent backend/calendar stdlib shadow.

Background: phase-16.34 renamed `backend/calendar/` -> `backend/econ_calendar/`
to eliminate a stdlib-shadow that broke `cd backend && python -c "import
calendar"` (the local package shadowed the standard library `calendar`
module). This test guards against accidental re-introduction.

Three checks:
1. `cd backend && python -c "import calendar; print(calendar.__file__)"`
   resolves to the stdlib path (NOT a local backend file).
2. `calendar` is in `sys.stdlib_module_names` (Python 3.10+ canonical
   stdlib registry).
3. `backend/calendar/` directory does NOT exist on disk; only
   `backend/econ_calendar/` does.

If any of these regresses, fix the root cause (rename the local module)
rather than papering over the test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_calendar_imports_stdlib_when_cwd_is_backend():
    """When cwd=backend, `import calendar` must still resolve to stdlib."""
    backend_dir = REPO_ROOT / "backend"
    assert backend_dir.exists(), f"missing backend dir: {backend_dir}"

    result = subprocess.run(
        [sys.executable, "-c", "import calendar; print(calendar.__file__)"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"import calendar failed under cwd=backend: stderr={result.stderr!r}"
    )

    resolved_path = result.stdout.strip()
    assert resolved_path, f"empty path output: {result.stdout!r}"

    # Stdlib calendar should NEVER live under our backend tree, and should
    # NEVER have the substring "econ_calendar" (which would be wrong even
    # if we accidentally ship a calendar/calendar.py inside econ_calendar).
    assert "econ_calendar" not in resolved_path, (
        f"calendar import shadowed by econ_calendar variant: {resolved_path}"
    )
    assert str(REPO_ROOT) not in resolved_path or "/lib/" in resolved_path, (
        f"calendar import resolved to project tree (shadow regression!): "
        f"{resolved_path}"
    )

    # Sanity: stdlib path always contains either "python" or "lib" in some form
    lower = resolved_path.lower()
    assert "python" in lower or "lib" in lower, (
        f"resolved path does not look like stdlib: {resolved_path}"
    )


def test_calendar_in_stdlib_module_names():
    """`calendar` must be in Python's canonical stdlib registry (3.10+)."""
    assert hasattr(sys, "stdlib_module_names"), (
        "sys.stdlib_module_names missing (Python < 3.10?)"
    )
    assert "calendar" in sys.stdlib_module_names, (
        "calendar is not in sys.stdlib_module_names; this should never happen"
    )


def test_no_backend_calendar_directory_exists():
    """The legacy `backend/calendar/` dir must not be re-created."""
    legacy = REPO_ROOT / "backend" / "calendar"
    canonical = REPO_ROOT / "backend" / "econ_calendar"

    assert not legacy.exists(), (
        f"backend/calendar/ re-introduced -- this shadows stdlib! "
        f"Rename to backend/econ_calendar/ (see phase-16.34). Path: {legacy}"
    )
    assert canonical.exists() and canonical.is_dir(), (
        f"backend/econ_calendar/ should exist (it's the canonical name "
        f"after phase-16.34): {canonical}"
    )
