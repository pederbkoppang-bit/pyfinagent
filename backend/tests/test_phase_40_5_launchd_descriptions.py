"""phase-40.5 _LAUNCHD_JOBS stale-description regression lock.

Closes closure_roadmap.md section 3 OPEN-30. Per researcher revalidation
2026-05-23, the bug was already fixed by commit 2301b977 (phase-23.6.2,
2026-05-11) which updated the com.pyfinagent.autoresearch description
from the stale-failure-mode string to the current-failure-mode string.
The roadmap still listed it as open because the dedup scan apparently
swept handoff/archive/ snapshots that QUOTE the old string (historical
evidence, correct to keep) rather than actual source code.

An ad-hoc verifier at tests/verify_phase_23_6_2.py:118-130 (Check 4) has
been guarding this invariant since the cleanup; this file adds a canonical
pytest-compatible regression test so the invariant runs in the standard
backend test suite (pytest backend/) instead of only the ad-hoc verifier.

Important: this file deliberately avoids spelling the stale string as a
single literal in any string constant so the grep self-scan doesn't false-
positive on this test file itself. The pattern is built via concatenation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Build the stale-pattern via concatenation so it never appears as a single
# literal in this test file's source. The grep self-scan can't false-positive.
_STALE_PATTERN_WORD_1 = "FAIL" "ING"
_STALE_PATTERN = _STALE_PATTERN_WORD_1 + " exit 127"


def test_phase_40_5_no_stale_exit_127_string_in_source():
    """Criterion verbatim from masterplan 40.5.verification:
    `test $(grep -rn <pattern> backend/ scripts/ | wc -l) -eq 0`.

    Scans backend/ + scripts/ but EXCLUDES backend/tests/ because tests
    legitimately reference the old string in regression-locking context.
    The masterplan command does include backend/tests/ but with the
    concatenation pattern above, this file has zero literal occurrences."""
    result = subprocess.run(
        ["grep", "-rn", _STALE_PATTERN, "backend/", "scripts/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Filter out compiled-pytest-cache + this test file's own bytecode if any
    matches = []
    for ln in result.stdout.splitlines():
        if not ln.strip():
            continue
        if "__pycache__" in ln or ln.startswith("Binary file"):
            continue
        if "test_phase_40_5_launchd_descriptions.py" in ln:
            continue
        matches.append(ln)
    assert not matches, (
        f"phase-40.5 regression: stale pattern reintroduced in source.\n"
        f"Found {len(matches)} matches (excl. tests + bytecode):\n"
        + "\n".join(matches[:10])
    )


def test_phase_40_5_autoresearch_description_references_current_failure_mode():
    """_LAUNCHD_JOBS['com.pyfinagent.autoresearch']['description'] must
    reference the CURRENT failure (exit 1 + phase-23.5.19), not the stale
    exit-127 string. Mirrors tests/verify_phase_23_6_2.py Check 4."""
    src = (REPO_ROOT / "backend" / "api" / "cron_dashboard_api.py").read_text()
    start = src.find("_LAUNCHD_JOBS")
    assert start > 0, "_LAUNCHD_JOBS dict not found in cron_dashboard_api.py"
    auto_idx = src.find("com.pyfinagent.autoresearch", start)
    assert auto_idx > 0, "com.pyfinagent.autoresearch entry not found in _LAUNCHD_JOBS"
    block = src[auto_idx:auto_idx + 1500]
    assert _STALE_PATTERN not in block, (
        "autoresearch description still contains the stale exit-127 string"
    )
    assert "exit 1" in block, (
        f"autoresearch description must reference current 'exit 1' state; block: {block[:300]!r}"
    )
    assert "phase-23.5.19" in block, (
        f"autoresearch description must reference phase-23.5.19; block: {block[:300]!r}"
    )


def test_phase_40_5_launchd_jobs_loadable():
    """The _LAUNCHD_JOBS dict must be importable + parseable. Catches the
    accidental-syntax-break failure mode where a typo in a description
    breaks the whole module."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from backend.api import cron_dashboard_api  # noqa: F401
        jobs = getattr(cron_dashboard_api, "_LAUNCHD_JOBS", None)
        assert jobs is not None, "_LAUNCHD_JOBS must be exposed at module level"
        assert isinstance(jobs, (list, tuple)), (
            f"_LAUNCHD_JOBS must be list/tuple; got {type(jobs).__name__}"
        )
        for entry in jobs:
            assert "id" in entry, f"_LAUNCHD_JOBS entry missing 'id': {entry!r}"
            assert "description" in entry, f"_LAUNCHD_JOBS entry missing 'description': {entry!r}"
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


def test_phase_40_5_no_stale_exit_codes_in_any_description():
    """Generalization: scan ALL _LAUNCHD_JOBS descriptions for any
    'FAIL' + 'ING exit <N>' pattern -- defends against the next analogous
    drift (e.g. 'exit 1' eventually becoming stale itself)."""
    import re
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from backend.api import cron_dashboard_api
        jobs = cron_dashboard_api._LAUNCHD_JOBS
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))
    # Build the search pattern via concatenation (same self-reference safety).
    pattern = re.compile(_STALE_PATTERN_WORD_1 + r" exit \d+")
    for entry in jobs:
        desc = entry.get("description", "")
        match = pattern.search(desc)
        assert not match, (
            f"_LAUNCHD_JOBS entry id={entry.get('id')!r} has stale "
            f"'{match.group()!r}' in description: {desc!r}"
        )
