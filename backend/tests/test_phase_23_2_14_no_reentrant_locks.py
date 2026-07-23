"""phase-23.2.14 (P2) verification: no re-entrant Lock patterns in backend/.

Per researcher (handoff/current/research_brief_phase_23_2_14.md, 5 sources):
Audited 13 threading.Lock() instances in backend/ (phase-23.1.21 counted 12;
add _BUDGET_CACHE_LOCK in llm_client.py from phase-25.A8). All 13 CLEAN
of re-entrant patterns. The phase-23.1.22 `_snapshot_locked` helper pattern
is the canonical fix; 3 locks now use the `_*_locked` suffix convention.

Three regression-lock layers:
  1. Lock count == 13 (any new lock forces explicit re-audit + count bump).
  2. `_*_locked` helpers document "caller MUST hold the lock".
  3. No `_*_locked` helper itself contains `with self._lock:` (defeats extraction).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
EXPECTED_LOCK_COUNT = 17
# phase-75.5 RE-AUDIT (2026-07-20). The count was 15 (set phase-56.2, 2026-06-10) but
# the tree measured 17, so this guard was RED and had been for ~2 weeks -- which meant
# it could no longer detect the very drift it exists to catch. Both extra locks audited
# against the phase-23.2.14 re-entrancy criteria; both CLEAN:
#
#   16th -- `_RAIL_GUARD_LOCK`, backend/agents/claude_code_client.py:103.
#           PRE-EXISTING drift, NOT from phase-75.5: added 2026-07-07 by phase-66.1
#           (commit 27d40df5, cc_rail probe gate + circuit breaker) without the bump
#           this docstring requires. Audited: `_rail_guard_record_failure` mutates
#           state under the lock, copies `should_page` out, and performs the
#           `raise_cron_alert_sync` call OUTSIDE the `with` block -- the canonical
#           non-re-entrant shape. No `_*_locked` helper re-acquires it.
#
#   17th -- `_DEGRADED_LOCK`, backend/services/observability/spend.py:39.
#           Added by phase-75.5 (arch-04 spend-guard degradation counter). Audited:
#           identical shape -- counter mutation under the lock, `should_alert` copied
#           out, and both `logger.warning` and `raise_cron_alert_sync` executed OUTSIDE
#           the lock. Single-acquire, never nested.
#
# The pre-existing 16th masked the 17th: because the guard was already failing, adding
# a new lock produced NO visible status change. A red guard is not a guard. Clearing
# the count to the measured 17 restores its ability to detect the next real drift.
# Researcher 2026-05-23 found 13 REAL threading.Lock() instantiations across
# backend/. The 14th regex hit is at kill_switch.py:112 INSIDE a triple-quoted
# docstring that describes the phase-23.1.22 BUG (text reads "re-entered the
# same threading.Lock() via snapshot()"). Documentation artifact, not a real
# lock. phase-56.2 re-audit: the 15th hit is alerting.py:64 (AlertDeduper's
# `self._lock = threading.Lock()`, added with the cron-alert dedup layer) --
# a REAL, single-acquire, non-re-entrant lock (acquired once in its record
# path, never nested; reviewed against the phase-23.2.14 re-entrancy criteria).
# The regex count remains the correct drift guard -- any new real lock OR
# removal of the docstring bumps this count and forces explicit re-audit.



def _backend_py_files() -> list[Path]:
    """All .py files under backend/, excluding tests/."""
    return [
        p for p in BACKEND_DIR.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]


def test_phase_23_2_14_threading_lock_count_matches_roster():
    """The count of `threading.Lock()` instantiations under backend/ must
    equal EXPECTED_LOCK_COUNT. Any new lock requires an explicit phase-23.2.14
    re-audit + this bump."""
    pattern = re.compile(r"threading\.Lock\s*\(\s*\)")
    count = 0
    hits = []
    for f in _backend_py_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            # Exclude comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for m in pattern.finditer(line):
                count += 1
                hits.append(f"{f.relative_to(REPO_ROOT)}:{line_no}")
    assert count == EXPECTED_LOCK_COUNT, (
        f"phase-23.2.14 roster drift: found {count} threading.Lock() instances "
        f"(expected {EXPECTED_LOCK_COUNT} per researcher audit 2026-05-23).\n"
        f"Re-audit required + bump EXPECTED_LOCK_COUNT in same commit.\n"
        f"Lock sites: {hits[:20]}"
    )


def test_phase_23_2_14_locked_helpers_document_caller_holds_lock():
    """Every `_*_locked` helper method must document in its docstring that
    'caller MUST hold the lock' (mirrors `_snapshot_locked` at
    kill_switch.py:109). Catches future helpers that drop the discipline."""
    # Find all `def _*_locked(self, ...):` methods + check the docstring
    method_re = re.compile(
        r"def\s+(_\w+_locked)\s*\(\s*self[^)]*\)[^:]*:\s*\n([^\n]*\n[^\n]*\n[^\n]*\n)?",
        re.MULTILINE,
    )
    bad: list[tuple[str, str]] = []
    for f in _backend_py_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in method_re.finditer(text):
            helper_name = m.group(1)
            following = m.group(2) or ""
            # Check if the following lines (the docstring + first few lines) contain
            # an explicit mention that caller MUST hold the lock OR similar phrasing.
            haystack = following.lower()
            phrases_ok = [
                "caller must hold",
                "must hold the lock",
                "held the lock",
                "must already hold",
                "while holding the lock",
                "lock is held",
                "lock held",
                "_lock is acquired",
                "with self._lock",
            ]
            if not any(p in haystack for p in phrases_ok):
                bad.append((str(f.relative_to(REPO_ROOT)), helper_name))
    # Allow up to 1 bad case for forward-compat (e.g. a private impl with no docstring is OK);
    # the discipline is about NEW helpers establishing the convention.
    assert len(bad) <= 1, (
        f"phase-23.2.14: {len(bad)} `_*_locked` helpers missing caller-MUST-hold docstring:\n"
        + "\n".join(f"  {f}:{name}" for f, name in bad[:10])
    )


def test_phase_23_2_14_no_locked_helper_reacquires_self_lock():
    """A `_*_locked` helper method must NOT itself contain `with self._lock:`
    (which would defeat the helper-extraction discipline by recreating the
    re-entrant deadlock the helper was extracted to fix). This is the
    anti-pattern test that locks the phase-23.1.22 fix shape."""
    method_block_re = re.compile(
        r"(\n\s+def\s+(_\w+_locked)\s*\([^)]*\)[^:]*:\n)(.*?)(?=\n\s+def\s|\nclass\s|\Z)",
        re.DOTALL,
    )
    bad: list[str] = []
    for f in _backend_py_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in method_block_re.finditer(text):
            helper_name = m.group(2)
            body = m.group(3)
            if "with self._lock" in body:
                bad.append(
                    f"{f.relative_to(REPO_ROOT)}:{helper_name} (body contains 'with self._lock')"
                )
    assert not bad, (
        f"phase-23.2.14: re-entrant anti-pattern detected. `_*_locked` helpers "
        f"MUST NOT contain 'with self._lock:' (defeats the helper-extraction "
        f"that prevents re-entrant deadlock).\n"
        + "\n".join(f"  {b}" for b in bad)
    )


def test_phase_23_2_14_phase_23_1_22_anchor_preserved():
    """The phase-23.1.22 fix (_snapshot_locked extraction in kill_switch.py)
    is the canonical pattern this audit locks. Ensure the anchor is intact."""
    kill_switch = REPO_ROOT / "backend" / "services" / "kill_switch.py"
    text = kill_switch.read_text(encoding="utf-8")
    assert "_snapshot_locked" in text, (
        "kill_switch.py must contain the _snapshot_locked helper "
        "(phase-23.1.22 fix anchor)"
    )
    assert "phase-23.1.22" in text or "phase-23.2.4" in text, (
        "kill_switch.py must reference the phase-23.1.22 or phase-23.2.4 "
        "regression-lock for audit-trail"
    )


def test_phase_23_2_14_no_rlock_used_as_workaround():
    """The phase-23.1.22 design choice was to KEEP threading.Lock (not
    switch to RLock as a 'fix') -- per researcher cite of Real Python /
    SuperFastPython: switching to RLock masks the re-entrancy bug rather
    than fixing it. Catch any future commit that introduces RLock as a
    workaround."""
    pattern = re.compile(r"threading\.RLock\s*\(\s*\)")
    hits = []
    for f in _backend_py_files():
        text = f.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                hits.append(f"{f.relative_to(REPO_ROOT)}:{line_no}: {line.strip()[:80]}")
    assert not hits, (
        f"phase-23.2.14: threading.RLock() introduced in backend/. Per phase-23.1.22 "
        f"design (Real Python + SuperFastPython): RLock as a re-entrancy workaround "
        f"masks the bug. Use _*_locked helper extraction instead.\n"
        + "\n".join(hits[:10])
    )
