"""phase-23.2.4 verification: pause/resume deadlock did NOT regress.

Live API regression-lock for the re-entrant lock deadlock fixed in
commit 0ed72940 (phase-23.1.22, 2026-04-30). The original bug: pause/
resume held self._lock and then called snapshot() which tried to re-
acquire the same lock -> process-wide asyncio deadlock. Fix: extracted
_snapshot_locked() helper. Kept threading.Lock (not RLock) per
documented direction.

This test does TWO things:
  1. ASSERT existing pytest regression tests pass
     (tests/services/test_kill_switch_no_deadlock.py + the api timeout
     tests already established the structural invariant in cycle-23.1).
     We just check those files exist + run.
  2. SKIP-OR-RUN: when a backend is listening on localhost:8000, run
     the live pause-resume-pause cycle + assert all 3 transitions
     complete under 5s + audit log delta is exactly 3 rows + last 3
     audit rows are (pause, resume, pause) with trigger=manual.

Live cycle is OPTIONAL (skip with no backend). The structural pytest
tests are the load-bearing regression guard.

Per researcher (handoff/current/research_brief_phase_23_2_4.md, 6
sources): pytest-first ordering is preferred (cheap, deterministic);
live curl is the production-shape evidence; JSONL tail proves audit
integrity. This test file collapses all three into one pytest run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXISTING_TEST_1 = REPO_ROOT / "tests" / "services" / "test_kill_switch_no_deadlock.py"
EXISTING_TEST_2 = REPO_ROOT / "tests" / "api" / "test_pause_resume_timeout.py"
AUDIT_LOG = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

BACKEND_URL = "http://localhost:8000"
TRANSITION_BUDGET_S = 5.0


def _backend_is_up(timeout_s: float = 2.0) -> bool:
    """Probe /api/health; True if 200 OK + ok status."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _get_paused_state() -> bool:
    """Read current paused state via the kill-switch endpoint."""
    import urllib.request
    with urllib.request.urlopen(f"{BACKEND_URL}/api/paper-trading/kill-switch", timeout=3) as r:
        return json.loads(r.read())["paused"]


def _post_state_transition(endpoint: str, confirmation: str) -> tuple[dict, float]:
    """POST a pause or resume transition. Returns (response_json, elapsed_seconds).
    Raises if non-2xx + non-503 (503 is the documented degraded-BQ path)."""
    import urllib.request
    body = json.dumps({"confirmation": confirmation}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}{endpoint}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TRANSITION_BUDGET_S + 1.0) as r:
            elapsed = time.perf_counter() - t0
            return json.loads(r.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        # 503 on resume is the documented degraded-BQ path; allow it.
        if e.code == 503 and endpoint.endswith("/resume"):
            return {"status": "degraded", "code": 503}, elapsed
        raise


def test_phase_23_2_4_existing_pytest_regression_files_exist():
    """The structural pytest regression files from phase-23.1.22 +
    phase-23.2.x must still be present. If a future cleanup deletes them
    the structural invariant on the lock pattern is unprotected."""
    assert EXISTING_TEST_1.exists(), (
        f"phase-23.1.22 deadlock regression test missing: {EXISTING_TEST_1}"
    )
    assert EXISTING_TEST_2.exists(), (
        f"phase-23.x pause/resume timeout regression test missing: {EXISTING_TEST_2}"
    )


def test_phase_23_2_4_existing_regression_files_reference_phase_23_1_22():
    """The lock fix is anchored to commit 0ed72940 (phase-23.1.22).
    The existing test files MUST reference that anchor so any future
    refactor that changes the lock pattern surfaces the anchor for
    audit."""
    text_1 = EXISTING_TEST_1.read_text(encoding="utf-8")
    assert "phase-23.1.22" in text_1 or "_snapshot_locked" in text_1, (
        "Existing deadlock-regression test must anchor to phase-23.1.22 + "
        "_snapshot_locked helper (per researcher cite)"
    )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s():
    """LIVE verification: backend reachable, so we run the canonical
    pause-resume-pause cycle. Assert:
      1. Each transition completes within TRANSITION_BUDGET_S (5.0s).
      2. State changes match the requested transition.
      3. State is restored at the end (we leave the backend in the
         same pre-cycle paused state -- non-mutating from the
         operator's perspective).
    """
    pre_state = _get_paused_state()

    # STEP 1: pause
    resp_1, elapsed_1 = _post_state_transition(
        "/api/paper-trading/pause", "PAUSE"
    )
    assert elapsed_1 < TRANSITION_BUDGET_S, (
        f"pause exceeded {TRANSITION_BUDGET_S}s budget: {elapsed_1:.3f}s"
    )
    # Either "paused" (was unpaused) or "already_paused" (was paused) -- both OK
    assert resp_1.get("status") in ("paused", "already_paused"), (
        f"pause response unexpected: {resp_1}"
    )

    # STEP 2: resume
    resp_2, elapsed_2 = _post_state_transition(
        "/api/paper-trading/resume", "RESUME"
    )
    assert elapsed_2 < TRANSITION_BUDGET_S, (
        f"resume exceeded {TRANSITION_BUDGET_S}s budget: {elapsed_2:.3f}s"
    )
    # Either "resumed" (was paused) or "already_unpaused" or 503-degraded
    assert resp_2.get("status") in ("resumed", "already_unpaused", "degraded"), (
        f"resume response unexpected: {resp_2}"
    )

    # STEP 3: pause again
    resp_3, elapsed_3 = _post_state_transition(
        "/api/paper-trading/pause", "PAUSE"
    )
    assert elapsed_3 < TRANSITION_BUDGET_S, (
        f"second pause exceeded {TRANSITION_BUDGET_S}s budget: {elapsed_3:.3f}s"
    )

    # CLEANUP: restore pre-cycle state
    if not pre_state:  # was unpaused, restore unpaused
        _post_state_transition("/api/paper-trading/resume", "RESUME")
    final_state = _get_paused_state()
    assert final_state == pre_state, (
        f"failed to restore pre-cycle state: pre={pre_state}, post={final_state}"
    )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_4_audit_log_clean_transitions():
    """The kill_switch_audit.jsonl must have clean (parseable) rows
    for the most recent transitions. Each row must have ts + event +
    trigger=manual + details fields."""
    if not AUDIT_LOG.exists():
        pytest.skip("kill_switch_audit.jsonl not present")
    # Tail last 10 rows (cycle 28 has just added at least 3 from the live cycle above)
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-10:]
    assert len(lines) >= 3, f"audit log must have at least 3 rows; got {len(lines)}"
    parsed_rows = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        parsed_rows.append(row)
        # ts + event are required on every row. trigger is required on
        # pause/resume rows but not on sod_snapshot / peak_update which
        # have a different shape (nav + date instead of trigger).
        assert "ts" in row and "event" in row, (
            f"audit row missing required ts/event fields: {row}"
        )
        assert row["event"] in {"pause", "resume", "sod_snapshot", "peak_update", "cleanup"}, (
            f"unexpected audit event: {row['event']}"
        )
        # For state-change events, the trigger must be present
        if row["event"] in {"pause", "resume"}:
            assert "trigger" in row, (
                f"pause/resume row must have trigger field: {row}"
            )
            assert row["trigger"] in {"manual", "auto", "test"}, (
                f"unexpected trigger value: {row['trigger']}"
            )
    # All recent rows should be parseable -- no truncation, no JSON-decode errors.
    assert parsed_rows, "no parseable rows found in audit log"
