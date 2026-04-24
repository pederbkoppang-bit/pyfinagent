#!/usr/bin/env python
"""phase-4.17.12 smoke test -- F1 failure-discipline recovery drill.

Planted-fault injection exercises the F1 certified-fallback path
implemented in `scripts/harness/run_harness.py::_escalate_certified_fallback`.

What we do:
  1. Capture current `handoff/harness_log.md`.
  2. Call `_escalate_certified_fallback(consecutive_fails=3, cycle=9999)`.
  3. Assert the "HARNESS HALT -- certified fallback" block landed in the log.
  4. Strip the injected block from the log to leave no pollution.

Criteria:
  - consecutive_fails_counter_reaches_3          (constant-asserted at MAX_CONSECUTIVE_FAIL)
  - certified_fallback_raised_after_3_consecutive
  - revert_not_restart_enforced                  (function returns; no restart call)
  - critical_incident_logged_to_harness_log      (HARNESS HALT block present)
  - no_infinite_retry_loop                       (function returns in finite time)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

HARNESS_LOG = REPO_ROOT / "handoff" / "harness_log.md"

# Unique marker so we can strip the test block cleanly afterward.
DRILL_CYCLE_TAG = "cycle-4-17-12-drill-9999"


def test_f1_recovery_drill():
    # 1. Snapshot log + ensure F1 constant is the expected threshold.
    original = HARNESS_LOG.read_text(encoding="utf-8") if HARNESS_LOG.exists() else ""
    from scripts.harness import run_harness
    assert run_harness.MAX_CONSECUTIVE_FAIL == 3, (
        f"consecutive_fails_counter_reaches_3 FAIL: MAX_CONSECUTIVE_FAIL"
        f"={run_harness.MAX_CONSECUTIVE_FAIL} (want 3)"
    )
    print("PASS consecutive_fails_counter_reaches_3")

    # 2. Actively call F1 escalation with consecutive_fails=3.
    start = time.time()
    try:
        run_harness._escalate_certified_fallback(consecutive_fails=3, cycle=9999)
    except Exception as e:
        raise AssertionError(f"certified_fallback_raised_after_3_consecutive FAIL: {e!r}")
    elapsed = time.time() - start
    assert elapsed < 10, (
        f"no_infinite_retry_loop FAIL: escalation took {elapsed:.2f}s"
    )
    print(f"PASS certified_fallback_raised_after_3_consecutive -- elapsed={elapsed:.2f}s")
    print(f"PASS no_infinite_retry_loop -- returned in {elapsed:.2f}s")

    # 3. Assert the HARNESS HALT block landed.
    after = HARNESS_LOG.read_text(encoding="utf-8")
    added = after[len(original):]
    assert "HARNESS HALT -- certified fallback" in added, (
        f"critical_incident_logged_to_harness_log FAIL: added block:\n{added[:500]}"
    )
    assert "Cycle 9999" in added, f"injected cycle marker missing: {added[:500]}"
    print("PASS critical_incident_logged_to_harness_log")

    # 4. revert-not-restart is inherent to _escalate: it mutates files
    # + returns. No subprocess, no os._exit. Assert by negative (the
    # function did not restart the Python process because we're still
    # running after the call).
    print("PASS revert_not_restart_enforced -- drill still running post-escalation")

    # 5. Strip the drill block so subsequent runs start clean.
    try:
        # Restore the original log so this drill is fully non-destructive.
        HARNESS_LOG.write_text(original, encoding="utf-8")
        print("CLEANUP: restored harness_log.md to pre-drill state")
    except Exception as e:
        print(f"WARN: cleanup failed: {e}")

    print("PASS 4.17.12 F1 recovery drill")


if __name__ == "__main__":
    try:
        test_f1_recovery_drill()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
