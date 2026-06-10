"""phase-23.2.5 (P0) verification: kill-switch breach evaluation never falsely fired.

Closes closure_roadmap.md section 1 P0 line. Per researcher
(handoff/current/research_brief_phase_23_2_5.md, 5 sources read in full):

  * `handoff/kill_switch_audit.jsonl` has 9 historical false-fire rows on
    2026-05-05 (all `trigger=drawdown_breach` with `daily_loss_pct=-2.5`,
    i.e. NAV ABOVE start-of-day -> mathematically impossible breach).
  * Post-fix window (2026-05-06 -> today): 0 false-fires.
  * `grep -rn "drawdown_breach" backend/` returns 0 hits today -- the
    auto-pause-on-breach code path was removed entirely; `evaluate_breach()`
    is now read-only.
  * `backend/services/kill_switch.py:202-236` math is correct (sign-checked
    line-by-line; `daily_loss_pct = (sod - current_nav) / sod * 100.0`
    so a profit returns NEGATIVE and `>=` against a positive limit can
    never be True).

This test file enforces:
  1. Audit-log scan: no auto-pause triggers post 2026-05-06 (the fix date).
  2. Math correctness: 6 concrete cases covering profit / breach-at-limit /
     just-under-limit / trailing-dd / no-state / zero-sod (no DivisionError).
  3. Trigger string regression-lock: `drawdown_breach` MUST NOT reappear in
     backend/ source code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

# Allowed pause triggers AFTER the fix (per researcher audit-log scan)
ALLOWED_POST_FIX_TRIGGERS = {
    "manual",
    "test",
    "test-pre",
    "bench-1",
    "bench-2",
    "bench-3",
    "uat-16.6-drill",
    "phase-30-overnight-remediation",
}

FIX_DATE = "2026-05-06"  # Day after the 2026-05-05 false-fires


def test_phase_23_2_5_no_unexpected_auto_pauses_post_fix():
    """Every pause row dated >= 2026-05-06 must have an allowed trigger.
    Catches any future re-introduction of the false-fire bug (e.g. via
    a misguided "restore auto-pause" refactor)."""
    if not AUDIT_LOG.exists():
        pytest.skip(f"audit log not present: {AUDIT_LOG}")
    bad = []
    with AUDIT_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") != "pause":
                continue
            ts = row.get("ts", "")
            if ts < FIX_DATE:
                continue
            trigger = row.get("trigger")
            if trigger not in ALLOWED_POST_FIX_TRIGGERS:
                bad.append({
                    "ts": ts,
                    "trigger": trigger,
                    "details": row.get("details"),
                })
    assert not bad, (
        f"phase-23.2.5 REGRESSION: unexpected auto-pause triggers post-{FIX_DATE}: "
        f"{bad}\nAllowed: {ALLOWED_POST_FIX_TRIGGERS}"
    )


def test_phase_23_2_5_drawdown_breach_trigger_string_absent_from_backend_source():
    """The auto-fire code path was removed in phase-23.1.x. Re-introducing
    the literal `drawdown_breach` string in backend/ source is a regression
    we want to catch at lint time."""
    result = subprocess.run(
        ["grep", "-rn", "drawdown_breach", "backend/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # grep returns 1 when no matches (success for us); 0 when matches found (failure)
    hits = [ln for ln in result.stdout.splitlines() if ln.strip()]
    # Exclude this test file's own self-references (which we keep for documentation).
    hits = [ln for ln in hits if "test_phase_23_2_5" not in ln]
    assert not hits, (
        f"phase-23.2.5 REGRESSION: 'drawdown_breach' reintroduced in backend/ source:\n"
        + "\n".join(hits[:10])
    )


@pytest.fixture
def isolated_kill_switch_state():
    """Snapshot + reset the module-global kill-switch state for tests.
    The state is rebuilt from the live audit log at import time
    (`_load_from_audit`), so without isolation tests collide with prod state."""
    import backend.services.kill_switch as ks
    orig_sod_nav = ks._state._sod_nav
    orig_sod_date = ks._state._sod_date
    orig_peak_nav = ks._state._peak_nav
    # Clean slate for the test
    ks._state._sod_nav = None
    ks._state._sod_date = None
    ks._state._peak_nav = None
    yield ks
    # Restore
    ks._state._sod_nav = orig_sod_nav
    ks._state._sod_date = orig_sod_date
    ks._state._peak_nav = orig_peak_nav


def test_phase_23_2_5_evaluate_breach_profit_does_not_breach(isolated_kill_switch_state):
    """REGRESSION for 2026-05-05 9x false-fire: NAV above SOD must NOT
    breach the daily-loss limit. The 9 false-fires reported
    daily_loss_pct=-2.5 (i.e. profit) yet triggered an auto-pause."""
    ks = isolated_kill_switch_state
    ks._state._sod_nav = 10000.0
    ks._state._sod_date = "2026-05-22"
    # peak_nav remains None (no trailing-dd factor)
    r = ks.evaluate_breach(
        current_nav=10250.0,
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
    )
    assert r["daily_loss_pct"] == pytest.approx(-2.5), (
        f"profit must produce NEGATIVE daily_loss_pct; got {r['daily_loss_pct']}"
    )
    assert r["daily_loss_breached"] is False, (
        "profit must NEVER breach daily_loss_limit (the 2026-05-05 false-fire root)"
    )
    assert r["any_breached"] is False


def test_phase_23_2_5_evaluate_breach_real_breach_at_limit():
    """A real breach (loss >= limit) MUST set daily_loss_breached True."""
    import backend.services.kill_switch as ks
    # snapshot
    _orig = (ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav)
    ks._state._sod_nav = 10000.0
    ks._state._sod_date = "2026-05-22"
    ks._state._peak_nav = None
    try:
        # 4% loss exactly = breach
        r = ks.evaluate_breach(
            current_nav=9600.0,
            daily_loss_limit_pct=4.0,
            trailing_dd_limit_pct=10.0,
        )
        assert r["daily_loss_pct"] == pytest.approx(4.0)
        assert r["daily_loss_breached"] is True
        assert r["any_breached"] is True
    finally:
        ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav = _orig


def test_phase_23_2_5_evaluate_breach_just_under_limit_no_breach():
    """Loss just under limit (3.99%) must NOT breach (boundary check)."""
    import backend.services.kill_switch as ks
    # snapshot
    _orig = (ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav)
    ks._state._sod_nav = 10000.0
    ks._state._sod_date = "2026-05-22"
    ks._state._peak_nav = None
    try:
        r = ks.evaluate_breach(
            current_nav=9601.0,  # 3.99% loss
            daily_loss_limit_pct=4.0,
            trailing_dd_limit_pct=10.0,
        )
        assert r["daily_loss_breached"] is False
    finally:
        ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav = _orig


def test_phase_23_2_5_evaluate_breach_trailing_dd_at_limit():
    """trailing_dd_limit breach when current is exactly limit-pct below peak."""
    import backend.services.kill_switch as ks
    _orig = (ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav)
    ks._state._sod_nav = None  # neutralize daily-loss factor
    ks._state._peak_nav = 10000.0
    try:
        # 10% drawdown
        r = ks.evaluate_breach(
            current_nav=9000.0,
            daily_loss_limit_pct=4.0,
            trailing_dd_limit_pct=10.0,
        )
        assert r["trailing_dd_breached"] is True
    finally:
        ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav = _orig


def test_phase_23_2_5_evaluate_breach_no_state_returns_no_breach():
    """When sod + peak are both None (e.g. cold start), evaluate_breach
    must return any_breached=False (defensive)."""
    import backend.services.kill_switch as ks
    # Clear state
    ks._state._sod_nav = None
    ks._state._peak_nav = None
    ks._state._sod_date = None
    r = ks.evaluate_breach(
        current_nav=5000.0,
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
    )
    assert r["any_breached"] is False, "cold-start must not auto-breach"


def test_phase_23_2_5_evaluate_breach_zero_sod_does_not_div_zero():
    """sod_nav=0.0 must not raise ZeroDivisionError (the `if sod and sod > 0`
    guard at line 218 covers this)."""
    import backend.services.kill_switch as ks
    _orig = (ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav)
    ks._state._sod_nav = 0.0
    ks._state._sod_date = "2026-05-22"
    try:
        r = ks.evaluate_breach(
            current_nav=10000.0,
            daily_loss_limit_pct=4.0,
            trailing_dd_limit_pct=10.0,
        )
        # No exception + no breach
        assert r["daily_loss_breached"] is False
    finally:
        ks._state._sod_nav, ks._state._sod_date, ks._state._peak_nav = _orig


@pytest.mark.requires_live
@pytest.mark.skipif(
    os.getenv("PYFINAGENT_LIVE_TESTS") != "1",
    reason="live-system state probe: asserts 5-20 historical 'drawdown_breach' rows "
    "in handoff/kill_switch_audit.jsonl; the live log has been rotated since the "
    "2026-05-05 incident so the count reflects file state, not code (phase-56.2 "
    "quarantine; set PYFINAGENT_LIVE_TESTS=1 to assert against the full log)",
)
def test_phase_23_2_5_audit_log_historical_false_fires_documented():
    """The 9 historical false-fires from 2026-05-05 ARE in the audit log
    (not retroactively scrubbed); they're the smoking-gun evidence. This
    test verifies they remain visible so future auditors can find them."""
    if not AUDIT_LOG.exists():
        pytest.skip(f"audit log not present: {AUDIT_LOG}")
    false_fire_count = 0
    with AUDIT_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("trigger") == "drawdown_breach":
                false_fire_count += 1
    # Researcher found 9 such rows; defensive bound 5-20 accommodates minor
    # housekeeping (rotation, partial archive) without breaking the test.
    assert 5 <= false_fire_count <= 20, (
        f"expected 5-20 historical 'drawdown_breach' rows (researcher found 9); "
        f"got {false_fire_count}. If this changed, investigate audit-log rotation/scrubbing."
    )
