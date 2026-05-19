"""phase-30.2 tests for autonomous_loop Step 5.6 backfill-then-check wiring.

Audit basis: handoff/archive/phase-30.0/experiment_results.md Stage 7
(FAIL) + P1-2: `paper_trader.py::backfill_missing_stops` (phase-25.2)
exists but had zero production callers, leaving 7 of 11 open positions
with `stop_loss_price=NULL`. phase-30.2 wires the helper into Step 5.6
BEFORE `check_stop_losses` so legacy positions get a synthesized stop
on the next cycle.

Test plan (4 cases):
  1. Cycle 1 with legacy NULL-stop positions -> backfill returns N
     backfilled, check is invoked AFTER backfill.
  2. Cycle 2 idempotency: all stops set -> backfill returns 0
     backfilled, check still runs.
  3. Backfill raises -> exception swallowed; check_stop_losses is
     still invoked (fail-open).
  4. Grep-equivalent assertion: backfill_missing_stops appears in the
     Step 5.6 block of autonomous_loop.py and precedes check_stop_losses.
     Mirrors the masterplan verification command so a refactor that
     unwires the call breaks pytest.

These are isolated unit tests of the Step 5.6 sequence semantics; they
use `asyncio.run()` to drive the async reproducer (no pytest-asyncio
dep required).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


async def _step_5_6_under_test(trader, summary: dict) -> tuple[list, list]:
    """Reproduces the production Step 5.6 sequence (backfill then check)
    so that we can assert call ordering against a mocked PaperTrader.
    The same shape is in
    `backend/services/autonomous_loop.py::run_daily_cycle` Step 5.6
    (the `# phase-30.2:` block). Keeping the reproduction in the test
    file avoids importing the 1700-line module under test.
    """
    summary["stop_loss_triggered"] = []
    summary["stop_loss_backfilled"] = []
    try:
        backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
        summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
    except Exception:
        # Fail-open: backfill exception must not block check_stop_losses.
        pass
    triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
    return triggered_stops, summary["stop_loss_backfilled"]


# ---------------------------------------------------------------------
# 1. Cycle 1 -- legacy NULL-stop positions: backfill runs BEFORE check
# ---------------------------------------------------------------------
def test_step_5_6_backfill_runs_before_check_stop_losses():
    parent = MagicMock()
    parent.backfill_missing_stops.return_value = {
        "backfilled": [
            {"ticker": "WDC", "entry_price": 404.0, "stop_loss_price": 371.68},
            {"ticker": "SNDK", "entry_price": 989.9, "stop_loss_price": 910.71},
        ],
        "skipped": ["COHR", "MU"],
        "count_backfilled": 2,
        "count_skipped": 2,
    }
    parent.check_stop_losses.return_value = ["WDC"]  # WDC triggers immediately

    summary: dict = {}
    triggered, backfilled = asyncio.run(_step_5_6_under_test(parent, summary))

    # Both methods invoked exactly once.
    assert parent.backfill_missing_stops.call_count == 1
    assert parent.check_stop_losses.call_count == 1

    # Order: backfill BEFORE check in the recorded call sequence.
    method_names = [c[0] for c in parent.method_calls]
    backfill_idx = method_names.index("backfill_missing_stops")
    check_idx = method_names.index("check_stop_losses")
    assert backfill_idx < check_idx, (
        f"backfill_missing_stops must precede check_stop_losses; "
        f"got method_calls={method_names!r}"
    )

    # Verdict surfaces both effects.
    assert len(backfilled) == 2
    assert {b["ticker"] for b in backfilled} == {"WDC", "SNDK"}
    assert triggered == ["WDC"]
    assert summary["stop_loss_backfilled"] == backfilled


# ---------------------------------------------------------------------
# 2. Cycle 2 -- idempotency: nothing left to backfill, check still runs
# ---------------------------------------------------------------------
def test_step_5_6_idempotent_backfill_no_op():
    parent = MagicMock()
    parent.backfill_missing_stops.return_value = {
        "backfilled": [],
        "skipped": ["WDC", "SNDK", "COHR", "MU"],
        "count_backfilled": 0,
        "count_skipped": 4,
    }
    parent.check_stop_losses.return_value = []

    summary: dict = {}
    triggered, backfilled = asyncio.run(_step_5_6_under_test(parent, summary))

    assert parent.backfill_missing_stops.call_count == 1
    assert parent.check_stop_losses.call_count == 1
    assert backfilled == []
    assert triggered == []


# ---------------------------------------------------------------------
# 3. Fail-open -- backfill raises; check_stop_losses still invoked
# ---------------------------------------------------------------------
def test_step_5_6_backfill_exception_does_not_block_check():
    parent = MagicMock()
    parent.backfill_missing_stops.side_effect = RuntimeError("BQ unavailable")
    parent.check_stop_losses.return_value = ["KEYS"]  # unrelated stop fires

    summary: dict = {}
    triggered, backfilled = asyncio.run(_step_5_6_under_test(parent, summary))

    # Both methods STILL invoked (fail-open contract).
    assert parent.backfill_missing_stops.call_count == 1
    assert parent.check_stop_losses.call_count == 1
    # Backfill effect was lost to the exception; check effect is preserved.
    assert backfilled == []
    assert triggered == ["KEYS"]


# ---------------------------------------------------------------------
# 4. Verification-command sanity: grep symbol present near Step 5.6
# ---------------------------------------------------------------------
def test_autonomous_loop_step_5_6_contains_backfill_symbol():
    """The masterplan verification command is
    `grep -A 5 'Step 5.6' backend/services/autonomous_loop.py | grep -q 'backfill_missing_stops'`.
    This test mirrors that grep predicate against the on-disk file so a
    pytest run catches any future refactor that breaks the wiring."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "autonomous_loop.py"
    )
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find the Step 5.6 header line.
    header_idx = None
    for i, ln in enumerate(lines):
        if "Step 5.6" in ln:
            header_idx = i
            break
    assert header_idx is not None, "Step 5.6 header not found"

    # Identify the actual CALL sites (not comment mentions). A call site is
    # a line containing `await asyncio.to_thread(trader.<method>` -- the
    # production async-wrap pattern. Limiting the scan to lines whose left-
    # margin is unindented-comment-free ('#' is not the first non-whitespace
    # character) filters out the docstring/comment occurrences.
    window_lines = lines[header_idx : header_idx + 50]
    backfill_call_line = None
    check_call_line = None
    for i, ln in enumerate(window_lines):
        stripped = ln.strip()
        if stripped.startswith("#"):
            continue
        if backfill_call_line is None and "trader.backfill_missing_stops" in ln:
            backfill_call_line = i
        if check_call_line is None and "trader.check_stop_losses" in ln:
            check_call_line = i
    assert backfill_call_line is not None, (
        "trader.backfill_missing_stops must be called inside Step 5.6 "
        "(non-comment line) per phase-30.2 wiring"
    )
    assert check_call_line is not None, (
        "trader.check_stop_losses must be called inside Step 5.6"
    )
    assert backfill_call_line < check_call_line, (
        f"backfill_missing_stops must precede check_stop_losses (call sites); "
        f"got backfill at offset {backfill_call_line}, check at {check_call_line}"
    )
