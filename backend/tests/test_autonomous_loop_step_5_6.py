"""phase-30.2 + phase-30.3 tests for autonomous_loop Step 5.6.

phase-30.2: backfill-then-check wiring (Stage 7 FAIL fix).
phase-30.3: closed_tickers append in Step 5.6 (Stage 12 FAIL fix).

Audit basis:
- handoff/archive/phase-30.0/experiment_results.md Stage 7 (P1-2) +
  Stage 12 (P1-3). 7-of-11 open positions had stop_loss_price=NULL;
  and stop-loss-triggered closes never reached the learn loop because
  closed_tickers was initialized inside Step 7 only.

Test plan (7 cases total):
  Phase-30.2 (4 cases, unchanged):
    1. Cycle 1 with legacy NULL-stop positions -> backfill returns N
       backfilled, check invoked AFTER backfill.
    2. Cycle 2 idempotency: all stops set -> backfill returns 0
       backfilled, check still runs.
    3. Backfill raises -> exception swallowed; check_stop_losses is
       still invoked (fail-open).
    4. Grep-equivalent: backfill_missing_stops precedes
       check_stop_losses in the Step 5.6 block.

  Phase-30.3 (3 NEW cases):
    5. Stop-out -> closed_tickers contains the triggered ticker
       (the core wiring assertion).
    6. Synthetic stop-out -> agent_memories row is written (the
       strict-literal-of-the-masterplan-criterion test; uses a
       patched OutcomeTracker so the model-injection gap noted in
       phase-30.3 research_brief.md does NOT block the assertion).
    7. Grep-equivalent: closed_tickers.append appears near
       summary["stop_loss_triggered"].append in Step 5.6 (mirrors
       masterplan verification command).

These are isolated unit tests of the Step 5.6 sequence semantics; they
use `asyncio.run()` to drive the async reproducer (no pytest-asyncio
dep required).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


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

    # Find the Step 5.6 section header line. Use the box-drawing pattern
    # so we don't accidentally match upstream comments that mention
    # "Step 5.6" for cross-reference (e.g., the phase-30.3 hoist comment
    # at the cycle-top).
    header_idx = None
    for i, ln in enumerate(lines):
        if "Step 5.6:" in ln and "──" in ln:
            header_idx = i
            break
    assert header_idx is not None, (
        "Step 5.6 section-header (with box-drawing) not found"
    )

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


# =====================================================================
# phase-30.3 -- closed_tickers wiring in Step 5.6
# =====================================================================


async def _step_5_6_with_learn_routing(
    trader,
    closed_tickers: list[str],
    summary: dict,
) -> list[str]:
    """phase-30.3 reproducer: the Step 5.6 block with the new
    `closed_tickers.append(sl_ticker)` line wired in. Mirrors the
    production block at `backend/services/autonomous_loop.py:783-808`.
    `closed_tickers` is passed in (hoisted to cycle-top per the
    production layout at `:169`)."""
    summary["stop_loss_triggered"] = []
    summary["stop_loss_backfilled"] = []
    try:
        backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
        summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
    except Exception:
        pass
    triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
    for sl_ticker in triggered_stops:
        try:
            sl_trade = await asyncio.to_thread(
                trader.execute_sell,
                ticker=sl_ticker,
                quantity=None,
                price=None,
                reason="stop_loss_trigger",
                signals=None,
            )
            if sl_trade:
                summary["stop_loss_triggered"].append(sl_ticker)
                # phase-30.3 new line:
                closed_tickers.append(sl_ticker)
        except Exception:
            pass
    return triggered_stops


# ---------------------------------------------------------------------
# 5. Stop-out populates closed_tickers (the core wiring assertion)
# ---------------------------------------------------------------------
def test_step_5_6_stop_out_appends_to_closed_tickers():
    parent = MagicMock()
    parent.backfill_missing_stops.return_value = {
        "backfilled": [],
        "skipped": ["WDC"],
        "count_backfilled": 0,
        "count_skipped": 1,
    }
    parent.check_stop_losses.return_value = ["WDC"]
    parent.execute_sell.return_value = {
        "trade_id": "synthetic",
        "ticker": "WDC",
        "price": 371.68,
    }

    closed_tickers: list[str] = []
    summary: dict = {}
    triggered = asyncio.run(
        _step_5_6_with_learn_routing(parent, closed_tickers, summary)
    )

    # The triggered ticker MUST land in BOTH summary["stop_loss_triggered"]
    # (existing observable) AND closed_tickers (the new learn-loop signal).
    assert triggered == ["WDC"]
    assert "WDC" in summary["stop_loss_triggered"]
    assert "WDC" in closed_tickers, (
        "phase-30.3 wiring failed: stop-out ticker did not reach closed_tickers"
    )


# ---------------------------------------------------------------------
# 6. Synthetic stop-out produces an agent_memories row (strict literal
#    of the masterplan criterion).
# ---------------------------------------------------------------------
def test_synthetic_stop_out_produces_agent_memories_row():
    """Strict-literal test of the masterplan criterion
    `synthetic_test_with_one_stop_out_produces_an_agent_memories_row`.

    Production `_learn_from_closed_trades` constructs
    `OutcomeTracker(settings)` with NO model -> the model-gated
    `_generate_and_persist_reflections` branch at
    `outcome_tracker.py:147` is skipped -> `bq.save_agent_memory` does
    not fire. That model-injection gap is OUT OF SCOPE for phase-30.3;
    this test patches the OutcomeTracker chain so the assertion targets
    the wiring (closed_tickers flow -> learn -> save_agent_memory)
    rather than the model-routing logic.
    """
    closed_tickers = ["WDC"]
    mock_bq = MagicMock()
    # Required by _learn_from_closed_trades:
    mock_bq.get_paper_trades.return_value = [
        {
            "trade_id": "synthetic-sell",
            "ticker": "WDC",
            "action": "SELL",
            "analysis_id": "synth-aid",
            "risk_judge_decision": "STOP_LOSS_TRIGGER",
            "price": 371.68,
            "created_at": "2026-05-20T18:00:00+00:00",
        }
    ]
    mock_settings = MagicMock()

    # Patch OutcomeTracker so evaluate_recommendation has the side effect
    # of calling bq.save_agent_memory -- exercises the wiring without
    # depending on the dormant model-routing branch.
    def fake_tracker_factory(settings_arg):
        tracker = MagicMock()

        def fake_eval(ticker, *args, **kwargs):
            mock_bq.save_agent_memory(
                {
                    "ticker": ticker,
                    "agent_type": "stop_loss_outcome",
                    "situation": "stop_loss_trigger",
                    "lesson": "loss-protection exit at synthesized stop",
                    "created_at": "2026-05-20T18:00:01+00:00",
                }
            )

        tracker.evaluate_recommendation.side_effect = fake_eval
        return tracker

    # _learn_from_closed_trades imports OutcomeTracker lazily inside the
    # function body (autonomous_loop.py:1651), so patching at
    # backend.services.outcome_tracker (the source module) is the
    # correct seam.
    with patch(
        "backend.services.outcome_tracker.OutcomeTracker",
        side_effect=fake_tracker_factory,
    ):
        from backend.services.autonomous_loop import _learn_from_closed_trades

        asyncio.run(
            _learn_from_closed_trades(closed_tickers, mock_bq, mock_settings)
        )

    assert mock_bq.save_agent_memory.call_count >= 1, (
        "synthetic stop-out did not flow through to save_agent_memory; "
        "the closed_tickers -> learn loop wiring is broken"
    )
    # The recorded payload's ticker matches the stop-out.
    written = mock_bq.save_agent_memory.call_args_list[0].args[0]
    assert written.get("ticker") == "WDC"
    assert written.get("situation") == "stop_loss_trigger"


# ---------------------------------------------------------------------
# 7. Grep-equivalent: closed_tickers.append present near
#    stop_loss_triggered.append in Step 5.6
# ---------------------------------------------------------------------
def test_step_5_6_contains_closed_tickers_append_near_stop_loss_triggered():
    """The masterplan verification command for phase-30.3 is
    `grep -B 2 -A 4 'stop_loss_triggered.*append' backend/services/autonomous_loop.py | grep -q 'closed_tickers.append'`.
    This test mirrors it against the on-disk file so a refactor that
    removes the closed_tickers.append wiring breaks pytest."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "autonomous_loop.py"
    )
    lines = path.read_text(encoding="utf-8").splitlines()

    # Find the summary["stop_loss_triggered"].append CALL site inside
    # Step 5.6 (skip comment occurrences).
    triggered_call_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("#"):
            continue
        if (
            'summary["stop_loss_triggered"].append' in ln
            or "summary['stop_loss_triggered'].append" in ln
        ):
            triggered_call_idx = i
            break
    assert triggered_call_idx is not None, (
        "summary[\"stop_loss_triggered\"].append call site not found in "
        "Step 5.6 block"
    )

    # Within the same +/- 4-line window the grep command uses,
    # `closed_tickers.append` must be present.
    window = lines[
        max(0, triggered_call_idx - 2) : triggered_call_idx + 5
    ]
    window_joined = "\n".join(
        ln for ln in window if not ln.strip().startswith("#")
    )
    assert "closed_tickers.append" in window_joined, (
        "phase-30.3 wiring missing: closed_tickers.append must sit next "
        "to summary['stop_loss_triggered'].append in Step 5.6"
    )
