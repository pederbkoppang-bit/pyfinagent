"""phase-35.1 learn-loop writer fan-out tests.

Verifies the fix from closure_roadmap.md §3 + §9 + cycle 12 BQ-probe
finding: outcome_tracking + agent_memories were SCHEMA-EMPTY because
autonomous_loop._learn_from_closed_trades' dispatcher never called
_generate_and_persist_reflections AND evaluate_recommendation
early-returned when yfinance current_price was missing.

This test exercises the fix:
  - Flag OFF (default) -> NO new BQ writes (backward-compat)
  - Flag ON + evaluate_recommendation returns a real outcome
    -> outcome_tracking row (via primary path) + agent_memories
       reflections (via _generate_and_persist_reflections fan-out)
  - Flag ON + evaluate_recommendation returns None (early-return
    on missing yfinance current_price) -> fallback outcome_tracking
    row written from trade fields + agent_memories fan-out still
    fires

No real BQ calls; bigquery_client is mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_bq():
    """A bigquery_client mock that records save_outcome + save_agent_memory calls."""
    bq = MagicMock()
    bq.get_paper_trades.return_value = [
        {
            "ticker": "COHR",
            "action": "SELL",
            "reason": "stop_loss_trigger",
            "price": 378.32,
            # phase-47.7: REAL paper_trades P&L field is realized_pnl_pct (NOT
            # return_pct). Using the real field here makes the fallback-path test a
            # genuine guard against the field bug -- it fails on code that reads
            # the non-existent return_pct (-> 0.0) and passes once the loop reads
            # realized_pnl_pct.
            "realized_pnl_pct": 17.89,
            "holding_days": 25,
            "analysis_id": "2026-04-27T00:00:00",
            "created_at": "2026-05-22T18:35:45+00:00",
            "risk_judge_decision": "",
        }
    ]
    bq.get_report.return_value = None
    bq.save_outcome = MagicMock(return_value=None)
    bq.save_agent_memory = MagicMock(return_value=None)
    return bq


@pytest.fixture
def settings_flag_off():
    from backend.config.settings import Settings
    s = Settings()
    s.paper_learn_loop_enabled = False
    s.gemini_model = "gemini-2.5-pro"
    return s


@pytest.fixture
def settings_flag_on():
    from backend.config.settings import Settings
    s = Settings()
    s.paper_learn_loop_enabled = True
    s.gemini_model = "gemini-2.5-pro"
    return s


def _run_learn(bq, settings):
    from backend.services.autonomous_loop import _learn_from_closed_trades
    asyncio.run(_learn_from_closed_trades(["COHR"], bq, settings))


def test_phase_35_1_flag_off_no_new_writes_backward_compat(mock_bq, settings_flag_off):
    """When paper_learn_loop_enabled=False (default), the writer fan-out is
    inert: only the legacy evaluate_recommendation path runs (which may or
    may not write to outcome_tracking depending on yfinance availability).
    save_agent_memory MUST NOT be called -- that's the new fan-out which
    must stay gated."""
    with patch("backend.services.outcome_tracker.OutcomeTracker") as MockTracker:
        instance = MockTracker.return_value
        instance.evaluate_recommendation.return_value = {
            "ticker": "COHR", "analysis_date": "2026-04-27T00:00:00",
            "recommendation": "HOLD", "return_pct": 17.89, "holding_days": 25,
        }
        instance._generate_and_persist_reflections = MagicMock()
        _run_learn(mock_bq, settings_flag_off)

        # legacy path still called
        assert instance.evaluate_recommendation.called
        # new fan-out NOT called (flag is off)
        assert not instance._generate_and_persist_reflections.called
        # fallback writer also NOT called (flag is off)
        assert not mock_bq.save_outcome.called


def test_phase_35_1_flag_on_real_outcome_fires_reflections(mock_bq, settings_flag_on):
    """Flag ON + evaluate_recommendation returns a real outcome ->
    _generate_and_persist_reflections is called with the outcome + a
    full_report dict (empty if bq.get_report returns None)."""
    with patch("backend.services.outcome_tracker.OutcomeTracker") as MockTracker:
        instance = MockTracker.return_value
        instance.evaluate_recommendation.return_value = {
            "ticker": "COHR", "analysis_date": "2026-04-27T00:00:00",
            "recommendation": "HOLD", "return_pct": 17.89, "holding_days": 25,
        }
        instance._generate_and_persist_reflections = MagicMock()
        _run_learn(mock_bq, settings_flag_on)

        assert instance.evaluate_recommendation.called
        # new fan-out IS called when flag is on + outcome is non-None
        assert instance._generate_and_persist_reflections.called
        call_args = instance._generate_and_persist_reflections.call_args
        outcome_arg, full_report_arg = call_args.args[0], call_args.args[1]
        assert outcome_arg["ticker"] == "COHR"
        assert outcome_arg["return_pct"] == 17.89
        assert full_report_arg == {}  # bq.get_report returned None -> empty dict


def test_phase_35_1_flag_on_yfinance_early_return_triggers_fallback(mock_bq, settings_flag_on):
    """Flag ON + evaluate_recommendation returns None (yfinance flake
    early-return) -> fallback writer fires bq.save_outcome with trade
    fields + reflections fan-out still fires with the synthetic outcome."""
    with patch("backend.services.outcome_tracker.OutcomeTracker") as MockTracker:
        instance = MockTracker.return_value
        instance.evaluate_recommendation.return_value = None  # early-return
        instance._generate_and_persist_reflections = MagicMock()
        _run_learn(mock_bq, settings_flag_on)

        # fallback path: bq.save_outcome called directly with trade fields
        assert mock_bq.save_outcome.called
        saved = mock_bq.save_outcome.call_args
        assert saved.kwargs["ticker"] == "COHR"
        assert saved.kwargs["return_pct"] == 17.89
        assert saved.kwargs["holding_days"] == 25
        assert saved.kwargs["beat_benchmark"] is True  # +17.89% pnl

        # reflections fan-out also fires on synthetic outcome
        assert instance._generate_and_persist_reflections.called


def test_phase_86_25_empty_risk_judge_decision_becomes_UNKNOWN_not_HOLD(
        mock_bq, settings_flag_on):
    """REPLACES `test_phase_35_1_empty_risk_judge_decision_coerced_to_hold`,
    which pinned the DEFECT.

    The old test asserted `rec_arg == "HOLD"` and described the coercion as
    something the dispatcher "MUST" do so OutcomeTracker "doesn't barf on the
    empty string". Avoiding a barf was a real need; spelling the marker "HOLD"
    was the wrong way to meet it. "HOLD" is IN-DOMAIN -- a considered neutral
    call -- so once persisted no reader can tell "the analyst said hold" from
    "we had nothing to say". That is PEP 661's named anti-pattern, and the test
    froze it in place.

    phase-86.25 resolves the vocabulary at the boundary instead: the dispatcher
    hands `evaluate_recommendation` a real analyst recommendation or an explicit
    out-of-domain UNKNOWN. The empty string is still never passed through, so
    the original concern is still met.

    This test is kept (not deleted) and still drives the REAL
    `_learn_from_closed_trades`, because it is the behavioural coverage of that
    call site: revert the fix and this goes red.
    """
    from backend.services.recommendation_vocab import (
        UNKNOWN_RECOMMENDATION, is_directional,
    )

    with patch("backend.services.outcome_tracker.OutcomeTracker") as MockTracker:
        instance = MockTracker.return_value
        instance.evaluate_recommendation.return_value = {
            "ticker": "COHR", "analysis_date": "2026-04-27T00:00:00",
            "recommendation": UNKNOWN_RECOMMENDATION,
            "return_pct": 17.89, "holding_days": 25,
        }
        instance._generate_and_persist_reflections = MagicMock()
        _run_learn(mock_bq, settings_flag_on)

        call_args = instance.evaluate_recommendation.call_args
        ticker_arg, date_arg, rec_arg, price_arg = call_args.args

        assert rec_arg == UNKNOWN_RECOMMENDATION, (
            f"got {rec_arg!r}; the dispatcher must pass an explicit unknown, not a "
            "fabricated HOLD and not a risk-approval token"
        )
        assert rec_arg != "HOLD", "REGRESSION: the in-domain marker is back"
        assert rec_arg != "", "the empty string must still never reach the tracker"
        assert not is_directional(rec_arg), (
            "the marker must be non-directional, or every unlabelled outcome starts "
            "scoring as a real call"
        )


def test_phase_35_1_field_default_off():
    """Sanity: the paper_learn_loop_enabled Field defaults to False (per
    /goal integration gate 3 -- new operator-visible behavior is gated
    behind a default-OFF flag)."""
    from backend.config.settings import Settings
    s = Settings()
    assert s.paper_learn_loop_enabled is False
