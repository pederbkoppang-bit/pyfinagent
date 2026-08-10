"""phase-86.25 -- the outcome-row vocabulary boundary.

THE DEFECT, IN TWO SEAMS. Both handed a value from a NON-recommendation
vocabulary to a parameter typed for an analyst recommendation:

  S1  autonomous_loop._learn_from_closed_trades  -- `risk_judge_decision`
      (APPROVE_REDUCED / REJECT / APPROVE_HEDGED), coerced to the literal
      "HOLD" when empty. Behind `paper_learn_loop_enabled` (default False).
  S2  nightly_outcome_rebuild._compute_outcomes  -- `risk_judge_decision or
      action`, so an empty approval fell through to the trade ACTION and a
      literal "SELL" was persisted. NOT flag-gated; cron 04:00 UTC.

THE TWO SEAMS FAIL IN OPPOSITE DIRECTIONS, which is why both are tested here
rather than one standing in for the other:

  * S1's "HOLD" is NOT directional, so `directionally_correct` is False even
    when the realised return proves the call was right -- a FALSE NEGATIVE.
  * S2's "SELL" IS directional, so a losing long scores
    `directionally_correct=True` -- a FALSE POSITIVE that credits the system
    for a call nobody made. This is the worse of the two and it is the one
    that ran ungated in production.

MEASURED 2026-08-10 against the same table these read (re-derivable with
`scripts/qa/measure_86_25_join_hitrate.py`): `risk_judge_decision` is
'' 46 / APPROVE_REDUCED 15 / REJECT 3 / APPROVE_HEDGED 1, and **32 of 32 SELL
rows are empty** -- SELL rows being the only ones the learn loop reads.
"""
from __future__ import annotations

import pytest

from backend.services.recommendation_vocab import (
    UNKNOWN_RECOMMENDATION,
    is_buy_intent,
    is_directional,
    is_sell_intent,
    resolve_outcome_recommendation,
)
from backend.slack_bot.jobs.nightly_outcome_rebuild import _compute_outcomes

#: EVERY non-empty spelling measured in `paper_trades.risk_judge_decision`,
#: plus the empty forms. Not a hand-picked sample: this is the full observed
#: value set, and the point of the test is that NONE of them may become a
#: direction.
MEASURED_RISK_JUDGE_VALUES = [
    "APPROVE_REDUCED", "APPROVE_HEDGED", "REJECT", "", "   ", None,
]


class TestBoundaryNeverInventsADirection:
    """Criterion 4: APPROVE_* is NOT mapped onto a buy or sell intent."""

    @pytest.mark.parametrize("value", MEASURED_RISK_JUDGE_VALUES)
    def test_no_risk_judge_value_becomes_a_direction(self, value):
        resolved = resolve_outcome_recommendation(value)
        assert resolved == UNKNOWN_RECOMMENDATION, (
            f"{value!r} resolved to {resolved!r}; a risk-approval token is not a "
            "directional call and must never be read as one"
        )
        assert not is_buy_intent(resolved)
        assert not is_sell_intent(resolved)
        assert not is_directional(resolved)

    def test_the_unknown_marker_is_outside_the_scale(self):
        """The marker must be provably not a member of the decision alphabet.

        If UNKNOWN were ever directional, every unlabelled outcome would start
        scoring as a real call -- which is the failure this whole step removes.
        """
        assert not is_directional(UNKNOWN_RECOMMENDATION)
        assert not is_buy_intent(UNKNOWN_RECOMMENDATION)
        assert not is_sell_intent(UNKNOWN_RECOMMENDATION)

    def test_the_trade_action_is_not_a_recommendation(self):
        """S2's exact defect: the ACTION must not survive as a direction.

        `resolve_outcome_recommendation("SELL")` DOES return SELL -- correctly,
        because a genuine analyst SELL is a real recommendation. The fix is that
        the call site no longer hands it the action. That is asserted
        behaviourally in TestS2, not here; this test pins the distinction so a
        future reader does not "fix" the resolver instead of the call site.
        """
        assert is_directional(resolve_outcome_recommendation("SELL"))
        assert resolve_outcome_recommendation(None) == UNKNOWN_RECOMMENDATION

    def test_a_real_recommendation_still_passes_through(self):
        """Including HOLD: a considered hold is a real call, not a marker."""
        assert resolve_outcome_recommendation("Strong Buy") == "STRONG_BUY"
        assert resolve_outcome_recommendation("Hold") == "HOLD"
        assert is_directional(resolve_outcome_recommendation("Strong Buy"))
        assert not is_directional(resolve_outcome_recommendation("Hold"))
        # ...and HOLD must stay distinguishable from UNKNOWN, or the fix has
        # merely renamed the old coercion.
        assert resolve_outcome_recommendation("Hold") != UNKNOWN_RECOMMENDATION


class TestS2NightlyRebuildCallSite:
    """S2 driven through the REAL production function, not a re-implementation.

    `_compute_outcomes` is the actual row builder the 04:00 UTC cron calls.
    """

    @staticmethod
    def _sell_trade(**over):
        """A SELL trade shaped like the measured population: empty
        risk_judge_decision, action='SELL', a realised loss."""
        t = {
            "trade_id": "t-1", "ticker": "TER", "pnl": -12.5,
            "analysis_id": "", "created_at": "2026-05-14T18:02:54.506004+00:00",
            "risk_judge_decision": "", "action": "SELL",
            "price": 90.0, "holding_days": 21,
        }
        t.update(over)
        return t

    def test_the_action_no_longer_leaks_into_recommendation(self):
        rows = _compute_outcomes([self._sell_trade()])
        assert len(rows) == 1, "the close must NOT be silently dropped"
        rec = rows[0]["recommendation"]
        assert rec == UNKNOWN_RECOMMENDATION, (
            f"recommendation={rec!r}; the pre-86.25 code produced 'SELL' here by "
            "falling through `risk_judge_decision or action`"
        )
        assert rec != "SELL", (
            "REGRESSION: the trade ACTION is being persisted as an analyst "
            "recommendation again -- this is the exact row that reached production"
        )
        assert not is_directional(rec), (
            "a fabricated direction would score a losing long as directionally "
            "correct, crediting a call nobody made"
        )

    def test_an_approval_token_no_longer_leaks_either(self):
        rows = _compute_outcomes([self._sell_trade(risk_judge_decision="APPROVE_REDUCED")])
        assert rows[0]["recommendation"] == UNKNOWN_RECOMMENDATION
        assert not is_directional(rows[0]["recommendation"])

    def test_row_count_is_unchanged_so_nothing_is_silently_dropped(self):
        """Option (B) -- skip the scoring -- was explicitly rejected.

        UNKNOWN is truthy, so the `if not recommendation: continue` skip must
        still not fire. A fix that quietly stopped writing rows would hide the
        close entirely, which is the invisibility that let this run for weeks.
        """
        trades = [self._sell_trade(trade_id=f"t-{i}") for i in range(5)]
        assert len(_compute_outcomes(trades)) == 5

    def test_a_real_recommendation_is_preferred_when_present(self):
        """The (A) shape, honestly labelled.

        MEASURED: this branch runs for 0 of 32 real rows today, because no SELL
        row carries a reachable analyst recommendation. It is tested so the
        behaviour is pinned if the anchor ever lands -- NOT presented as
        coverage of the live population.
        """
        rows = _compute_outcomes([self._sell_trade(analyst_recommendation="Strong Buy")])
        assert rows[0]["recommendation"] == "STRONG_BUY"


class TestReproduceTheScoringDefect:
    """Criterion 1: the defect, executed through the REAL scorer.

    `evaluate_recommendation` is called for real; only its price source is
    stubbed, so the scoring expression under test is production code.
    """

    #: A NAIVE ISO timestamp, and the reason is a defect this step UNCOVERED
    #: but does not own. `evaluate_recommendation` does
    #: `datetime.fromisoformat(analysis_date)` and subtracts it from a NAIVE
    #: `datetime.now(...).replace(tzinfo=None)`. Its comment claims "rec_date
    #: from fromisoformat is naive" -- MEASURED FALSE: every real anchor is
    #: tz-AWARE ('...+00:00'), so the subtraction raises TypeError on 32 of 32
    #: SELL rows, swallowed by the enclosing try in _learn_from_closed_trades.
    #: Using an aware value here would test that crash instead of the scoring
    #: expression criterion 1 is about. Queued as its own step; see
    #: experiment_results_86.25.md "Discovered defects".
    NAIVE_ANCHOR = "2026-05-14T18:02:54.506004"

    class _BQRecorder:
        """Neutralises the ONE write chokepoint and records what WOULD be
        persisted. Nothing leaves the process, and the assertion is about the
        persisted column rather than the in-memory dict -- which matters,
        because `directionally_correct` is computed but NEVER persisted
        (save_outcome writes 9 columns and that is not one of them)."""

        def __init__(self):
            self.saved = []

        def save_outcome(self, **kw):
            self.saved.append(kw)

    @classmethod
    def _tracker(cls, monkeypatch, current_price: float):
        from backend.services import outcome_tracker as ot

        monkeypatch.setattr(
            ot, "get_comprehensive_financials",
            lambda ticker: {"valuation": {"Current Price": current_price}},
        )
        tracker = ot.OutcomeTracker.__new__(ot.OutcomeTracker)
        tracker.bq = cls._BQRecorder()
        tracker._model = None
        return tracker

    def test_S1_shape_false_negative_a_right_call_scored_wrong(self, monkeypatch):
        """The price FELL, so selling was right -- and the old 'HOLD' label
        makes `directionally_correct` False anyway."""
        tracker = self._tracker(monkeypatch, current_price=80.0)
        out = tracker.evaluate_recommendation(
            "TER", self.NAIVE_ANCHOR, "HOLD", 100.0)
        assert out["return_pct"] < 0, "precondition: the trade lost, so a SELL was right"
        assert out["directionally_correct"] is False, (
            "this IS the defect: a correct sell scored as not directionally correct "
            "because the label was a fabricated HOLD"
        )

    def test_S2_shape_false_positive_an_unmade_call_credited(self, monkeypatch):
        """The worse direction, and the one that ran ungated: the fabricated
        'SELL' label scores the same trade as a CORRECT call."""
        tracker = self._tracker(monkeypatch, current_price=80.0)
        out = tracker.evaluate_recommendation(
            "TER", self.NAIVE_ANCHOR, "SELL", 100.0)
        assert out["directionally_correct"] is True, (
            "pre-86.25 behaviour: an ACTION persisted as a recommendation credits "
            "the system for a call no analyst made"
        )

    def test_after_the_fix_the_label_is_honest_rather_than_either(self, monkeypatch):
        """UNKNOWN scores False -- but for the honest reason (no direction was
        known), and it is distinguishable from a considered HOLD in the
        persisted column, which is what criterion 5 asks for."""
        tracker = self._tracker(monkeypatch, current_price=80.0)
        out = tracker.evaluate_recommendation(
            "TER", self.NAIVE_ANCHOR, UNKNOWN_RECOMMENDATION, 100.0)
        assert out["directionally_correct"] is False
        assert out["recommendation"] == UNKNOWN_RECOMMENDATION
        # Criterion 5 asks about what is PERSISTED. `directionally_correct` is
        # not persisted at all, so the question lands on `recommendation`, and
        # the assertion is made against the write chokepoint rather than the
        # returned dict.
        persisted = tracker.bq.saved
        assert len(persisted) == 1, "the outcome must still be written, not dropped"
        assert persisted[0]["recommendation"] == UNKNOWN_RECOMMENDATION
        assert persisted[0]["recommendation"] != "HOLD", (
            "criterion 5: 'direction unknown' must be distinguishable from a real "
            "hold in what is persisted"
        )
        assert "directionally_correct" not in persisted[0], (
            "F2: directionally_correct is NOT among the persisted columns -- if "
            "that ever changes, criterion 5's answer has to be revisited"
        )
