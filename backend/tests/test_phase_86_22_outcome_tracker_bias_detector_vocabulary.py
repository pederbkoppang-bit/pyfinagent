"""phase-86.22 -- the recommendation-vocabulary split is CROSS-MODULE.

phase-86.20 fixed ONE consumer (`portfolio_manager`'s trade gate). Five others
read the same `financial_reports.analysis_results.recommendation` column in TWO
MUTUALLY INCOMPATIBLE dialects, and they break in OPPOSITE directions:

  UPPER_SNAKE consumers  -- `.upper()` folds case but not the separator, so the
                            producer's own "Strong Buy" -> "STRONG BUY" matches
                            nothing. (bias_detector, api/portfolio,
                            conflict_detector)
  TITLE-CASE consumers   -- exact, case-SENSITIVE match against
                            ("Strong Buy","Buy") with NO folding at all, so the
                            91 rows spelled "BUY" match NOTHING.
                            (outcome_tracker, memory)

THE SECOND ONE IS WORSE, AND NOT FOR THE OBVIOUS REASON. A dropped candidate is
an absence. A wrong `directionally_correct` is a LABEL: measured over the same
table `outcome_tracker` reads, all 91 literal `BUY` rows were scored
`directionally_correct=False` regardless of the actual return -- roughly two
thirds of every buy-intent row -- and `memory.py` renders that into a reflection
persisted to `agent_memories`, where free-text prose is beyond any schema check.

MEASURED AT FIX TIME: `agent_memories` rows=0, and `directionally_correct` is
never persisted at all (`save_outcome` does not carry it; the table has no such
column). So there is NOTHING TO BACKFILL -- this lands before the writer's first
row, which is the only cheap moment it will ever have.

NOT CLAIMED: no lost trade and no lost P&L. These are evaluation, analytics and
bias-detection paths; none places an order.

Contract: handoff/current/contract_86.22.md
Research: handoff/current/research_brief_86.22.md
"""
from __future__ import annotations

import pytest

from backend.services.recommendation_vocab import (
    is_buy_intent,
    is_sell_intent,
    is_directional,
)

# The two dialects, as they actually occur in the column.
PRODUCER_SPACED = "Strong Buy"      # what the full pipeline emits
UPPER_LITERAL = "BUY"               # 91 rows, every one genuine
TITLE_LITERAL = "Buy"               # worked in the title-case consumers only


# ── criterion 1: REPRODUCE, in BOTH directions ─────────────────────────────


def _legacy_title_case(rec) -> tuple[bool, bool]:
    """The pre-86.22 expression in outcome_tracker.py / memory.py, verbatim."""
    return (rec in ("Strong Buy", "Buy"), rec in ("Strong Sell", "Sell"))


def _legacy_upper_snake(rec) -> bool:
    """The pre-86.22 expression in bias_detector / api.portfolio, verbatim."""
    return (rec or "").upper() in ("STRONG_BUY", "BUY")


def test_REPRODUCE_title_case_consumer_misses_the_upper_literal():
    """The learn-loop direction: 'BUY' matches NEITHER leg, so a correct call
    is labelled incorrect."""
    is_buy, is_sell = _legacy_title_case(UPPER_LITERAL)
    assert not is_buy and not is_sell, "defect not reproduced: 'BUY' matched a leg"

    # ...and that is what made the label wrong, not merely absent:
    return_pct = +12.0                       # the call was RIGHT
    legacy_directionally_correct = (is_buy and return_pct > 0) or (is_sell and return_pct < 0)
    assert legacy_directionally_correct is False, "the wrong label is the defect"

    # control: the spelling the legacy expression WAS written for still worked
    assert _legacy_title_case(TITLE_LITERAL)[0] is True


def test_REPRODUCE_upper_snake_consumer_misses_the_producer_spelling():
    """The 86.20 direction, in the OTHER modules: 'Strong Buy' matches nothing."""
    assert _legacy_upper_snake(PRODUCER_SPACED) is False, "defect not reproduced"
    assert _legacy_upper_snake("STRONG_BUY") is True, "control failed"


# ── criterion 5 + the fix: one vocabulary, both dialects, no widening ───────


@pytest.mark.parametrize(
    "spelling",
    ["Strong Buy", "strong buy", "STRONG-BUY", "Strong_Buy", "STRONG_BUY",
     "Buy", "BUY", "buy"],
)
def test_every_buy_spelling_is_a_buy_intent(spelling):
    assert is_buy_intent(spelling) is True
    assert is_sell_intent(spelling) is False
    assert is_directional(spelling) is True


@pytest.mark.parametrize(
    "spelling", ["Strong Sell", "STRONG_SELL", "strong-sell", "Sell", "SELL"]
)
def test_every_sell_spelling_is_a_sell_intent(spelling):
    assert is_sell_intent(spelling) is True
    assert is_buy_intent(spelling) is False
    assert is_directional(spelling) is True


@pytest.mark.parametrize(
    "value",
    ["HOLD", "Hold", "hold", "N/A", "Accumulate", "Overweight", "Outperform",
     "BUYING", "NOT A BUY", "STRONG", "Strong Buy!", "", "   ", None, 123,
     {"a": 1}, ["BUY"]],
)
def test_nothing_else_becomes_a_direction(value):
    """No widening. 'BUYING' and 'NOT A BUY' are here because a substring
    matcher admits both -- that is conflict_detector's historical defect."""
    assert is_buy_intent(value) is False
    assert is_sell_intent(value) is False
    assert is_directional(value) is False


def test_HOLD_is_recognised_but_NOT_directional():
    """A considered HOLD and an unparseable value must not collapse together --
    that conflation is what let a wrong label look like an ordinary False."""
    from backend.services.recommendation_vocab import canonical_recommendation

    assert canonical_recommendation("Hold") == "HOLD"      # recognised
    assert is_directional("Hold") is False                 # but not directional
    assert canonical_recommendation("N/A") is None         # NOT recognised
    assert is_directional("N/A") is False


# ── the migrated consumers, end to end ─────────────────────────────────────


# REMOVED in cycle 2: `test_outcome_tracker_labels_a_correct_buy_call_CORRECT`.
# It called itself "the load-bearing behavioural assertion" and was neither. It
# recomputed `directionally_correct` from `is_buy_intent` IN THE TEST BODY and
# then asserted `t is not None and Settings is not None` -- a tautology plus an
# import check, dressed as behaviour. It could not have failed if
# outcome_tracker had never been migrated at all, which the cycle-1 Q/A proved
# by reverting the file and watching the suite stay green.
#
# Its replacement is `test_outcome_tracker_evaluate_recommendation_IS_DRIVEN_
# with_literal_BUY` below, which calls the real function.


def _bias_report(rec: str) -> dict:
    """Drive the REAL public entry point with the REAL signature."""
    from backend.agents.bias_detector import detect_biases

    return detect_biases(
        ticker="AAPL",                       # in TECH_TICKERS
        recommendation=rec,
        score=9.0,                           # >= 7.5, the tech-bias threshold
        enrichment_signals={"a": {"signal": "BULLISH"}, "b": {"signal": "BULLISH"}},
        debate_result={},
        quant_data={"yf_data": {"valuation": {"Market Cap": 3_000_000_000_000}}},
    )


@pytest.mark.parametrize("rec", ["Strong Buy", "STRONG_BUY", "strong buy", "BUY"])
def test_bias_detector_fires_on_every_strong_buy_spelling(rec):
    """A bias check that never fires on the highest-conviction spelling is a
    bias check that is off exactly when it matters most."""
    kinds = {f["bias_type"] for f in _bias_report(rec)["flags"]}
    assert "tech_bias" in kinds, f"{rec!r} did not trip the tech-bias check"


def test_bias_detector_does_NOT_fire_on_a_hold_or_an_unparseable_value():
    """Precision: the fix must not turn every recommendation into a buy."""
    for rec in ("Hold", "N/A", "Accumulate", ""):
        kinds = {f["bias_type"] for f in _bias_report(rec)["flags"]}
        assert "tech_bias" not in kinds, f"{rec!r} wrongly counted as a buy"


def test_conflict_detector_grades_a_strong_buy_at_the_STRICTER_threshold():
    """The worst true positive. Before: a missed 'STRONG BUY' fell through to
    the substring `elif "BUY"` and was graded at 5.5 instead of 7.0 -- the
    strictest check silently became the loosest."""
    from backend.agents.conflict_detector import _check_recommendation_alignment

    report = {"recommendation": {"recommendation": "Strong Buy"},
              "final_weighted_score": 6.0}          # < 7.0 but > 5.5
    conflicts = _check_recommendation_alignment(report)
    assert conflicts, "a Strong Buy at 6.0 must conflict (threshold 7.0)"
    assert "STRONG_BUY" in conflicts[0].llm_belief


def test_conflict_detector_does_not_grade_a_SELL_as_a_BUY():
    """`"STRONG_SELL"` contains `"SELL"`, and a substring chain conflates them."""
    from backend.agents.conflict_detector import _check_recommendation_alignment

    report = {"recommendation": {"recommendation": "Strong Sell"},
              "final_weighted_score": 9.0}
    conflicts = _check_recommendation_alignment(report)
    for c in conflicts:
        assert "BUY" not in c.llm_belief, "a SELL was graded as a BUY"


# ── criterion 4: no SECOND vocabulary may survive ──────────────────────────


def _derivation():
    """Load the ONE derivation. Deliberately not a second implementation: two
    checkers that can disagree about what counts as a consumer would be this
    step's own defect in miniature."""
    import importlib.util
    import pathlib

    p = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "qa" / "derive_recommendation_consumers_86_22.py")
    spec = importlib.util.spec_from_file_location("derive_86_22", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_second_recommendation_vocabulary_survives_in_backend():
    """Criterion 4, as a TEST rather than a claim."""
    d = _derivation()
    rows = d.scan_tree(lambda rel: (d.REPO_ROOT / rel).read_text(
        encoding="utf-8", errors="ignore"))
    offenders = [r for r in rows if not r.get("allowed")]
    assert not offenders, "a second recommendation vocabulary survives:\n  " + "\n  ".join(
        f"{r['file']}:{r['line']} ({r.get('func')})  {r['tested']}  {r['members']}"
        for r in offenders
    )


def test_the_derivation_has_measured_recall_AND_precision():
    """The guard above is worth exactly what its method is worth. Re-run the
    method's own validation here so a regression in the DETECTOR fails the
    suite, not just the operator's manual run.

    Both directions are asserted. A detector that flags everything has perfect
    recall and is useless; one that flags nothing passes any all-clear check.
    """
    d = _derivation()

    missed = [name for name, src in d.KNOWN_POSITIVES if not d.scan_source(src, "<f>")]
    assert not missed, f"derivation MISSES known defects: {missed}"

    false_pos = [name for name, src in d.KNOWN_NEGATIVES if d.scan_source(src, "<f>")]
    assert not false_pos, f"derivation FLAGS separate vocabularies: {false_pos}"

    # And the positives must include the substring shape -- the one R1+R2
    # missed entirely, which is why conflict_detector went unfound.
    r3 = [n for n, s in d.KNOWN_POSITIVES
          if any("R3" in row["rule"] for row in d.scan_source(s, "<f>"))]
    assert len(r3) >= 3, f"the substring shape is not covered: {r3}"


# ── cycle 2: DRIVE THE CONSUMERS, not a copy of their logic ────────────────
#
# The cycle-1 Q/A returned FAIL on exactly this, and it was right. Every
# assertion above this line tests `is_buy_intent` -- the shared vocabulary --
# and NOTHING tested that a consumer actually calls it. The Q/A proved the gap
# by reverting each migrated file to its pre-fix source: four of six left the
# suite fully GREEN, including BOTH learning-path consumers.
#
# `test_outcome_tracker_labels_a_correct_buy_call_CORRECT` was the worst of it.
# Its docstring called it "the load-bearing behavioural assertion" while it
# recomputed `directionally_correct` IN THE TEST BODY and asserted only that
# the imports resolved. A test that re-implements the logic it is checking
# passes whatever the subject does.
#
# The tests below call the REAL functions. Each is paired with a per-site cell
# in scripts/qa/mutation_matrix_86_22.py that reverts that consumer to its
# pre-fix source; the mutant dying is what proves the test drives the consumer.


def test_outcome_tracker_evaluate_recommendation_IS_DRIVEN_with_literal_BUY(monkeypatch):
    """C1: drive the real function with the 91-row 'BUY' spelling.

    Pre-fix this returned directionally_correct=False for a call that gained
    12%, because 'BUY' matched neither ("Strong Buy","Buy") nor
    ("Strong Sell","Sell"). The corresponding mutation cell reverts this file
    and this assertion is what kills it.
    """
    import backend.services.outcome_tracker as ot

    monkeypatch.setattr(ot, "get_comprehensive_financials",
                        lambda ticker: {"valuation": {"Current Price": 112.0}})
    tracker = ot.OutcomeTracker.__new__(ot.OutcomeTracker)
    saved = {}
    tracker.bq = type("BQ", (), {"save_outcome": lambda self, **kw: saved.update(kw)})()

    outcome = tracker.evaluate_recommendation(
        ticker="AAPL", analysis_date="2026-07-01T00:00:00",
        recommendation="BUY", price_at_rec=100.0,
    )

    assert outcome is not None, "evaluate_recommendation returned None -- not driven"
    assert outcome["return_pct"] > 0, "fixture must represent a WINNING call"
    assert outcome["directionally_correct"] is True, (
        "a BUY that gained 12% was scored directionally INCORRECT -- this is the "
        "defect, measured over 91 rows"
    )
    assert saved, "save_outcome was not called -- the persist path was not driven"


@pytest.mark.parametrize(
    "rec,ret,expected",
    [("BUY", 12.0, True), ("Buy", 12.0, True), ("Strong Buy", 12.0, True),
     ("STRONG_BUY", 12.0, True), ("BUY", -8.0, False), ("SELL", -8.0, True),
     ("Sell", -8.0, True), ("HOLD", 12.0, False), ("N/A", 12.0, False)],
)
def test_outcome_tracker_label_is_decided_by_the_RETURN_in_every_spelling(
    rec, ret, expected, monkeypatch
):
    """The label must follow the realised return, not the spelling.

    Includes the negatives C5 asks for per consumer: HOLD and an unparseable
    value must stay non-directional even when the return is positive.
    """
    import backend.services.outcome_tracker as ot

    price = 100.0 * (1.0 + ret / 100.0)
    monkeypatch.setattr(ot, "get_comprehensive_financials",
                        lambda ticker: {"valuation": {"Current Price": price}})
    tracker = ot.OutcomeTracker.__new__(ot.OutcomeTracker)
    tracker.bq = type("BQ", (), {"save_outcome": lambda self, **kw: None})()

    outcome = tracker.evaluate_recommendation("AAPL", "2026-07-01T00:00:00", rec, 100.0)
    assert outcome["directionally_correct"] is expected, (
        f"{rec!r} with return {ret}% scored {outcome['directionally_correct']}, "
        f"expected {expected}"
    )


def test_memory_generate_reflection_IS_DRIVEN_and_the_PROMPT_carries_the_label():
    """The durable half: the label is rendered into a prompt that becomes a
    lesson persisted to agent_memories, where no schema check can catch it.

    `model` is a PARAMETER of generate_reflection, so the real function is
    driven directly -- no monkeypatching, and no LLM call."""
    import backend.agents.memory as mem

    captured = {}

    class _Model:
        def generate_content(self, prompt, generation_config=None):
            captured["prompt"] = prompt
            return type("R", (), {"text": "lesson"})()

    mem.generate_reflection(
        model=_Model(),
        agent_type="quant", ticker="AAPL", original_recommendation="BUY",
        actual_return_pct=12.0, situation="ctx", holding_days=30,
    )
    assert "Directionally correct: YES" in captured["prompt"], (
        "a winning BUY was written into the reflection prompt as NOT "
        f"directionally correct; prompt said: "
        f"{[l for l in captured['prompt'].splitlines() if 'Directionally' in l]}"
    )


def test_memory_reflection_FALLBACK_text_also_carries_the_right_label():
    """The LLM path can fail; the fallback string is what then reaches memory."""
    import backend.agents.memory as mem

    class _Boom:
        def generate_content(self, prompt, generation_config=None):
            raise RuntimeError("LLM down")

    text = mem.generate_reflection(
        model=_Boom(),
        agent_type="quant", ticker="AAPL", original_recommendation="BUY",
        actual_return_pct=12.0, situation="ctx", holding_days=30,
    )
    assert text.startswith("Correct call on AAPL"), (
        f"fallback lesson mislabels a winning BUY: {text!r}"
    )


def test_memory_reflection_does_NOT_call_a_HOLD_directional():
    """C5 negative for this consumer."""
    import backend.agents.memory as mem

    class _Boom:
        def generate_content(self, prompt, generation_config=None):
            raise RuntimeError("LLM down")

    for rec in ("HOLD", "N/A", ""):
        text = mem.generate_reflection(
            model=_Boom(),
            agent_type="quant", ticker="AAPL", original_recommendation=rec,
            actual_return_pct=12.0, situation="ctx", holding_days=30,
        )
        assert text.startswith("Incorrect call"), (
            f"{rec!r} was treated as a directional call: {text!r}"
        )


def test_api_portfolio_accuracy_DENOMINATOR_includes_every_buy_spelling(monkeypatch):
    """Pre-fix, a 'Strong Buy' position was excluded from the accuracy
    DENOMINATOR entirely -- an analytics lie, not just a rounding difference."""
    import asyncio

    import backend.api.portfolio as pf

    # The 'Strong Buy' is deliberately a LOSING position. If it is excluded
    # from the denominator -- the pre-fix behaviour -- accuracy reads 1/1 =
    # 100%. Included, it reads 1/2 = 50%. Had this fixture made it a winner,
    # both readings would be 100% and the test could not tell the fix from the
    # defect: an excluded row only shows up when it would have been counted
    # AGAINST the score.
    positions = {"p1": {}, "p2": {}, "p3": {}}
    enriched = {
        "p1": {"ticker": "AAA", "cost_basis": 100, "market_value": 90,
               "unrealized_pnl": -10, "unrealized_pnl_pct": -10,
               "recommendation": "Strong Buy"},        # a WRONG high-conviction call
        "p2": {"ticker": "BBB", "cost_basis": 100, "market_value": 110,
               "unrealized_pnl": 10, "unrealized_pnl_pct": 10,
               "recommendation": "BUY"},               # a right one
        "p3": {"ticker": "CCC", "cost_basis": 100, "market_value": 90,
               "unrealized_pnl": -10, "unrealized_pnl_pct": -10,
               "recommendation": "HOLD"},              # not directional at all
    }

    async def _fake_enrich(pos_id, pos):
        return enriched[pos_id]

    monkeypatch.setattr(pf, "_positions", positions)
    monkeypatch.setattr(pf, "_enrich_position_async", _fake_enrich)

    result = asyncio.run(pf.get_portfolio_performance())

    # 2 directional positions (Strong Buy, BUY); 1 of them correct; HOLD excluded.
    assert result["recommendation_accuracy"] == 50.0, (
        f"accuracy {result['recommendation_accuracy']} -- 100.0 means the losing "
        "'Strong Buy' was dropped from the denominator (the pre-fix analytics "
        "lie); 33.3 means HOLD wrongly entered it"
    )


def test_slack_formatter_rec_color_handles_BOTH_dialects():
    """The display consumer. Its own hand-rolled canonicaliser spelled out both
    'STRONG_BUY' and 'STRONG BUY'; the shared one must not regress that."""
    from backend.slack_bot.formatters import _rec_color

    assert _rec_color("Strong Buy") == _rec_color("STRONG_BUY") == "#22c55e"
    assert _rec_color("Buy") == _rec_color("BUY") == "#4ade80"
    assert _rec_color("Sell") == _rec_color("STRONG_SELL") == "#ef4444"
    # C5 negatives for this consumer: neutral colour, never a buy colour.
    for junk in ("Hold", "N/A", "BUYOUT", "", None):
        assert _rec_color(junk) == "#f59e0b", f"{junk!r} rendered as a direction"


def test_skill_optimizer_consensus_uses_the_shared_vocabulary(monkeypatch):
    """`debate_consensus` comes from the SAME persisted table as
    `recommendation`, so it is exposed to the same dialect drift.

    Measured at fix time the column held only '' / NULL / 'HOLD' / 'BUY', so
    the pre-fix expression was correct in EFFECT -- this test pins the
    behaviour for the spelling that is not there YET, which is the whole point
    of a shared vocabulary.
    """
    from backend.services.recommendation_vocab import is_buy_intent, is_sell_intent

    # The scoring branches key off exactly these two predicates.
    assert is_buy_intent("Strong Buy") is True      # would have been missed
    assert is_buy_intent("BUY") is True             # the one value present today
    assert is_sell_intent("Strong Sell") is True
    for neutral in ("HOLD", "", None):
        assert is_buy_intent(neutral) is False
        assert is_sell_intent(neutral) is False

    # ...and the module must actually import them, not re-derive its own.
    import backend.agents.skill_optimizer as so

    assert so.is_buy_intent is is_buy_intent, (
        "skill_optimizer does not use the shared vocabulary"
    )
    assert so.is_sell_intent is is_sell_intent
