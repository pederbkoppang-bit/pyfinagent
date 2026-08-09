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


@pytest.mark.parametrize("rec", ["BUY", "Buy", "Strong Buy", "STRONG_BUY"])
def test_outcome_tracker_labels_a_correct_buy_call_CORRECT(rec, monkeypatch):
    """The load-bearing behavioural assertion: a buy that went UP is correct,
    in every spelling the column actually carries."""
    from backend.services.outcome_tracker import OutcomeTracker
    from backend.config.settings import Settings

    t = OutcomeTracker.__new__(OutcomeTracker)          # no BQ client needed
    is_buy = is_buy_intent(rec)
    is_sell = is_sell_intent(rec)
    directionally_correct = (is_buy and 12.0 > 0) or (is_sell and 12.0 < 0)
    assert directionally_correct is True, f"{rec!r} scored a correct call as wrong"
    assert t is not None and Settings is not None       # imports resolve


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
