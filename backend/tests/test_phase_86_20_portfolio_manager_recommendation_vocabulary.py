"""phase-86.20 -- the trade gate and the analyzer speak different recommendation vocabularies.

`portfolio_manager` tests membership against UNDERSCORE sets (`_BUY_RECS =
{"BUY","STRONG_BUY"}`) after `.upper()` only. `.upper()` folds CASE but never the
SEPARATOR, and the full-pipeline producer emits SPACED TITLE CASE, so:

    "Strong Buy".upper()  == "STRONG BUY"   not in _BUY_RECS        -> dropped
    "Strong Sell".upper() == "STRONG SELL"  not in _SELL_RECS
                                            not in _DOWNGRADE_RECS  -> NOT SOLD

Plain "Buy" works by accident, which is why this survived: the mismatch destroys
the HIGHEST-conviction spelling while letting the medium one through.

THE TWO HALVES HAVE OPPOSITE RISK POLARITY, and that is the whole design tension:
  * buy side  -- a dropped candidate costs OPPORTUNITY (fail-safe direction)
  * sell side -- a dropped exit costs PROTECTION (fail-DANGEROUS): only the
                 stop-loss can still close that position.

FIXTURES MUST USE THE PRODUCTION SPELLING. The existing suite is green against a
dialect production never emits -- `test_dod4_tier1_coverage_investment.py` feeds
`"STRONG_BUY"` (underscore), which the full path does not produce. A test written
in the underscore dialect is blind to this defect by construction. Every fixture
below that represents producer output uses the SPACED form.

Contract: handoff/current/contract_86.20.md
Research: handoff/current/research_brief_86.20.md

ISOLATION (enumerated per channel, per the phase-36.17 lesson):
  * filesystem -- `risk_overrides` reads and appends
    `handoff/risk_overrides_audit.jsonl` at module scope; `decide_trades` calls
    `risk_overrides.get_effective` four times. Redirected to tmp_path, with a
    byte-comparison guard on the live file.
  * network / BQ / LLM -- none: `decide_trades` is a pure function over dicts.
    Asserted by construction (no client is constructed anywhere below).
  * module singletons -- `Settings()` is constructed per test, never mutated
    globally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The SPACED title-case vocabulary the full-pipeline producer actually emits
# (backend/agents/schemas.py: action is a free-text str whose vocabulary lives
# only in the prompt description).
PRODUCER_STRONG_BUY = "Strong Buy"
PRODUCER_STRONG_SELL = "Strong Sell"
# The UNDERSCORE vocabulary the gate tests against.
GATE_BUY = "BUY"
GATE_SELL = "SELL"


# ── isolation ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _risk_overrides_isolated(tmp_path, monkeypatch):
    """Point the risk-override audit trail at tmp_path.

    `decide_trades` calls `risk_overrides.get_effective` for
    paper_min_cash_reserve_pct, paper_max_per_sector and
    paper_max_per_sector_nav_pct. That module holds a module-scope
    `_AUDIT_PATH` under the live handoff/ tree which it reads on load and
    appends to on set. Reading is harmless today, but pinning a test to a live
    path is exactly how the 36.17 cycle-lock leak happened.
    """
    import backend.services.risk_overrides as ro

    monkeypatch.setattr(ro, "_AUDIT_PATH", tmp_path / "risk_overrides_audit.jsonl")
    yield


@pytest.fixture(autouse=True)
def _live_risk_override_audit_is_write_protected():
    live = REPO_ROOT / "handoff" / "risk_overrides_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        f"phase-86.20: a test in this module wrote to the LIVE {live}. "
        "Redirect risk_overrides._AUDIT_PATH to tmp_path."
    )


# ── harness ─────────────────────────────────────────────────────────────────


def _settings(vocab_fix: bool = False):
    """Settings with the phase-86.20 flag explicitly set.

    Explicit on BOTH sides on purpose: a test that relied on the default would
    silently change meaning the day the flag is promoted.
    """
    from backend.config.settings import Settings

    s = Settings()
    s.paper_recommendation_vocab_fix_enabled = vocab_fix
    return s


def _portfolio_state():
    """A clean book with plenty of cash, so nothing but the vocabulary gates."""
    return {"nav": 100_000.0, "cash": 100_000.0, "positions_value": 0.0, "position_count": 0}


def _candidate(ticker: str, recommendation: str) -> dict:
    """A buy candidate that is identical in every respect EXCEPT the spelling."""
    return {
        "ticker": ticker,
        "recommendation": recommendation,
        "final_score": 8.5,
        "price_at_analysis": 100.0,
        "analysis_date": "2026-08-09",
        "sector": "Technology",
        "risk_assessment": {"recommended_position_pct": 5.0, "stop_loss_price": 92.0},
    }


def _held_position(ticker: str, recommendation: str = "") -> dict:
    return {
        "ticker": ticker,
        "quantity": 100,
        "avg_entry_price": 100.0,
        "current_price": 110.0,      # ABOVE any stop, so the stop-loss branch
        "stop_loss_price": 90.0,     # cannot be what closes it
        "market_value": 11_000.0,
        "sector": "Technology",
        "recommendation": recommendation,
    }


def _holding_analysis(ticker: str, recommendation: str) -> dict:
    return {
        "ticker": ticker,
        "recommendation": recommendation,
        "final_score": 3.0,
        "analysis_date": "2026-08-09",
    }


def _decide(**kw):
    from backend.services.portfolio_manager import decide_trades

    return decide_trades(
        current_positions=kw.get("current_positions", []),
        candidate_analyses=kw.get("candidate_analyses", []),
        holding_analyses=kw.get("holding_analyses", []),
        portfolio_state=kw.get("portfolio_state", _portfolio_state()),
        settings=kw.get("settings") or _settings(),
    )


def _buys(orders):
    return [o for o in orders if o.action == "BUY"]


def _sells(orders):
    return [o for o in orders if o.action == "SELL"]


# ── REPRODUCE (criterion 1) -- these assert the DEFECT and must pass PRE-fix ──


def test_REPRODUCE_producer_strong_buy_yields_no_buy_while_gate_buy_does():
    """Criterion 1: isolate the difference to the VOCABULARY and nothing else.

    Two candidates identical in every field except the recommendation spelling.
    Under current code the producer's own spelling yields no order at all.
    """
    spaced = _decide(candidate_analyses=[_candidate("AAA", PRODUCER_STRONG_BUY)])
    underscored = _decide(candidate_analyses=[_candidate("AAA", GATE_BUY)])

    assert _buys(underscored), (
        "control failed: 'BUY' must produce a buy order, otherwise this test "
        "proves nothing about the vocabulary (phase-86.20)"
    )
    assert not _buys(spaced), (
        "phase-86.20 DEFECT NOT REPRODUCED: 'Strong Buy' produced a buy order. "
        "If this fails the separator gap is already closed."
    )


def test_REPRODUCE_producer_strong_sell_is_not_sold_at_all():
    """Criterion 1, the FAIL-DANGEROUS half (research finding A).

    'Strong Sell' matches neither _SELL_RECS nor _DOWNGRADE_RECS, so a held
    position is not exited by either branch. Priced ABOVE its stop so the
    stop-loss branch cannot be what closes it.
    """
    spaced = _decide(
        current_positions=[_held_position("BBB", recommendation=PRODUCER_STRONG_BUY)],
        holding_analyses=[_holding_analysis("BBB", PRODUCER_STRONG_SELL)],
    )
    underscored = _decide(
        current_positions=[_held_position("BBB", recommendation=GATE_BUY)],
        holding_analyses=[_holding_analysis("BBB", GATE_SELL)],
    )

    assert _sells(underscored), (
        "control failed: 'SELL' must produce a sell order, otherwise this test "
        "proves nothing about the vocabulary (phase-86.20)"
    )
    assert not _sells(spaced), (
        "phase-86.20 DEFECT NOT REPRODUCED: 'Strong Sell' produced a sell order."
    )


# ── the canonicaliser itself (criterion 3 + criterion 5, unit level) ────────


@pytest.mark.parametrize(
    "spelling",
    [
        "Strong Buy",      # what the full-pipeline producer actually emits
        "strong buy",      # case
        "STRONG-BUY",      # hyphen separator
        "Strong_Buy",      # underscore separator
        "Strong  Buy",     # collapsed run of whitespace
        "  Strong Buy  ",  # surrounding whitespace
    ],
)
def test_spacing_and_punctuation_variants_fold_to_one_token(spelling):
    """Criterion 3: variants of ONE intent must reach the SAME decision."""
    from backend.services.recommendation_vocab import canonical_recommendation

    assert canonical_recommendation(spelling) == "STRONG_BUY"


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Buy", "BUY"), ("BUY", "BUY"),           # 'Buy' was safe before; still is
        ("Hold", "HOLD"), ("HOLD", "HOLD"),
        ("Sell", "SELL"), ("SELL", "SELL"),
        ("Strong Sell", "STRONG_SELL"), ("STRONG_SELL", "STRONG_SELL"),
    ],
)
def test_already_working_spellings_are_unchanged(spelling, expected):
    """Criterion 3's second half: do not regress what already worked."""
    from backend.services.recommendation_vocab import canonical_recommendation

    assert canonical_recommendation(spelling) == expected


@pytest.mark.parametrize(
    "value",
    [
        "Accumulate",    # a real broker word that is NOT on our scale
        "Overweight",    # ditto
        "Outperform",    # ditto
        "Strong Buy!",   # trailing punctuation is NOT stripped -- do not guess
        "BUYING",        # substring-adjacent: must NOT match BUY
        "STRONG",        # prefix-adjacent
        "NOT A BUY",     # contains BUY; a substring matcher would say yes
        "N/A", "", "   ", None, 123, {"a": 1}, ["BUY"],
    ],
)
def test_unrecognised_values_are_never_guessed_into_an_intent(value):
    """Criterion 5 at the unit level -- and the anti-widening guard.

    An over-permissive gate is worse than a narrow one because it moves money.
    'NOT A BUY' and 'BUYING' are here specifically because a substring matcher
    would admit both -- that is the independent defect class filed as 86.22.
    """
    from backend.services.recommendation_vocab import canonical_recommendation

    assert canonical_recommendation(value) is None


# ── the gate, with the fix ARMED (criteria 3, 4, 5) ─────────────────────────


@pytest.mark.parametrize(
    "spelling", ["Strong Buy", "strong buy", "STRONG-BUY", "Strong_Buy", "STRONG_BUY"]
)
def test_armed_every_strong_buy_spelling_reaches_the_buy_stage(spelling):
    """Criterion 3 end-to-end: three-plus variants reach the SAME decision."""
    orders = _decide(
        candidate_analyses=[_candidate("AAA", spelling)], settings=_settings(True)
    )
    assert _buys(orders), f"{spelling!r} did not reach the buy stage with the fix armed"


def test_armed_plain_buy_still_reaches_the_buy_stage():
    """'Buy' worked by accident before; it must still work after."""
    orders = _decide(
        candidate_analyses=[_candidate("AAA", "Buy")], settings=_settings(True)
    )
    assert _buys(orders)


@pytest.mark.parametrize("spelling", ["Strong Sell", "strong sell", "STRONG-SELL", "STRONG_SELL"])
def test_armed_strong_sell_exits_the_position(spelling):
    """Criterion 4: the FAIL-DANGEROUS half. Not covering SELL leaves the book
    holding a position its own analyst says to dump."""
    orders = _decide(
        current_positions=[_held_position("BBB", recommendation="Strong Buy")],
        holding_analyses=[_holding_analysis("BBB", spelling)],
        settings=_settings(True),
    )
    sells = _sells(orders)
    assert sells, f"{spelling!r} did not produce a sell with the fix armed"
    assert sells[0].reason == "sell_signal"


def test_armed_downgrade_from_a_spaced_prior_recommendation_exits():
    """Criterion 4, downgrade branch -- and the phase-61.2 interaction.

    The position row carries the PRODUCER spelling ('Strong Buy') because
    paper_trader persists it verbatim. Before this fix `old_rec` resolved to
    'STRONG BUY', which is not in _BUY_RECS, so the signal_downgrade branch
    could never fire for full-path rows -- the exact rule phase-61.2 exists to
    revive. This asserts that interaction directly.
    """
    orders = _decide(
        current_positions=[_held_position("CCC", recommendation="Strong Buy")],
        holding_analyses=[_holding_analysis("CCC", "Hold")],
        settings=_settings(True),
    )
    sells = _sells(orders)
    assert sells, "a Strong Buy -> Hold downgrade did not exit with the fix armed"
    assert sells[0].reason == "signal_downgrade"


def test_armed_the_same_downgrade_does_NOT_fire_with_the_flag_off():
    """The OFF path must stay byte-identical -- this is the control for the
    test above, and it is what makes 'flag-gated' a measured claim."""
    orders = _decide(
        current_positions=[_held_position("CCC", recommendation="Strong Buy")],
        holding_analyses=[_holding_analysis("CCC", "Hold")],
        settings=_settings(False),
    )
    assert not _sells(orders)


@pytest.mark.parametrize("value", ["HOLD", "Hold", "Sell", "N/A", "", None, "Accumulate"])
def test_armed_no_non_buy_string_ever_produces_a_buy(value):
    """Criterion 5: the gate must not have been widened by the fix."""
    orders = _decide(
        candidate_analyses=[_candidate("AAA", value)], settings=_settings(True)
    )
    assert not _buys(orders), f"{value!r} produced a BUY -- the gate was widened"


@pytest.mark.parametrize("value", ["N/A", "Accumulate", "Overweight"])
def test_armed_an_unrecognised_holding_is_not_sold_either(value):
    """UNKNOWN is neither a buy NOR a sell NOR a downgrade. A fix that turned
    an unparseable value into a downgrade would liquidate on a parse error."""
    orders = _decide(
        current_positions=[_held_position("BBB", recommendation="Strong Buy")],
        holding_analyses=[_holding_analysis("BBB", value)],
        settings=_settings(True),
    )
    assert not _sells(orders)


# ── observability (criterion 6) -- UNCONDITIONAL, so it is tested flag-OFF ──


def test_an_unrecognised_recommendation_is_logged_distinctly(caplog):
    """Criterion 6: unrecognised must be distinguishable from a recognised
    non-buy such as HOLD."""
    import logging

    with caplog.at_level(logging.WARNING):
        _decide(candidate_analyses=[_candidate("AAA", "Accumulate")], settings=_settings(False))
    assert any("UNRECOGNISED recommendation" in r.getMessage() for r in caplog.records), \
        "an unrecognised recommendation was dropped silently"


def test_a_recognised_non_buy_is_NOT_logged_as_unrecognised(caplog):
    """The other half of criterion 6: HOLD is a decision, not a drift signal.
    Without this the alarm would fire on every hold and be ignored."""
    import logging

    with caplog.at_level(logging.WARNING):
        _decide(candidate_analyses=[_candidate("AAA", "HOLD")], settings=_settings(False))
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "UNRECOGNISED recommendation" not in joined
    assert "VOCABULARY MISMATCH" not in joined


def test_the_live_defect_is_loud_even_with_the_flag_OFF(caplog):
    """This is the point of making loudness unconditional: with the fix DARK,
    'Strong Buy' is still dropped -- but it is no longer dropped SILENTLY."""
    import logging

    with caplog.at_level(logging.WARNING):
        orders = _decide(
            candidate_analyses=[_candidate("AAA", PRODUCER_STRONG_BUY)],
            settings=_settings(False),
        )
    assert not _buys(orders)          # still dropped: the flag is OFF
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "VOCABULARY MISMATCH" in joined, "the drop is still silent"
    assert "STRONG_BUY" in joined and "STRONG BUY" in joined, \
        "the log must name both the canonical token and what the legacy gate saw"


def test_an_absent_recommendation_on_a_legacy_position_row_is_quiet(caplog):
    """Legacy position rows carry no recommendation at all. That is not a drift
    signal, and an alarm that fires on it would be trained away."""
    import logging

    with caplog.at_level(logging.WARNING):
        _decide(
            current_positions=[_held_position("BBB", recommendation="")],
            holding_analyses=[_holding_analysis("BBB", "HOLD")],
            settings=_settings(False),
        )
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "UNRECOGNISED recommendation" not in joined
