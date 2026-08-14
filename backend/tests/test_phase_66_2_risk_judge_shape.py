"""phase-66.2 RJ-shape fix (money-engine audit 2026-07-08, operator-approved
dark build). Full-orchestrator RiskJudge verdict nests under
risk_assessment['judge'] (risk_debate.py:310) but decide_trades read top-level
-> full-path BUYs sized at the 10%-NAV default, REJECT unenforceable, and
risk_judge_decision persisted ''. Flag paper_risk_judge_shape_fix_enabled
(default OFF) resolves nested-first; OFF = byte-identical top-level reads.

File name carries '66_2' so the immutable -k expression matches.
"""

from types import SimpleNamespace

import pytest

from backend.services.portfolio_manager import decide_trades


def _settings(**over):
    base = dict(
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_max_positions=10,
        paper_max_per_sector=2,
        paper_max_per_sector_nav_pct=30.0,
        paper_max_factor_corr=0.0,
        paper_swap_enabled=False,
        paper_default_stop_loss_pct=8.0,
        paper_risk_judge_reject_binding=False,
        paper_risk_judge_shape_fix_enabled=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


NAV = 23997.71
PORTFOLIO = {"nav": NAV, "cash": NAV, "position_count": 0}


def _full_path_analysis(decision="APPROVE_REDUCED", pct=3.0):
    """Full-orchestrator shape: judge nested under risk_assessment['judge']."""
    return {
        "ticker": "TST",
        "recommendation": "BUY",
        "final_score": 8.0,
        "price_at_analysis": 100.0,
        "analysis_date": "2026-07-08",
        "risk_assessment": {
            "judge": {"decision": decision, "recommended_position_pct": pct},
            "analysts": [],
        },
    }


def _lite_path_analysis(decision="APPROVE_REDUCED", pct=3.0):
    """Lite shape: flat decision/recommended_position_pct at top level."""
    return {
        "ticker": "TST",
        "recommendation": "BUY",
        "final_score": 8.0,
        "price_at_analysis": 100.0,
        "analysis_date": "2026-07-08",
        "risk_assessment": {"decision": decision, "recommended_position_pct": pct},
    }


def _buy(orders):
    return next((o for o in orders if o.action == "BUY"), None)


# ── phase-86.74: BOTH flag states, because production runs OFF ──────────────
#
# THESE TWO TESTS PREVIOUSLY ASSERTED THE DEFECT, and are rewritten rather than
# deleted so the inversion is visible in the diff:
#
#   test_full_path_sizes_at_10pct_default_and_empty_decision
#       asserted `abs(amount_usd - NAV*0.10) < 0.5` -- i.e. it REQUIRED the
#       10%-NAV default, with the comment "10% NAV default (the bug)".
#   test_full_path_reject_not_blocked_even_binding_on
#       asserted `_buy(orders) is not None` -- i.e. it REQUIRED a REJECT to buy,
#       with the comment "REJECT invisible top-level -> buys".
#
# Both encoded the DELL defect as expected behaviour, so the suite was green
# while the bug ran in production. They now assert the corrected behaviour and
# fail if it regresses.

BOTH_FLAG_STATES = [False, True]


class TestRejectBindsInBothFlagStates:
    """DELL's exact case: a nested REJECT at 0% must NEVER produce an order.

    Parametrised over the flag because the SHIPPED PRODUCTION STATE IS OFF, and
    a test that only covers the corrected branch cannot fail when the shipped
    branch is the broken one (criterion 8).
    """

    @pytest.mark.parametrize("shape_fix", BOTH_FLAG_STATES)
    def test_full_path_reject_at_zero_pct_produces_no_order(self, shape_fix):
        orders = decide_trades(
            [], [_full_path_analysis(decision="REJECT", pct=0)], [], PORTFOLIO,
            _settings(paper_risk_judge_shape_fix_enabled=shape_fix,
                      paper_risk_judge_reject_binding=True),
        )
        assert _buy(orders) is None, (
            f"DELL regression: a REJECT/0% verdict produced a BUY "
            f"(shape_fix={shape_fix})"
        )

    @pytest.mark.parametrize("shape_fix", BOTH_FLAG_STATES)
    def test_full_path_zero_pct_never_sizes_at_the_10pct_default(self, shape_fix):
        """The inversion itself: 0% must not become the LARGEST position."""
        orders = decide_trades(
            [], [_full_path_analysis(decision="REJECT", pct=0)], [], PORTFOLIO,
            _settings(paper_risk_judge_shape_fix_enabled=shape_fix),
        )
        b = _buy(orders)
        if b is not None:
            assert abs(b.amount_usd - round(NAV * 0.10, 2)) > 0.5, (
                f"0% verdict inverted into the 10%-NAV default "
                f"(shape_fix={shape_fix})"
            )

    @pytest.mark.parametrize("shape_fix", BOTH_FLAG_STATES)
    def test_full_path_sizes_at_the_judges_pct_not_the_default(self, shape_fix):
        orders = decide_trades(
            [], [_full_path_analysis(decision="APPROVE_REDUCED", pct=3.0)], [],
            PORTFOLIO, _settings(paper_risk_judge_shape_fix_enabled=shape_fix),
        )
        b = _buy(orders)
        assert b is not None
        assert abs(b.amount_usd - round(NAV * 0.03, 2)) < 0.5, (
            f"sized at {b.amount_usd}, expected the judge's 3% "
            f"({round(NAV * 0.03, 2)}), shape_fix={shape_fix}"
        )
        assert b.risk_judge_decision == "APPROVE_REDUCED"


# ── flag ON: nested judge resolved ─────────────────────────────────────────

class TestFlagOnResolved:
    def test_full_path_sizes_at_judge_pct_and_records_decision(self):
        orders = decide_trades(
            [], [_full_path_analysis(decision="APPROVE_REDUCED", pct=3.0)], [],
            PORTFOLIO, _settings(paper_risk_judge_shape_fix_enabled=True),
        )
        b = _buy(orders)
        assert b is not None
        assert abs(b.amount_usd - round(NAV * 0.03, 2)) < 0.5  # judge's 3%, not 10%
        assert b.risk_judge_decision == "APPROVE_REDUCED"  # recorded, not ''

    def test_full_path_reject_binds_when_binding_on(self):
        blocked = []
        orders = decide_trades(
            [], [_full_path_analysis(decision="REJECT", pct=0)], [], PORTFOLIO,
            _settings(paper_risk_judge_shape_fix_enabled=True,
                      paper_risk_judge_reject_binding=True),
            blocked_out=blocked,
        )
        assert _buy(orders) is None
        assert blocked and blocked[0]["decision"] == "REJECT"

    def test_explicit_zero_pct_is_no_buy_not_10pct_default(self):
        # APPROVE with 0% -> min-ticket floor skips (no 10% default inversion).
        orders = decide_trades(
            [], [_full_path_analysis(decision="APPROVE_REDUCED", pct=0.0)], [],
            PORTFOLIO, _settings(paper_risk_judge_shape_fix_enabled=True),
        )
        assert _buy(orders) is None

    def test_lite_path_unaffected_flag_on(self):
        # Lite is already flat; flag ON must not change it.
        orders = decide_trades(
            [], [_lite_path_analysis(decision="APPROVE_REDUCED", pct=3.0)], [],
            PORTFOLIO, _settings(paper_risk_judge_shape_fix_enabled=True),
        )
        b = _buy(orders)
        assert b is not None
        assert abs(b.amount_usd - round(NAV * 0.03, 2)) < 0.5
        assert b.risk_judge_decision == "APPROVE_REDUCED"

    def test_lite_path_byte_identical_across_flag(self):
        off = _buy(decide_trades([], [_lite_path_analysis()], [], PORTFOLIO, _settings()))
        on = _buy(decide_trades(
            [], [_lite_path_analysis()], [], PORTFOLIO,
            _settings(paper_risk_judge_shape_fix_enabled=True)))
        assert off.amount_usd == on.amount_usd
        assert off.risk_judge_decision == on.risk_judge_decision


def test_settings_flag_default_off():
    from backend.config.settings import Settings
    assert Settings.model_fields["paper_risk_judge_shape_fix_enabled"].default is False


# ── phase-66.2 review C1: None-safe recommendation guard (crash fix) ─────────

def test_none_recommendation_does_not_crash_decide_trades():
    """The lite fallback can return recommendation=None; decide_trades read
    analysis.get('recommendation','HOLD').upper() which crashed on present-None
    (the .get default only fires on a MISSING key). Guard makes None -> HOLD."""
    a_new = {"ticker": "NN", "recommendation": None, "final_score": 7.0,
             "price_at_analysis": 100.0, "analysis_date": "x", "risk_assessment": {}}
    a_hold = {"ticker": "HH", "recommendation": None, "final_score": 3.0,
              "analysis_date": "x", "risk_assessment": {}, "current_price": 50.0}
    pos = {"ticker": "HH", "recommendation": "BUY", "quantity": 5.0,
           "avg_entry_price": 50.0, "cost_basis": 250.0, "current_price": 50.0,
           "market_value": 250.0, "stop_loss_price": 40.0, "sector": "Tech"}
    # must not raise (was AttributeError: 'NoneType' object has no attribute 'upper')
    orders = decide_trades([pos], [a_new], [a_hold],
                           {"nav": 10000.0, "cash": 9000.0, "position_count": 1},
                           _settings())
    # None rec is treated as HOLD -> no BUY for NN, no downgrade-SELL for HH
    assert not any(o.action == "BUY" for o in orders)


# ═══════════════════════════════════════════════════════════════════════════
# phase-86.74 -- the falsy-zero inversion (DELL, 2026-08-13)
#
# `BUY 4.8064 x DELL @ $497.72 = $2392.26` on NAV 23,920.63 = EXACTLY 10.00% of
# NAV, against a risk verdict of REJECT / HIGH / 0%. `Optional[float]` cannot
# carry three states, so "the judge said zero" and "the judge said nothing"
# were the same `None`, and every `or 10.0` downstream resolved that ambiguity
# in the most dangerous direction.
# ═══════════════════════════════════════════════════════════════════════════

from backend.services.portfolio_manager import (  # noqa: E402
    ABSENT,
    DEFAULT_POSITION_PCT,
    SIZE,
    UNPARSEABLE,
    _extract_position_pct,
    _resolve_position_pct,
    _sizing_pct,
)


class TestHelperDistinguishesZeroFromAbsent:
    """Criterion 1 -- fixed AT THE HELPER, and unconditional (no flag involved).

    None of these tests take a settings object at all: the helper has no flag to
    read. That is the point -- the shipped production state is flag-OFF, so a fix
    that lives behind the flag does not fix production.
    """

    def test_explicit_zero_is_a_SIZE_not_an_absence(self):
        v = _resolve_position_pct({"recommended_position_pct": 0.0}, {})
        assert v.kind == SIZE
        assert v.pct == 0.0
        assert v.blocks_buy is True

    def test_absent_is_ABSENT(self):
        assert _resolve_position_pct({}, {}).kind == ABSENT

    def test_zero_and_absent_are_DISTINGUISHABLE(self):
        """The whole defect in one assertion: pre-86.74 both were `None`."""
        zero = _resolve_position_pct({"recommended_position_pct": 0.0}, {})
        absent = _resolve_position_pct({}, {})
        assert zero != absent

    def test_second_source_also_honours_explicit_zero(self):
        """R1: the fallback source kept `if pct:` under EVERY flag setting --
        the phase-66.2 flag only ever worked around the FIRST source."""
        v = _resolve_position_pct({}, {"risk_judge_position_pct": 0.0})
        assert v.kind == SIZE and v.pct == 0.0

    def test_malformed_verdict_is_UNPARSEABLE_not_absent(self):
        """R2: a present-but-garbage value must not be read as silence."""
        v = _resolve_position_pct({"recommended_position_pct": "garbage"}, {})
        assert v.kind == UNPARSEABLE
        assert v.blocks_buy is True

    def test_legacy_shim_returns_zero_not_none(self):
        """The exact falsy-zero: `if pct:` returned None here, which became 10%."""
        assert _extract_position_pct({"recommended_position_pct": 0.0}, {}) == 0.0

    def test_legacy_shim_still_returns_none_when_absent(self):
        assert _extract_position_pct({}, {}) is None


class TestSizingDefaultReachableOnlyFromAbsent:
    """Criterion 3 -- the default is now reachable from ONE function, so the
    claim "only a genuinely absent verdict reaches it" is checkable by
    enumerating that function's branches rather than auditing four call sites."""

    def test_absent_verdict_reaches_the_default(self):
        assert _sizing_pct({"position_pct": None,
                            "position_pct_state": ABSENT}) == DEFAULT_POSITION_PCT

    def test_zero_verdict_does_NOT_reach_the_default(self):
        assert _sizing_pct({"position_pct": 0.0, "position_pct_state": SIZE}) == 0.0

    def test_unparseable_verdict_does_NOT_reach_the_default(self):
        assert _sizing_pct({"position_pct": None,
                            "position_pct_state": UNPARSEABLE}) == 0.0

    def test_specified_size_is_returned_verbatim(self):
        assert _sizing_pct({"position_pct": 3.0, "position_pct_state": SIZE}) == 3.0

    def test_default_is_reachable_from_ABSENT_AND_NOTHING_ELSE(self):
        """Criterion 3, DERIVED by exhaustive sweep rather than asserted in prose.

        The first version of this step ASSERTED "the default is reachable from
        ABSENT and only ABSENT". The 86.74 Q/A executed the function over its
        state/pct grid and found that FALSE: `(SIZE, pct=None)` and any
        UNRECOGNISED state also returned the default -- and the latter OVERRODE
        an explicit 0.0. Both were unreachable in production, so it was a false
        CLAIM rather than a live defect; the function now fails closed on both,
        and this test derives the set instead of restating it.
        """
        states = [SIZE, ABSENT, UNPARSEABLE, None, "BOGUS", ""]
        # DELIBERATELY EXCLUDES pct == DEFAULT_POSITION_PCT. The return value is
        # only a PROXY for which branch ran, and a judge that explicitly says
        # "10%" returns the same number as the default branch -- so including it
        # would report a legitimate explicit size as a default-reach. Caught by
        # this very test on its first run.
        pcts = [None, 0.0, 3.0, 7.5]
        assert DEFAULT_POSITION_PCT not in pcts, "a probe value collides with the default"
        defaulting = []
        for st in states:
            for pc in pcts:
                cand = {"position_pct": pc}
                if st is not None:
                    cand["position_pct_state"] = st
                if _sizing_pct(cand) == DEFAULT_POSITION_PCT:
                    defaulting.append((st, pc))
        # Every surviving default-yielding cell must be a genuinely absent
        # verdict: an explicit ABSENT, or the legacy no-state/no-pct shape that
        # `_sizing_pct` derives to ABSENT.
        offenders = [
            (st, pc) for st, pc in defaulting
            if not (st == ABSENT or (st is None and pc is None))
        ]
        assert offenders == [], (
            f"the 10% default is reachable from non-absent verdicts: {offenders}"
        )
        # The sweep must actually FIND the legitimate ones, else `offenders == []`
        # could pass simply because nothing ever returned the default.
        assert (ABSENT, 0.0) in defaulting and (None, None) in defaulting, (
            f"sweep found no legitimate default path -- probe is vacuous: {defaulting}"
        )

    def test_contradictory_SIZE_with_no_number_fails_closed(self):
        assert _sizing_pct({"position_pct": None, "position_pct_state": SIZE}) == 0.0

    def test_unrecognised_state_never_overrides_an_explicit_zero(self):
        assert _sizing_pct({"position_pct": 0.0, "position_pct_state": "BOGUS"}) == 0.0

    def test_legacy_candidate_without_state_key_still_safe_on_zero(self):
        """Defensive branch: a cand built by a path predating the state key must
        still treat an explicit 0.0 as a size, not as silence."""
        assert _sizing_pct({"position_pct": 0.0}) == 0.0
        assert _sizing_pct({"position_pct": None}) == DEFAULT_POSITION_PCT

    def test_every_sizing_site_routes_through_the_single_seam(self):
        """Criterion 3, derived from SOURCE rather than asserted.

        Pre-86.74 four sites sized with `or 10.0` (:507 flag-guarded, :800, :853,
        :878 unguarded). Parsed with the AST so the check cannot be fooled by the
        phrase appearing in a comment or docstring -- including in THIS file's own
        explanatory text.
        """
        import ast
        import pathlib

        src = pathlib.Path(
            "backend/services/portfolio_manager.py").read_text()
        tree = ast.parse(src)
        offenders = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
            and any(isinstance(v, ast.Constant) and v.value == DEFAULT_POSITION_PCT
                    for v in n.values)
        ]
        assert offenders == [], (
            f"`or {DEFAULT_POSITION_PCT}` sizing idiom reintroduced at lines "
            f"{offenders}; route it through _sizing_pct instead"
        )

        # POSITIVE CONTROL: the detector must be able to FIND one, else `== []`
        # is vacuous and this test would pass on an empty or unparsed file.
        ctl = ast.parse("x = cand.get('position_pct') or 10.0")
        assert [
            n.lineno for n in ast.walk(ctl)
            if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
            and any(isinstance(v, ast.Constant) and v.value == DEFAULT_POSITION_PCT
                    for v in n.values)
        ], "detector is blind -- the `offenders == []` assertion above is vacuous"


# ═══════════════════════════════════════════════════════════════════════════
# phase-86.74 criteria 4/5/6 -- the verdict must leave a DURABLE, ATTRIBUTABLE
# trace. All three defects share one consequence: this incident had to be
# reconstructed from log timestamps by elimination, because the verdict existed
# nowhere an auditor could query per-ticker.
# ═══════════════════════════════════════════════════════════════════════════

class TestVerdictIsPersistedPerTicker:
    """Criterion 4. Measured baseline: risk_judge_decision / risk_level /
    recommended_position_pct were empty on 129 of 129 analysis_results rows over
    2026-07-20..2026-08-13 -- while `save_report` accepted all three the entire
    time. The autonomous loop's writer simply never passed them."""

    def _capture(self, analysis):
        import asyncio
        from backend.services import autonomous_loop as al

        captured = {}

        class _BQ:
            def save_report(self, **kw):
                captured.update(kw)

        asyncio.run(al._persist_analysis(analysis, _BQ()))
        # `_persist_analysis` swallows every exception by design (a BQ failure
        # must not halt the trading cycle), so an empty capture means the write
        # never happened rather than that it wrote nothing. Fail loudly here, or
        # these assertions could pass vacuously on a silently-broken writer.
        assert captured, "save_report was never called -- the write path errored"
        return captured

    def test_nested_judge_verdict_reaches_its_own_columns(self):
        got = self._capture({
            "ticker": "TST", "final_score": 8.0, "recommendation": "Buy",
            "full_report": {},
            "risk_assessment": {"judge": {
                "decision": "REJECT", "risk_level": "HIGH",
                "recommended_position_pct": 0,
            }},
        })
        assert got.get("risk_judge_decision") == "REJECT"
        assert got.get("risk_level") == "HIGH"
        # 0.0, not None -- the falsy-zero must not erase the strongest verdict
        # on its way into the audit record either.
        assert got.get("recommended_position_pct") == 0.0

    def test_flat_lite_verdict_also_reaches_the_columns(self):
        got = self._capture({
            "ticker": "TST", "final_score": 8.0, "recommendation": "Buy",
            "full_report": {},
            "risk_assessment": {"decision": "APPROVE_REDUCED",
                                "risk_level": "MODERATE",
                                "recommended_position_pct": 3},
        })
        assert got.get("risk_judge_decision") == "APPROVE_REDUCED"
        assert got.get("recommended_position_pct") == 3.0

    def test_absent_verdict_persists_empty_not_crash(self):
        got = self._capture({
            "ticker": "TST", "final_score": 8.0, "recommendation": "Buy",
            "full_report": {}, "risk_assessment": {},
        })
        assert got.get("risk_judge_decision") == ""
        assert got.get("recommended_position_pct") is None


def test_risk_debate_completion_log_carries_the_ticker():
    """Criterion 5. The completion line logged decision/risk_level/position/rounds
    and NO ticker, so six concurrent debates on 2026-08-13 were unattributable and
    DELL's verdict had to be identified BY ELIMINATION. Asserted against the
    source text because the line is emitted deep inside a multi-LLM-call debate
    that cannot run in a unit test."""
    import inspect

    from backend.agents import risk_debate

    src = inspect.getsource(risk_debate.run_risk_debate)
    marker = "Risk debate complete:"
    assert marker in src, "completion log line not found -- probe is stale"
    line = next(ln for ln in src.splitlines() if marker in ln)
    assert "ticker=" in line, (
        f"completion log line carries no ticker, so concurrent debates stay "
        f"unattributable: {line.strip()!r}"
    )


class TestRiskJudgeAppearsInFactorsJson:
    """Criterion 6. DELL persisted 3 agents / 517 chars with NO RiskJudge row;
    NTAP 2026-07-31 persisted 4 agents / 1232 chars WITH `RiskJudge (gate)`. The
    missing row in the Agent Rationale UI is the only reason this was caught."""

    @staticmethod
    def _agents(analysis):
        from backend.services.signal_attribution import extract_all_signals
        return [s["agent"] for s in extract_all_signals(analysis)]

    def test_nested_judge_emits_a_riskjudge_row(self):
        agents = self._agents({
            "recommendation": "BUY",
            "risk_assessment": {"judge": {
                "decision": "APPROVE_REDUCED", "recommended_position_pct": 3}},
        })
        assert "RiskJudge" in agents

    def test_zero_pct_reject_still_emits_a_riskjudge_row(self):
        """The 0% REJECT -- the case an auditor most needs -- must not vanish."""
        agents = self._agents({
            "recommendation": "BUY",
            "risk_assessment": {"judge": {
                "decision": "REJECT", "recommended_position_pct": 0}},
        })
        assert "RiskJudge" in agents, (
            "a 0% REJECT left no trace in factors_json -- this is the DELL shape"
        )

    def test_pct_zero_with_no_decision_text_still_emits(self):
        agents = self._agents({
            "recommendation": "BUY",
            "risk_assessment": {"judge": {"recommended_position_pct": 0}},
        })
        assert "RiskJudge" in agents

    def test_genuinely_empty_risk_assessment_emits_nothing(self):
        """The guard must still be able to say 'no verdict' -- otherwise the
        assertions above would pass vacuously on any input at all."""
        assert "RiskJudge" not in self._agents({
            "recommendation": "BUY", "risk_assessment": {}})
