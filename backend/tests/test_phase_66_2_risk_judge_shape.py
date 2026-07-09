"""phase-66.2 RJ-shape fix (money-engine audit 2026-07-08, operator-approved
dark build). Full-orchestrator RiskJudge verdict nests under
risk_assessment['judge'] (risk_debate.py:310) but decide_trades read top-level
-> full-path BUYs sized at the 10%-NAV default, REJECT unenforceable, and
risk_judge_decision persisted ''. Flag paper_risk_judge_shape_fix_enabled
(default OFF) resolves nested-first; OFF = byte-identical top-level reads.

File name carries '66_2' so the immutable -k expression matches.
"""

from types import SimpleNamespace

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


# ── flag OFF: byte-identical legacy behavior (the documented defect) ────────

class TestFlagOffLegacy:
    def test_full_path_sizes_at_10pct_default_and_empty_decision(self):
        orders = decide_trades([], [_full_path_analysis()], [], PORTFOLIO, _settings())
        b = _buy(orders)
        assert b is not None
        # 10% NAV default (the bug): ~2399.77, NOT the judge's 3% (~719.93).
        assert abs(b.amount_usd - round(NAV * 0.10, 2)) < 0.5
        assert b.risk_judge_decision == ""  # nested decision missed

    def test_full_path_reject_not_blocked_even_binding_on(self):
        orders = decide_trades(
            [], [_full_path_analysis(decision="REJECT", pct=0)], [], PORTFOLIO,
            _settings(paper_risk_judge_reject_binding=True),
        )
        assert _buy(orders) is not None  # REJECT invisible top-level -> buys


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
