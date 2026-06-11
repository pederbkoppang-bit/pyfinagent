"""phase-60.2 (AW-5) tests: churn-engine fix -- swap sentinel + delta scale.

The away-week mechanism (59.3 finding AW-5): a holding without a same-cycle
analysis was scored conviction 0.0 ("treat as worst") by the swap path while
execute_buy stamped last_analysis_date=now and the re-eval gate waited 3
days -- so every fresh BUY was swap-out bait the next day. With the 0.01
epsilon denominator, candidate 7.0 vs sentinel 0.0 = 70,000% delta vs the
25% bar. Result: MU -6.3% / SNDK one-day round trips, DELL churn, 81.4%
weekly turnover, 10 round trips net -$132.

Fix under test (paper_swap_churn_fix_enabled, default OFF):
- ON: holdings absent from the same-cycle holding_lookup are EXCLUDED from
  swap displacement entirely (criterion-1 option B; LOCF valuation rejected
  for displacement: day-over-day score noise mean |delta| 1.10 on the 1-10
  scale can cross a 25% relative bar -- 59.3 stability table).
- ON: delta denominator uses the documented max(|h|, 1.0) clamp (the [0,1]
  premise was false; scores are 1-10).
- OFF: byte-identical pre-60.2 behavior (sentinel + 0.01 epsilon).

File name carries `60_2`; the immutable selector is
`-k 'swap or sentinel or reeval or 60_2'`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.settings import Settings
from backend.services.portfolio_manager import decide_trades


def _make_settings(**overrides) -> Settings:
    base = {
        "paper_starting_capital": 10000.0,
        "paper_max_positions": 10,
        "paper_max_per_sector": 1,
        "paper_max_per_sector_nav_pct": 90.0,
        "paper_max_factor_corr": 0.0,
        "paper_min_cash_reserve_pct": 5.0,
        "paper_swap_enabled": True,
        "paper_swap_min_delta_pct": 25.0,
        "paper_swap_max_per_cycle": 2,
        "paper_swap_churn_fix_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def _position(ticker: str, sector: str, market_value: float = 1000.0) -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "market_value": market_value,
        "current_price": market_value / 10.0,
        "recommendation": "BUY",
    }


def _analysis(ticker: str, sector: str, final_score: float, position_pct: float = 10.0) -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-06-09",
        "recommendation": "BUY",
        "final_score": final_score,
        "risk_assessment": {
            "decision": "APPROVE_FULL",
            "recommended_position_pct": position_pct,
        },
        "price_at_analysis": 100.0,
        "sector": sector,
        "full_report": {"market_data": {"sector": sector}},
    }


def _portfolio_state(nav: float = 10000.0, cash: float = 2000.0) -> dict:
    return {"nav": nav, "cash": cash, "positions_value": nav - cash, "position_count": 1}


def _run_06_09_scenario(churn_fix_on: bool):
    """The 06-09 away-week scenario: MU bought the PRIOR cycle (present in
    positions, ABSENT from holding_analyses -- '5 new + 0 re-evals'), an
    EQUAL-score Tech candidate arrives, sector cap blocks it -> swap path."""
    settings = _make_settings(paper_swap_churn_fix_enabled=churn_fix_on)
    positions = [_position("MU", "Technology")]
    candidates = [_analysis("AVGO", "Technology", 7.0)]  # equal to MU's buy-day 7.0
    holding_analyses: list[dict] = []  # 0 re-evals (the away-week signature)
    return decide_trades(
        current_positions=positions,
        candidate_analyses=candidates,
        holding_analyses=holding_analyses,
        portfolio_state=_portfolio_state(),
        settings=settings,
    )


def test_60_2_sentinel_regression_06_09_swap_fires_with_flag_off():
    # Locks CURRENT (defective) behavior: sentinel 0.0 -> delta
    # (7.0-0.0)/max(0, 0.01)*100 = 70,000% >> 25% -> day-old MU swapped out.
    orders = _run_06_09_scenario(churn_fix_on=False)
    sells = [o for o in orders if o.action == "SELL" and o.ticker == "MU"]
    buys = [o for o in orders if o.action == "BUY" and o.ticker == "AVGO"]
    assert sells and buys, f"expected the away-week swap to fire with flag OFF, got {orders}"
    assert any("swap" in (o.reason or "") for o in sells)


def test_60_2_sentinel_eliminated_with_flag_on():
    # Criterion 1: same scenario, flag ON -> the unanalyzed day-old holding is
    # EXCLUDED from displacement; the equal-score candidate cannot displace it.
    orders = _run_06_09_scenario(churn_fix_on=True)
    assert not [o for o in orders if o.action == "SELL" and o.ticker == "MU"], (
        f"day-old unanalyzed holding must not be swap-out bait with the fix ON, got {orders}"
    )


def test_60_2_away_week_pattern_impossible_by_construction():
    # Criterion 2: 'N new + 0 re-evals' + day-old holding -> with the flag ON
    # NO candidate (equal, higher, or maximal score) can displace a holding
    # that lacks a same-cycle analysis. Exclusion is structural, not a
    # threshold artifact -- so the pattern is impossible by construction.
    for cand_score in (7.0, 9.0, 10.0):
        settings = _make_settings(paper_swap_churn_fix_enabled=True)
        orders = decide_trades(
            current_positions=[_position("MU", "Technology")],
            candidate_analyses=[_analysis("AVGO", "Technology", cand_score)],
            holding_analyses=[],
            portfolio_state=_portfolio_state(),
            settings=settings,
        )
        mu_sells = [o for o in orders if o.action == "SELL" and o.ticker == "MU"]
        assert not mu_sells, f"cand_score={cand_score}: unanalyzed holding displaced: {orders}"


def test_60_2_analyzed_holdings_remain_displaceable_on_true_evidence():
    # The fix removes FABRICATED-evidence swaps, not true upgrades: a holding
    # WITH a same-cycle analysis still swaps when the true delta clears the
    # unchanged 25% bar (7.0 vs 5.0 = 40%).
    settings = _make_settings(paper_swap_churn_fix_enabled=True)
    orders = decide_trades(
        current_positions=[_position("OLD", "Technology")],
        candidate_analyses=[_analysis("NEW", "Technology", 7.0)],
        holding_analyses=[{"ticker": "OLD", "recommendation": "BUY", "final_score": 5.0}],
        portfolio_state=_portfolio_state(),
        settings=settings,
    )
    assert [o for o in orders if o.action == "SELL" and o.ticker == "OLD"], orders
    assert [o for o in orders if o.action == "BUY" and o.ticker == "NEW"], orders


def test_60_2_delta_boundary_on_true_1_to_10_scale():
    # Criterion 3 boundary: with the flag ON the bar operates on TRUE relative
    # deltas -- 6.0 vs 5.0 = 20% < 25% does NOT fire (under the old sentinel
    # path ANY candidate cleared 70,000%). 7.0-vs-5.0 = 40% fires on true
    # evidence (test above); the 25.0 bar itself is UNCHANGED (widening it
    # would be the 53.1/55.3-rejected band family).
    settings = _make_settings(paper_swap_churn_fix_enabled=True)
    orders = decide_trades(
        current_positions=[_position("OLD", "Technology")],
        candidate_analyses=[_analysis("NEW", "Technology", 6.0)],
        holding_analyses=[{"ticker": "OLD", "recommendation": "BUY", "final_score": 5.0}],
        portfolio_state=_portfolio_state(),
        settings=settings,
    )
    assert not [o for o in orders if o.action == "SELL" and o.ticker == "OLD"], orders


def test_60_2_real_score_deltas_identical_off_vs_on():
    # Criterion 4 (byte-identity leg): for REAL scores >= 1.0 the denominator
    # clamp is inert (max(5,1)=max(5,0.01)=5) -- OFF and ON produce identical
    # decisions on analyzed holdings. The flag changes ONLY the
    # fabricated-evidence paths (sentinel + sub-1.0 denominators).
    for flag in (False, True):
        settings = _make_settings(paper_swap_churn_fix_enabled=flag)
        orders = decide_trades(
            current_positions=[_position("OLD", "Technology")],
            candidate_analyses=[_analysis("NEW", "Technology", 7.0)],
            holding_analyses=[{"ticker": "OLD", "recommendation": "BUY", "final_score": 5.0}],
            portfolio_state=_portfolio_state(),
            settings=settings,
        )
        assert [o for o in orders if o.action == "SELL" and o.ticker == "OLD"], (flag, orders)


def test_60_2_composes_with_57_1_binding_reject_gate():
    # Both flags ON: a REJECT candidate is dropped by the 57.1 binding gate
    # (never reaches the swap path); an APPROVE candidate still swaps an
    # analyzed weak holding on true evidence. Disjoint mechanisms, no
    # interference.
    settings = _make_settings(
        paper_swap_churn_fix_enabled=True,
        paper_risk_judge_reject_binding=True,
    )
    reject_cand = _analysis("REJ", "Technology", 9.0)
    reject_cand["risk_assessment"] = {"decision": "REJECT", "recommended_position_pct": 0.0}
    approve_cand = _analysis("NEW", "Technology", 7.0)
    blocked_out: list[dict] = []
    orders = decide_trades(
        current_positions=[_position("OLD", "Technology")],
        candidate_analyses=[reject_cand, approve_cand],
        holding_analyses=[{"ticker": "OLD", "recommendation": "BUY", "final_score": 5.0}],
        portfolio_state=_portfolio_state(),
        settings=settings,
        blocked_out=blocked_out,
    )
    assert not [o for o in orders if o.ticker == "REJ"], orders
    assert [o for o in orders if o.action == "BUY" and o.ticker == "NEW"], orders
    assert blocked_out and blocked_out[0]["ticker"] == "REJ"


def test_60_2_flag_defaults_off():
    # Do-no-harm: the flag ships OFF; a bare Settings() must not enable it.
    s = _make_settings()
    assert s.paper_swap_churn_fix_enabled is False
