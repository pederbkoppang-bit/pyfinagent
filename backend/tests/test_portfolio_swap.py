"""phase-cycle-1 (2026-05-26): position-swap framework regression test.

Reproduces the 2026-05-26 zero-buy scenario:
- 9 positions, 8 Tech (final_score 0.55-0.75) + 1 Industrials.
- 3 candidates: 2 Tech (final_score 0.85, 0.82), 1 Industrials (0.70).
- Sector COUNT cap = 2 -- Tech buys blocked at the cap.

Asserts the swap framework emits 2 swap pairs (lowest-Tech holdings replaced
by highest-Tech candidates) PLUS 1 standard BUY of the Industrials candidate
filling the open slot. Per the testing-phase mandate + north-star (maximize
profit at lowest cost; default to firing, not gating, when risk caps permit).

Citations in handoff/current/contract.md.
"""
from __future__ import annotations

from backend.config.settings import Settings
from backend.services.portfolio_manager import decide_trades


def _make_settings(**overrides) -> Settings:
    """Lean settings fixture; only the fields decide_trades reads."""
    base = {
        "paper_starting_capital": 10000.0,
        "paper_max_positions": 10,
        "paper_max_per_sector": 2,
        "paper_max_per_sector_nav_pct": 30.0,
        "paper_max_factor_corr": 0.0,  # disabled for clarity
        "paper_min_cash_reserve_pct": 5.0,
        "paper_swap_enabled": True,
        "paper_swap_min_delta_pct": 25.0,
        "paper_swap_max_per_cycle": 2,
    }
    base.update(overrides)
    return Settings(**base)


def _holding(ticker: str, sector: str, market_value: float, final_score: float) -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "market_value": market_value,
        "current_price": market_value / 10.0,  # 10 shares each for simplicity
        "recommendation": "BUY",
    }


def _holding_analysis(ticker: str, final_score: float) -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-05-26",
        "recommendation": "BUY",  # HOLD rec to prevent the sell-path triggering
        "final_score": final_score,
    }


def _candidate_analysis(ticker: str, sector: str, final_score: float, position_pct: float = 10.0) -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-05-26",
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


def test_swap_framework_fills_zero_buy_gap():
    """The 2026-05-26 scenario: 8/9 Tech + sector cap = zero-buy without swap.

    With swap enabled, expect 2 swap pairs + 1 standard BUY = 5 orders total
    (2 SELLs + 3 BUYs).
    """
    nav = 10_000.0
    # Cash above the 5% min_cash_reserve ($500) so the buy-loop's
    # `available_cash <= 0` guard doesn't short-circuit before
    # sector_blocked can populate. $2000 supports the Industrials slot-fill
    # AND leaves the swap path to do the Tech rebalance net-zero on cash.
    cash = 2_000.0
    # 8 Tech holdings with scores 0.55-0.75; 1 Industrials with score 0.65.
    tech_scores = [0.55, 0.58, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75]
    positions = [
        _holding(f"TECH{i}", "Technology", 1100.0, s)
        for i, s in enumerate(tech_scores)
    ]
    positions.append(_holding("INDU1", "Industrials", 1000.0, 0.65))

    # Re-evaluation: all hold (rec=BUY), no sell signal. So decide_trades
    # should NOT generate any signal-based SELLs.
    holding_analyses = []
    for i, p in enumerate(positions):
        score = tech_scores[i] if i < 8 else 0.65
        holding_analyses.append(_holding_analysis(p["ticker"], float(score)))

    # 3 candidates: 2 Tech high-conviction, 1 Industrials.
    candidate_analyses = [
        _candidate_analysis("TECH_NEW1", "Technology", 0.85),
        _candidate_analysis("TECH_NEW2", "Technology", 0.82),
        _candidate_analysis("INDU_NEW", "Industrials", 0.70),
    ]

    portfolio_state = {
        "nav": nav,
        "cash": cash,
        "positions_value": nav - cash,
        "position_count": len(positions),
    }
    settings = _make_settings()

    orders = decide_trades(
        current_positions=positions,
        candidate_analyses=candidate_analyses,
        holding_analyses=holding_analyses,
        portfolio_state=portfolio_state,
        settings=settings,
    )

    reasons = [o.reason for o in orders]
    actions = [o.action for o in orders]

    # Assert: 2 swap-SELLs + 2 swap-BUYs + 1 standard BUY = 5 orders total.
    swap_sells = [o for o in orders if o.reason == "swap_for_higher_conviction"]
    swap_buys = [o for o in orders if o.reason == "swap_buy"]
    standard_buys = [o for o in orders if o.reason == "new_buy_signal"]

    assert len(swap_sells) == 2, (
        f"Expected 2 swap SELLs, got {len(swap_sells)}; orders={orders}"
    )
    assert len(swap_buys) == 2, (
        f"Expected 2 swap BUYs, got {len(swap_buys)}; orders={orders}"
    )
    assert len(standard_buys) == 1, (
        f"Expected 1 standard Industrials BUY, got {len(standard_buys)}; orders={orders}"
    )

    # Assert: swap pairs SELL the LOWEST-score Tech holdings (TECH0 + TECH1).
    sold_tickers = {o.ticker for o in swap_sells}
    assert sold_tickers == {"TECH0", "TECH1"}, (
        f"Expected swap SELLs of TECH0+TECH1 (lowest scores), got {sold_tickers}"
    )

    # Assert: swap pairs BUY the HIGHEST-score Tech candidates.
    bought_tickers = {o.ticker for o in swap_buys}
    assert bought_tickers == {"TECH_NEW1", "TECH_NEW2"}, (
        f"Expected swap BUYs of TECH_NEW1+TECH_NEW2, got {bought_tickers}"
    )

    # Assert: standard BUY is the Industrials candidate filling the open slot.
    assert standard_buys[0].ticker == "INDU_NEW"

    # Assert: sell-first-then-buy ordering preserved.
    # All SELLs come before all BUYs in the orders list.
    first_buy_idx = next(i for i, a in enumerate(actions) if a == "BUY")
    last_sell_idx = max(
        (i for i, a in enumerate(actions) if a == "SELL"),
        default=-1,
    )
    assert last_sell_idx < first_buy_idx, (
        f"sell-first-then-buy violated: last SELL idx={last_sell_idx} >= first BUY idx={first_buy_idx}"
    )


def test_swap_disabled_reproduces_zero_buy():
    """With swap disabled, the 2026-05-26 scenario emits ZERO Tech BUYs.

    This is the pre-cycle-1 behavior. Captures the regression baseline so the
    framework's effect is unambiguous.
    """
    nav = 10_000.0
    tech_scores = [0.55, 0.58, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75]
    positions = [
        _holding(f"TECH{i}", "Technology", 1100.0, s)
        for i, s in enumerate(tech_scores)
    ]
    positions.append(_holding("INDU1", "Industrials", 1000.0, 0.65))

    holding_analyses = []
    for i, p in enumerate(positions):
        score = tech_scores[i] if i < 8 else 0.65
        holding_analyses.append(_holding_analysis(p["ticker"], float(score)))

    candidate_analyses = [
        _candidate_analysis("TECH_NEW1", "Technology", 0.85),
        _candidate_analysis("TECH_NEW2", "Technology", 0.82),
        _candidate_analysis("INDU_NEW", "Industrials", 0.70),
    ]

    portfolio_state = {
        "nav": nav,
        "cash": 2_000.0,  # above min_cash_reserve ($500); allows Industrials slot-fill.
        "positions_value": nav - 2_000.0,
        "position_count": len(positions),
    }
    # SWAP DISABLED -- pre-cycle-1 behavior.
    settings = _make_settings(paper_swap_enabled=False)

    orders = decide_trades(
        current_positions=positions,
        candidate_analyses=candidate_analyses,
        holding_analyses=holding_analyses,
        portfolio_state=portfolio_state,
        settings=settings,
    )

    swap_reasons = {o.reason for o in orders if "swap" in o.reason}
    assert swap_reasons == set(), (
        f"Swap-disabled cycle should emit ZERO swap orders, got {swap_reasons}"
    )
    # The Industrials candidate still fits in the 1 open slot.
    standard_buys = [o for o in orders if o.reason == "new_buy_signal"]
    assert any(o.ticker == "INDU_NEW" for o in standard_buys), (
        f"Industrials slot-fill should still happen; orders={orders}"
    )


def test_swap_skips_below_threshold():
    """When the delta is below the threshold, no swap fires.

    Holding at score 0.78, candidate at score 0.85 -- delta = 8.97% < 25%
    threshold. Expect zero swap orders.
    """
    nav = 10_000.0
    positions = [
        _holding("TECH_A", "Technology", 1000.0, 0.78),
        _holding("TECH_B", "Technology", 1000.0, 0.80),
    ]
    holding_analyses = [
        _holding_analysis("TECH_A", 0.78),
        _holding_analysis("TECH_B", 0.80),
    ]
    candidate_analyses = [
        # Delta vs TECH_A = (0.85 - 0.78) / 0.78 * 100 = 8.97% < 25%
        _candidate_analysis("TECH_NEW", "Technology", 0.85),
    ]
    portfolio_state = {
        "nav": nav,
        "cash": 100.0,
        "positions_value": nav - 100.0,
        "position_count": 2,
    }
    settings = _make_settings(
        paper_max_positions=2,  # full -- no open slot
        paper_swap_min_delta_pct=25.0,
    )

    orders = decide_trades(
        current_positions=positions,
        candidate_analyses=candidate_analyses,
        holding_analyses=holding_analyses,
        portfolio_state=portfolio_state,
        settings=settings,
    )
    swap_orders = [o for o in orders if "swap" in o.reason]
    assert swap_orders == [], (
        f"Below-threshold delta should not fire swap; orders={orders}"
    )


def test_swap_respects_max_per_cycle():
    """3 sector-blocked Tech candidates with paper_swap_max_per_cycle=1 -> 1 swap.

    9-position portfolio (8 Tech + 1 Indu) with max=10 so the buy-loop ENTERS
    (1 open slot), each Tech candidate blocks at sector cap and queues into
    sector_blocked. The max_per_cycle ceiling then caps the swap output.
    """
    nav = 10_000.0
    tech_scores = [0.55, 0.58, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75]
    positions = [
        _holding(f"TECH{i}", "Technology", 1000.0, s)
        for i, s in enumerate(tech_scores)
    ]
    positions.append(_holding("INDU1", "Industrials", 1000.0, 0.65))
    holding_analyses = []
    for i, p in enumerate(positions):
        score = tech_scores[i] if i < 8 else 0.65
        holding_analyses.append(_holding_analysis(p["ticker"], float(score)))

    # 3 Tech candidates -- ALL sector-blocked; max_per_cycle=1 caps to 1 swap.
    candidate_analyses = [
        _candidate_analysis(f"TECH_NEW{i}", "Technology", 0.90 - i * 0.01)
        for i in range(3)
    ]
    portfolio_state = {
        "nav": nav,
        "cash": 2_000.0,
        "positions_value": nav - 2_000.0,
        "position_count": len(positions),
    }
    settings = _make_settings(
        paper_max_positions=10,
        paper_swap_max_per_cycle=1,
    )
    orders = decide_trades(
        current_positions=positions,
        candidate_analyses=candidate_analyses,
        holding_analyses=holding_analyses,
        portfolio_state=portfolio_state,
        settings=settings,
    )
    swap_buys = [o for o in orders if o.reason == "swap_buy"]
    assert len(swap_buys) == 1, (
        f"max_per_cycle=1 should produce 1 swap, got {len(swap_buys)}; orders={orders}"
    )
