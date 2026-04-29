"""phase-23.1.13: decide_trades enforces paper_max_per_sector cap."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.services.portfolio_manager import decide_trades


def _settings(max_per_sector: int = 2, max_positions: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        paper_max_per_sector=max_per_sector,
        paper_max_positions=max_positions,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_default_stop_loss_pct=8.0,
    )


def _analysis(ticker: str, sector: str = "Technology", score: float = 7.0) -> dict:
    return {
        "ticker": ticker,
        "recommendation": "BUY",
        "final_score": score,
        "risk_assessment": {"reason": "test"},
        "price_at_analysis": 100.0,
        "analysis_date": "2026-04-28T00:00:00+00:00",
        "_path": "lite",
        "full_report": {"market_data": {"sector": sector}},
    }


def _portfolio_state(nav: float = 10000.0, cash: float = 10000.0) -> dict:
    return {"nav": nav, "cash": cash, "positions_value": 0.0, "position_count": 0}


def test_third_tech_buy_skipped_when_cap_is_2():
    """3 BUY candidates all in Technology, cap=2 -> only first 2 booked."""
    candidates = [_analysis(t) for t in ("INTC", "NVDA", "AMD")]
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert len(buys) == 2
    assert {o.ticker for o in buys} == {"INTC", "NVDA"}  # first two by score order


def test_disabled_cap_passes_all_through():
    """paper_max_per_sector=0 -> no cap enforced -> all 3 Tech candidates booked."""
    candidates = [_analysis(t) for t in ("INTC", "NVDA", "AMD")]
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(),
        settings=_settings(max_per_sector=0),
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert len(buys) == 3


def test_cap_counts_existing_positions():
    """2 existing Tech positions + cap=2 -> NO new Tech buys allowed."""
    existing = [
        {"ticker": "INTC", "sector": "Technology", "quantity": 10, "current_price": 100, "avg_entry_price": 100, "market_value": 1000, "recommendation": "BUY"},
        {"ticker": "NVDA", "sector": "Technology", "quantity": 5, "current_price": 200, "avg_entry_price": 200, "market_value": 1000, "recommendation": "BUY"},
    ]
    candidates = [_analysis("AMD", "Technology"), _analysis("XOM", "Energy")]
    orders = decide_trades(
        current_positions=existing,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    # AMD blocked (Tech at cap); XOM accepted (different sector)
    assert {o.ticker for o in buys} == {"XOM"}


def test_unknown_sector_treated_as_own_bucket():
    """Candidates with no sector -> Unknown bucket; counted independently."""
    candidates = [
        _analysis("INTC", "Technology"),
        _analysis("NVDA", "Technology"),
        {**_analysis("XYZ"), "full_report": {"market_data": {"sector": ""}}},
    ]
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    # 2 Tech + 1 Unknown all accepted (different sectors)
    assert {o.ticker for o in buys} == {"INTC", "NVDA", "XYZ"}


def test_diverse_sectors_all_booked():
    """5 candidates across 5 sectors -> all 5 booked."""
    candidates = [
        _analysis("INTC", "Technology"),
        _analysis("XOM", "Energy"),
        _analysis("JPM", "Financials"),
        _analysis("PG", "Consumer Staples"),
        _analysis("LLY", "Health Care"),
    ]
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert len(buys) == 5


def test_legacy_position_with_enriched_sector_blocks_same_sector_buy():
    """phase-23.1.14: positions whose sector was empty in BQ but enriched
    upstream (autonomous_loop._fetch_ticker_meta) must count toward the cap.
    Simulates the post-enrichment state: 2 Tech positions with sector populated.
    A new Tech BUY must be blocked, an Energy BUY must pass."""
    enriched_positions = [
        {"ticker": "INTC", "sector": "Technology", "quantity": 10, "current_price": 100, "avg_entry_price": 100, "market_value": 1000, "recommendation": "BUY"},
        {"ticker": "AAPL", "sector": "Technology", "quantity": 5, "current_price": 200, "avg_entry_price": 200, "market_value": 1000, "recommendation": "BUY"},
    ]
    candidates = [_analysis("MU", "Technology"), _analysis("XOM", "Energy")]
    orders = decide_trades(
        current_positions=enriched_positions,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert {o.ticker for o in buys} == {"XOM"}


def test_legacy_position_without_enrichment_falls_into_unknown():
    """phase-23.1.14 regression guard: positions with `sector=None`
    (BQ paper_positions rows predating the sector column) fall into the
    'Unknown' bucket. sector_counts['Technology'] stays 0, and new Tech BUYs
    pass the cap. This is the Bug A baseline that the autonomous_loop
    enrichment block exists to prevent in production."""
    legacy_positions = [
        {"ticker": "INTC", "sector": None, "quantity": 10, "current_price": 100, "avg_entry_price": 100, "market_value": 1000, "recommendation": "BUY"},
        {"ticker": "AAPL", "sector": None, "quantity": 5, "current_price": 200, "avg_entry_price": 200, "market_value": 1000, "recommendation": "BUY"},
    ]
    candidates = [_analysis("MU", "Technology"), _analysis("KEYS", "Technology")]
    orders = decide_trades(
        current_positions=legacy_positions,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    # Both Tech BUYs pass because the legacy positions are in 'Unknown' bucket
    # not 'Technology'. This documents the bug; production fix lives in
    # autonomous_loop.py which enriches before calling decide_trades.
    assert {o.ticker for o in buys} == {"MU", "KEYS"}


def test_sector_priority_via_candidates_by_ticker():
    """When screener candidate has sector, decide_trades uses it (over analysis fallback)."""
    candidates = [_analysis("INTC", "Technology")]
    cands_by_ticker = {"INTC": {"ticker": "INTC", "sector": "Communication Services"}}
    orders = decide_trades(
        current_positions=[
            {"ticker": "GOOGL", "sector": "Communication Services", "quantity": 10, "current_price": 100, "avg_entry_price": 100, "market_value": 1000, "recommendation": "BUY"},
            {"ticker": "META", "sector": "Communication Services", "quantity": 5, "current_price": 200, "avg_entry_price": 200, "market_value": 1000, "recommendation": "BUY"},
        ],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
        candidates_by_ticker=cands_by_ticker,
    )
    buys = [o for o in orders if o.action == "BUY"]
    # screener said COMM SVCS (cap reached) -> blocked
    assert len(buys) == 0
