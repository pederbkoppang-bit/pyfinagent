"""phase-23.2.22: regression guard for the position-cap diagnostic log line.

Pre-fix: when current positions >= paper_max_positions, the buy loop in
portfolio_manager.decide_trades silently `break`d and the cycle reported
"Executing 0 trades" with no log explaining why. This made the
working-as-designed cap look like a silent failure to operators.

This test asserts the diagnostic INFO log line fires with the exact held/max
substrings so future 0-trade cycles are diagnosable from backend.log.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from backend.services.portfolio_manager import decide_trades


def _mk_settings(max_positions: int = 10):
    """Minimal settings stub; only the fields decide_trades reads."""
    return SimpleNamespace(
        paper_max_positions=max_positions,
        paper_max_per_sector=0,            # disable sector cap for this test
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        # Other settings decide_trades may touch (defaults are safe-no-op):
        paper_buy_score_threshold=0.0,
        paper_sell_score_threshold=0.0,
    )


def _mk_position(ticker: str, sector: str = "Technology") -> dict:
    return {
        "ticker": ticker,
        "shares": 1.0,
        "entry_price": 100.0,
        "current_price": 105.0,
        "stop_loss_price": None,
        "recommendation": "HOLD",
        "sector": sector,
    }


def _mk_candidate(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "recommendation": "BUY",
        "score": 9.0,
        "position_pct": 10.0,
        "sector": "Industrials",
    }


def test_position_cap_emits_diagnostic_log_when_full(caplog):
    """At 15 held vs 10 max, the buy loop must log the diagnostic line."""
    settings = _mk_settings(max_positions=10)
    current_positions = [_mk_position(f"T{i:02d}") for i in range(15)]
    candidates = [_mk_candidate(f"NEW{i}") for i in range(3)]
    portfolio_state = {
        "nav": 10000.0,
        "cash": 5000.0,
        "positions_value": 5000.0,
        "position_count": 15,
    }

    with caplog.at_level(logging.INFO, logger="backend.services.portfolio_manager"):
        orders = decide_trades(
            current_positions=current_positions,
            candidate_analyses=candidates,
            holding_analyses=[],
            portfolio_state=portfolio_state,
            settings=settings,
        )

    # Diagnostic log line must be present
    cap_logs = [
        r for r in caplog.records
        if "Position cap reached" in r.getMessage()
    ]
    assert len(cap_logs) == 1, \
        f"expected 1 'Position cap reached' INFO log, got {len(cap_logs)}: {[r.getMessage() for r in caplog.records]}"
    msg = cap_logs[0].getMessage()
    assert "15" in msg, f"log message must include the held count: {msg!r}"
    assert "10" in msg, f"log message must include the max: {msg!r}"

    # No buy orders should have been emitted (the break short-circuits)
    buy_orders = [o for o in orders if o.action == "BUY"]
    assert buy_orders == [], \
        f"expected 0 BUY orders when cap reached, got {len(buy_orders)}"


def test_position_cap_does_not_log_when_room_remains(caplog):
    """Below cap, the diagnostic line must NOT fire."""
    settings = _mk_settings(max_positions=10)
    current_positions = [_mk_position(f"T{i:02d}") for i in range(3)]
    candidates = [_mk_candidate(f"NEW{i}") for i in range(2)]
    portfolio_state = {
        "nav": 10000.0,
        "cash": 5000.0,
        "positions_value": 3000.0,
        "position_count": 3,
    }

    with caplog.at_level(logging.INFO, logger="backend.services.portfolio_manager"):
        decide_trades(
            current_positions=current_positions,
            candidate_analyses=candidates,
            holding_analyses=[],
            portfolio_state=portfolio_state,
            settings=settings,
        )

    cap_logs = [
        r for r in caplog.records
        if "Position cap reached" in r.getMessage()
    ]
    assert cap_logs == [], \
        f"diagnostic line must NOT fire below cap; got {len(cap_logs)}"
