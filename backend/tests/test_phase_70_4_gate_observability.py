"""phase-70.4 (P2, S3): surface + reconcile the silent BUY-gates.

Deterministic (network-free) proofs:
  1. session-budget breach is LOGGED + SURFACED (never silent); the effective ceiling
     is reconcilable to the daily cap (flag-gated).
  2. price-tolerance rejections are accumulated with ticker + drift (already logged +
     tunable on HEAD).
  3. a lite parse-fail is COUNTED as degraded (does not masquerade as a score-5 HOLD).
  4. flags default-OFF; observability is always-on.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services import autonomous_loop as al
from backend.services import paper_trader as pt
from backend.services import fx_rates


# ─────────────── criterion 3: parse-fail counted as degraded (G3-B) ───────────
def test_degraded_check_counts_parse_failed():
    real_hold = {"recommendation": "HOLD", "final_score": 5, "confidence": 7}   # genuine neutral
    parse_fail = {"recommendation": "HOLD", "final_score": 5, "_parse_failed": True}  # score-5 mask
    fire, n_deg, n_tot = al._degraded_scoring_check([real_hold, parse_fail, dict(parse_fail), dict(parse_fail)])
    # the 3 parse-fails are all counted degraded (>=3 -> fire); the real HOLD is not
    assert n_deg == 3 and n_tot == 4 and fire is True


def test_degraded_check_ignores_genuine_hold():
    fire, n_deg, _ = al._degraded_scoring_check([{"recommendation": "HOLD", "final_score": 5, "confidence": 7}])
    assert n_deg == 0 and fire is False


# ─────────────── criterion 1: session-budget breach logged + effective ceiling ──
def test_session_budget_breach_logs_and_raises(caplog):
    from backend.agents.llm_client import BudgetBreachError
    orig_cost, orig_budget = al._session_cost, al._effective_session_budget
    try:
        al._session_cost = 1.50
        al._effective_session_budget = 1.00
        import logging
        caplog.set_level(logging.WARNING, logger="backend.services.autonomous_loop")
        with pytest.raises(BudgetBreachError):
            al._check_session_budget("pre_analysis_test")
        assert any("SESSION BUDGET BREACH" in r.getMessage() for r in caplog.records), (
            "the breach must be LOGGED before raising (never silent)"
        )
    finally:
        al._session_cost, al._effective_session_budget = orig_cost, orig_budget


def test_session_budget_below_ceiling_no_raise():
    orig_cost, orig_budget = al._session_cost, al._effective_session_budget
    try:
        al._session_cost = 0.50
        al._effective_session_budget = 2.00   # reconciled to the $2 daily cap
        al._check_session_budget("pre_analysis_test")   # must NOT raise
    finally:
        al._session_cost, al._effective_session_budget = orig_cost, orig_budget


# ─────────────── criterion 2: price-tolerance rejections accumulated (G2-A) ────
def _mk_trader():
    from backend.config.settings import get_settings
    bq = MagicMock()
    trader = pt.PaperTrader(get_settings(), bq)
    trader._maybe_notify_trade = lambda t: None
    return trader


def test_price_tolerance_rejection_is_accumulated():
    trader = _mk_trader()
    # ensure the gate is active (default 5.0); live price diverges 10% from analysis
    with patch.object(trader.settings, "paper_price_tolerance_pct", 5.0):
        out = trader.execute_buy(ticker="AMD", amount_usd=1000.0, price=110.0,
                                 price_at_analysis=100.0, market="US")
    assert out is None                                   # rejected
    assert len(trader.buy_rejections) == 1
    rej = trader.buy_rejections[0]
    assert rej["ticker"] == "AMD" and rej["reason"] == "price_tolerance"
    assert abs(rej["divergence_pct"] - 10.0) < 0.01 and rej["tolerance_pct"] == 5.0


def test_price_tolerance_accumulator_empty_when_within_tolerance():
    trader = _mk_trader()
    # 1% divergence < 5% tolerance -> not rejected here (proceeds past the gate).
    # get_or_create_portfolio is mocked (MagicMock) so execution may no-op, but the
    # gate must NOT have logged a rejection.
    with patch.object(fx_rates, "get_fx_rate", return_value=1.0):
        try:
            trader.execute_buy(ticker="AMD", amount_usd=100.0, price=101.0,
                               price_at_analysis=100.0, market="US")
        except Exception:
            pass
    assert trader.buy_rejections == []                   # no price-tolerance rejection


# ─────────────── criterion 4: flag default-OFF ────────────────────────────────
def test_flag_present_and_default_off():
    from backend.config.settings import Settings
    assert "paper_session_budget_reconcile_enabled" in Settings.model_fields
    assert Settings.model_fields["paper_session_budget_reconcile_enabled"].default is False
