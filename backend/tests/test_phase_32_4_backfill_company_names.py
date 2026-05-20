"""phase-32.4 tests: backfill_missing_company_names helper.

Audit basis: dashboard observation 2026-05-20 (9 of 11 paper_positions rows
show ticker-as-company-name).
Spec source: .claude/masterplan.json::phase-32.4.implementation_plan.test_specs.

Cosmetic helper -- MUST NOT affect trading decisions. Fail-open ALWAYS:
a yfinance error must not propagate. Idempotent: re-running when all names
are real returns 0 backfilled.

Test plan (6 cases):
  1. test_backfill_skips_real_name: position with company_name='Intel
     Corporation' is NOT mutated.
  2. test_backfill_fires_when_name_equals_ticker: position with
     company_name='MU' triggers a yfinance lookup, persists the resolved
     name.
  3. test_backfill_fires_when_name_empty: company_name='' or None is
     treated as needing-backfill.
  4. test_backfill_idempotent: run twice; second call returns 0 backfilled.
  5. test_fail_open_on_yfinance_error: yfinance Ticker.info raises ->
     position untouched, WARNING logged, no exception propagated.
  6. test_yfinance_returns_ticker_skips: yfinance returns the ticker as
     shortName (e.g., for an unknown ticker) -> we do NOT persist the
     sentinel; row stays unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.paper_trader import PaperTrader


def _mock_settings() -> SimpleNamespace:
    return SimpleNamespace(
        paper_price_tolerance_pct=5.0,
        paper_default_stop_loss_pct=8.0,
        paper_trailing_stop_pct=8.0,
        paper_max_positions=10,
        paper_transaction_cost_pct=0.05,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_trailing_dd_limit_pct=10.0,
        paper_daily_loss_limit_pct=4.0,
    )


def _trader_with_positions(positions: list[dict]) -> PaperTrader:
    bq = MagicMock()
    bq.get_paper_positions.return_value = positions
    bq.save_paper_position.return_value = None
    return PaperTrader(settings=_mock_settings(), bq_client=bq)


def _fake_yf(name_by_ticker: dict[str, str]):
    """Build a mock yfinance module where Ticker(t).info returns the mapped
    {shortName: name} per ticker. Tickers absent from the map raise."""
    def _factory(ticker):
        mod = MagicMock()
        if ticker in name_by_ticker:
            mod.info = {"shortName": name_by_ticker[ticker], "longName": name_by_ticker[ticker]}
        else:
            raise RuntimeError(f"no data for {ticker}")
        return mod
    mod = MagicMock()
    mod.Ticker = MagicMock(side_effect=_factory)
    return mod


# ── 1. Skips real name ────────────────────────────────────────────


def test_backfill_skips_real_name():
    positions = [
        {"ticker": "INTC", "company_name": "Intel Corporation"},
        {"ticker": "SNDK", "company_name": "Sandisk Corporation"},
    ]
    trader = _trader_with_positions(positions)
    with patch.dict("sys.modules", {"yfinance": _fake_yf({})}):
        result = trader.backfill_missing_company_names()
    assert result["count_backfilled"] == 0
    assert result["count_skipped"] == 2
    trader.bq.save_paper_position.assert_not_called()


# ── 2. Fires when company_name equals ticker ──────────────────────


def test_backfill_fires_when_name_equals_ticker():
    positions = [{"ticker": "MU", "company_name": "MU"}]
    trader = _trader_with_positions(positions)
    fake_yf = _fake_yf({"MU": "Micron Technology"})
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        result = trader.backfill_missing_company_names()
    assert result["count_backfilled"] == 1
    assert result["count_skipped"] == 0
    saved = trader.bq.save_paper_position.call_args[0][0]
    assert saved["company_name"] == "Micron Technology"
    assert saved["ticker"] == "MU"


# ── 3. Fires when name empty or None ──────────────────────────────


def test_backfill_fires_when_name_empty():
    positions = [
        {"ticker": "WDC", "company_name": ""},
        {"ticker": "LITE", "company_name": None},
    ]
    trader = _trader_with_positions(positions)
    fake_yf = _fake_yf({"WDC": "Western Digital", "LITE": "Lumentum Holdings"})
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        result = trader.backfill_missing_company_names()
    assert result["count_backfilled"] == 2
    saved_calls = [c[0][0] for c in trader.bq.save_paper_position.call_args_list]
    saved_by_ticker = {c["ticker"]: c["company_name"] for c in saved_calls}
    assert saved_by_ticker == {"WDC": "Western Digital", "LITE": "Lumentum Holdings"}


# ── 4. Idempotent ─────────────────────────────────────────────────


def test_backfill_idempotent_on_real_names():
    """Once positions have real names, a second backfill returns 0."""
    positions = [
        {"ticker": "MU", "company_name": "Micron Technology"},
        {"ticker": "INTC", "company_name": "Intel Corporation"},
    ]
    trader = _trader_with_positions(positions)
    with patch.dict("sys.modules", {"yfinance": _fake_yf({})}):
        result = trader.backfill_missing_company_names()
    assert result["count_backfilled"] == 0
    assert result["count_skipped"] == 2


# ── 5. Fail-open on yfinance error ────────────────────────────────


def test_fail_open_on_yfinance_error():
    positions = [
        {"ticker": "MU", "company_name": "MU"},
        {"ticker": "INTC", "company_name": "Intel Corporation"},  # real, skipped
    ]
    trader = _trader_with_positions(positions)
    fake_yf = MagicMock()
    fake_yf.Ticker = MagicMock(side_effect=RuntimeError("rate limit"))
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        # Must NOT raise
        result = trader.backfill_missing_company_names()
    # MU goes into skipped because yfinance failed; INTC was already real
    assert result["count_backfilled"] == 0
    assert "MU" in result["skipped"]
    trader.bq.save_paper_position.assert_not_called()


# ── 6. yfinance returns the ticker -> skip (don't persist sentinel) ──


def test_yfinance_returns_ticker_skips():
    """If yfinance returns the same ticker as shortName/longName (e.g., for
    an unknown / OTC ticker), we must NOT persist that as the company_name --
    it's the same sentinel we're trying to escape."""
    positions = [{"ticker": "OTCXYZ", "company_name": "OTCXYZ"}]
    trader = _trader_with_positions(positions)
    fake_yf = _fake_yf({"OTCXYZ": "OTCXYZ"})  # yfinance returns ticker
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        result = trader.backfill_missing_company_names()
    assert result["count_backfilled"] == 0
    assert "OTCXYZ" in result["skipped"]
    trader.bq.save_paper_position.assert_not_called()
