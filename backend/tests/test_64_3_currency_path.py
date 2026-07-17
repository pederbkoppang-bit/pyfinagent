"""phase-64.3: currency/money-path gap tests (pure; no live net/BQ).

Criterion 3: add-on-buy avg_entry must stay in the LOCAL currency (KR -> KRW
scale, EU -> EUR scale), not the legacy USD/local mix. The fix is
phase-70.3 (`paper_avg_entry_fx_fix_enabled`, default False); 61.3 was
display-only, so we assert 70.3-fix behavior in the SHAPE of the 61.3
criteria. Mirrors the proven KR harness (test_phase_70_3_atomic_swap.py:192)
and ADDS the EU (.DE) + US-byte-identical + fx-unavailable cases.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.config.settings import get_settings
from backend.services import fx_rates
from backend.services import paper_trader as pt

# realistic rates (local -> USD)
_KRW_USD = 0.000655
_EUR_USD = 1.08


def _fake_fx(frm, to, date=None):
    table = {
        ("KRW", "USD"): _KRW_USD, ("USD", "KRW"): 1 / _KRW_USD,
        ("EUR", "USD"): _EUR_USD, ("USD", "EUR"): 1 / _EUR_USD,
    }
    return 1.0 if frm == to else table.get((frm, to))


class _Fill:
    def __init__(self, p):
        self.fill_price, self.source = p, "bq_sim"


def _trader(*, market, ticker, local_price, local_to_usd, fix_on):
    """A PaperTrader holding 1 existing LOCAL-priced lot, with a mocked bq that
    captures the saved position row on the add-on buy."""
    s = get_settings().model_copy(update={"paper_avg_entry_fx_fix_enabled": fix_on})
    bq = MagicMock()
    pos = {
        "ticker": ticker, "quantity": 1.0, "avg_entry_price": local_price,  # LOCAL
        "cost_basis": round(1.0 * local_price * local_to_usd, 2),
        "current_price": local_price,
        "market_value": round(local_price * local_to_usd, 2),
        "market": market, "base_currency": "USD",
        "entry_date": "2026-01-01T00:00:00+00:00", "position_id": "p1",
        "sector": "Technology",
    }
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "d", "starting_capital": 10000.0, "current_cash": 10000.0,
        "total_nav": 10000.0, "total_pnl_pct": 0.0, "benchmark_return_pct": 0.0,
        "inception_date": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = [pos]
    bq.get_paper_position.return_value = pos
    bq.get_paper_trades_for_ticker_since.return_value = []
    trader = pt.PaperTrader(s, bq)
    trader._maybe_notify_trade = lambda trade: None
    captured = {}
    bq.save_paper_position.side_effect = lambda row: captured.update(row=row)
    return trader, captured


def _run_add_on(trader, ticker, amount_usd, price, market):
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _Fill(price)
        trader.execute_buy(ticker=ticker, amount_usd=amount_usd, price=price, market=market)


def test_64_3_currency_path_kr_avg_entry_stays_krw():
    """KR add-on: ON avg_entry stays KRW-scale (~70000); OFF mixes USD -> tiny."""
    t_on, cap_on = _trader(market="KR", ticker="005930.KS", local_price=70000.0,
                           local_to_usd=_KRW_USD, fix_on=True)
    _run_add_on(t_on, "005930.KS", amount_usd=45.85, price=70000.0, market="KR")
    avg_on = cap_on["row"]["avg_entry_price"]
    assert abs(avg_on - 70000.0) < 500.0, f"ON avg_entry should be KRW ~70000, got {avg_on}"

    t_off, cap_off = _trader(market="KR", ticker="005930.KS", local_price=70000.0,
                             local_to_usd=_KRW_USD, fix_on=False)
    _run_add_on(t_off, "005930.KS", amount_usd=45.85, price=70000.0, market="KR")
    avg_off = cap_off["row"]["avg_entry_price"]
    assert avg_off < 1000.0, f"OFF (legacy) mixes USD/local -> tiny avg, got {avg_off}"


def test_64_3_currency_path_eu_avg_entry_stays_eur():
    """EU (.DE) add-on: ON avg_entry stays EUR-scale (~150); OFF is USD-inflated."""
    # 150 EUR at 1.08 USD/EUR = 162 USD buys 1 share.
    t_on, cap_on = _trader(market="EU", ticker="SAP.DE", local_price=150.0,
                           local_to_usd=_EUR_USD, fix_on=True)
    _run_add_on(t_on, "SAP.DE", amount_usd=162.0, price=150.0, market="EU")
    avg_on = cap_on["row"]["avg_entry_price"]
    assert abs(avg_on - 150.0) < 2.0, f"ON avg_entry should be EUR ~150, got {avg_on}"

    t_off, cap_off = _trader(market="EU", ticker="SAP.DE", local_price=150.0,
                             local_to_usd=_EUR_USD, fix_on=False)
    _run_add_on(t_off, "SAP.DE", amount_usd=162.0, price=150.0, market="EU")
    avg_off = cap_off["row"]["avg_entry_price"]
    # legacy = new_cost(USD ~324) / new_qty(2) ~= 162 USD -> inflated above the 150 EUR scale.
    assert avg_off > 155.0, f"OFF (legacy) should be USD-inflated (~162), got {avg_off}"
    assert avg_off > avg_on, "OFF avg must exceed the correct EUR-scale ON avg"


def test_64_3_currency_path_us_byte_identical():
    """US: fx=1 so ON and OFF produce the same avg (byte-identical fast path)."""
    t_on, cap_on = _trader(market="US", ticker="AAPL", local_price=200.0,
                           local_to_usd=1.0, fix_on=True)
    _run_add_on(t_on, "AAPL", amount_usd=200.0, price=200.0, market="US")
    t_off, cap_off = _trader(market="US", ticker="AAPL", local_price=200.0,
                             local_to_usd=1.0, fix_on=False)
    _run_add_on(t_off, "AAPL", amount_usd=200.0, price=200.0, market="US")
    assert abs(cap_on["row"]["avg_entry_price"] - cap_off["row"]["avg_entry_price"]) < 1e-6


def test_64_3_currency_path_fx_unavailable_skips_buy():
    """FX unavailable for a non-USD market -> execute_buy returns None (never
    silently treats LOCAL as USD)."""
    t, cap = _trader(market="KR", ticker="005930.KS", local_price=70000.0,
                     local_to_usd=_KRW_USD, fix_on=True)
    with patch.object(fx_rates, "get_fx_rate", return_value=None), \
         patch.object(pt, "ExecutionRouter"):
        out = t.execute_buy(ticker="005930.KS", amount_usd=45.85, price=70000.0, market="KR")
    assert out is None
    assert "row" not in cap, "no position should be saved when fx is unavailable"
