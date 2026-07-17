"""phase-64.4: multi-market fixture-replay e2e (US/KR/EU), pure + no network.

Drives the PURE seam once per market -- screen_universe (yf.download mocked) ->
rank_candidates -> decide_trades -> list[TradeOrder] -- and asserts per-market
funnel counts >0 + currency invariants. Deliberately does NOT drive the full
autonomous loop (its phase-50.4 calendar gate calls datetime.now() and drops
intl tickers on weekends -> flaky funnel=0). Synthetic fixtures (no network).

Criterion-1 note: "EU under the 65.2 thresholds via test flag" is met by passing
lowered min_avg_volume/min_price KWARGS to screen_universe (a test-only override;
65.2 production code does not exist yet, and 64.4's DAG dep is 66.2, not 65.2).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.config.settings import get_settings
from backend.services import fx_rates
from backend.services import paper_trader as pt
from backend.services.portfolio_manager import decide_trades
from backend.tools import screener

US_TICKERS = ["AAPL", "MSFT"]
KR_TICKERS = ["005930.KS", "000660.KS"]
EU_TICKERS = ["SAP.DE", "SIE.DE"]


# ── synthetic yf.download-shaped OHLCV (survives validate_ohlcv R1-R3) ──
def _bars(n=60, start=100.0, vol=1_000_000):
    rows, px = [], start
    for i in range(n):
        px = px * (1.0 + (0.005 if i % 2 == 0 else -0.004))  # gentle drift
        rows.append([px * 0.99, px * 1.01, px * 0.985, px, vol + i])
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def _ohlcv(tickers, vol=1_000_000):
    # group_by="ticker" shape: MultiIndex columns (ticker, field).
    return pd.concat({t: _bars(vol=vol) for t in tickers}, axis=1)


def _screen(tickers, vol=1_000_000, **kwargs):
    with patch.object(screener, "yf") as yf:
        yf.download.return_value = _ohlcv(tickers, vol=vol)
        return screener.screen_universe(tickers, **kwargs)


def _buy_analyses(ranked):
    return [{
        "ticker": r["ticker"],
        "recommendation": "BUY",
        "risk_assessment": {},          # empty -> 10% default sizing, no REJECT
        "final_score": 0.7,
        "price_at_analysis": float(r.get("current_price") or 100.0),
        "sector": "Technology",
    } for r in ranked]


def _order_intent(ranked):
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=_buy_analyses(ranked),
        holding_analyses=[],
        portfolio_state={"nav": 100_000.0, "cash": 100_000.0, "position_count": 0},
        settings=get_settings(),
    )
    return [o for o in orders if o.action == "BUY"]


def _funnel(tickers, vol=1_000_000, **screen_kwargs):
    screened = _screen(tickers, vol=vol, **screen_kwargs)
    ranked = screener.rank_candidates(screened, strategy="momentum")
    orders = _order_intent(ranked)
    return {"universe": len(tickers), "screened": len(screened),
            "ranked": len(ranked), "order_intent": len(orders), "orders": orders}


# ── criterion 1: per-market funnel counts >0 for ALL THREE markets ──
def test_64_4_multi_market_e2e_us_funnel_positive():
    f = _funnel(US_TICKERS)
    assert f["screened"] > 0 and f["ranked"] > 0 and f["order_intent"] > 0
    assert all(o.market == "US" for o in f["orders"])


def test_64_4_multi_market_e2e_kr_funnel_positive():
    f = _funnel(KR_TICKERS)
    assert f["screened"] > 0 and f["ranked"] > 0 and f["order_intent"] > 0
    assert all(o.market == "KR" for o in f["orders"])


def test_64_4_multi_market_e2e_eu_funnel_under_lowered_thresholds():
    """EU tickers with sub-default volume: default thresholds screen 0; the
    lowered '65.2 via test flag' kwargs screen >0 (the flag is load-bearing)."""
    default = _screen(EU_TICKERS, vol=50_000)                       # < 100k default
    assert len(default) == 0, "EU low-volume must fail the default screener"
    f = _funnel(EU_TICKERS, vol=50_000, min_avg_volume=10_000, min_price=1.0)
    assert f["screened"] > 0 and f["ranked"] > 0 and f["order_intent"] > 0
    assert all(o.market == "EU" for o in f["orders"])


def test_64_4_multi_market_e2e_all_three_markets_funnel_positive():
    us = _funnel(US_TICKERS)
    kr = _funnel(KR_TICKERS)
    eu = _funnel(EU_TICKERS, vol=50_000, min_avg_volume=10_000, min_price=1.0)
    for name, f in (("US", us), ("KR", kr), ("EU", eu)):
        assert f["screened"] > 0, f"{name} screened must be >0"
        assert f["ranked"] > 0, f"{name} ranked must be >0"
        assert f["order_intent"] > 0, f"{name} order-intent must be >0"


# ── criterion 2: currency invariants in the same file ──
_KRW_USD, _EUR_USD = 0.000655, 1.08


def _fake_fx(frm, to, date=None):
    table = {("KRW", "USD"): _KRW_USD, ("USD", "KRW"): 1 / _KRW_USD,
             ("EUR", "USD"): _EUR_USD, ("USD", "EUR"): 1 / _EUR_USD}
    return 1.0 if frm == to else table.get((frm, to))


class _Fill:
    def __init__(self, p):
        self.fill_price, self.source = p, "bq_sim"


def _add_on_avg_entry(market, ticker, local_price, local_to_usd, amount_usd):
    s = get_settings().model_copy(update={"paper_avg_entry_fx_fix_enabled": True})
    bq = MagicMock()
    pos = {"ticker": ticker, "quantity": 1.0, "avg_entry_price": local_price,
           "cost_basis": round(local_price * local_to_usd, 2), "current_price": local_price,
           "market_value": round(local_price * local_to_usd, 2), "market": market,
           "base_currency": "USD", "entry_date": "2026-01-01T00:00:00+00:00",
           "position_id": "p1", "sector": "Technology"}
    bq.get_paper_portfolio.return_value = {"portfolio_id": "d", "starting_capital": 10000.0,
        "current_cash": 10000.0, "total_nav": 10000.0, "total_pnl_pct": 0.0,
        "benchmark_return_pct": 0.0, "inception_date": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00"}
    bq.get_paper_positions.return_value = [pos]
    bq.get_paper_position.return_value = pos
    bq.get_paper_trades_for_ticker_since.return_value = []
    trader = pt.PaperTrader(s, bq)
    trader._maybe_notify_trade = lambda trade: None
    cap = {}
    bq.save_paper_position.side_effect = lambda row: cap.update(row=row)
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _Fill(local_price)
        trader.execute_buy(ticker=ticker, amount_usd=amount_usd, price=local_price, market=market)
    return cap["row"]["avg_entry_price"]


def test_64_4_multi_market_e2e_currency_invariants():
    kr_avg = _add_on_avg_entry("KR", "005930.KS", 70000.0, _KRW_USD, amount_usd=45.85)
    assert abs(kr_avg - 70000.0) < 500.0, f"KR avg_entry must stay KRW-scale, got {kr_avg}"
    eu_avg = _add_on_avg_entry("EU", "SAP.DE", 150.0, _EUR_USD, amount_usd=162.0)
    assert abs(eu_avg - 150.0) < 2.0, f"EU avg_entry must stay EUR-scale, got {eu_avg}"


# ── criterion 3: requires_live variant, excluded from default runs ──
@pytest.mark.requires_live
def test_64_4_multi_market_e2e_live_smoke():
    """Hits REAL yfinance (excluded by `-m 'not requires_live'`)."""
    out = screener.screen_universe(["AAPL"], period="1mo")
    assert isinstance(out, list)
