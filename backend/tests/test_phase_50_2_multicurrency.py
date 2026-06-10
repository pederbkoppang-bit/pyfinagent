"""phase-50.2: multi-currency accounting -- byte-identity (USD) + EUR correctness.

Offline: FX is mocked so the test is deterministic + needs no network/BQ.
The load-bearing guarantee: every money term is x1.0 for USD/US, so the live
all-USD portfolio is byte-identical to pre-50.2.
"""
from unittest.mock import patch

from backend.services import paper_trader as pt
from backend.services import fx_rates


# ── byte-identity primitives (USD path x1.0) ────────────────────────
def test_fx_local_to_usd_is_identity_for_us():
    assert pt._fx_local_to_usd("US") == 1.0
    assert pt._fx_local_to_usd(None) == 1.0      # legacy positions (market NULL) -> USD
    assert pt._fx_usd_to_local("US") == 1.0
    assert fx_rates.get_fx_rate("USD", "USD") == 1.0


def test_market_value_formula_byte_identical_for_usd():
    """The mark_to_market / pos_row market_value formula is qty*price*fx;
    for a USD position fx==1.0 so it equals the pre-50.2 qty*price exactly."""
    qty, price = 10.0, 123.45
    fx = pt._fx_local_to_usd("US")        # 1.0
    assert qty * price * fx == qty * price          # byte-identical
    cost_basis = 1000.0
    assert (qty * price * fx) - cost_basis == qty * price - cost_basis


def test_share_count_byte_identical_for_usd():
    amount_usd, price = 1000.0, 50.0
    u2l = pt._fx_usd_to_local("US")       # 1.0
    assert (amount_usd * u2l) / price == amount_usd / price   # == 20 shares


# ── EUR correctness (mocked FX) ─────────────────────────────────────
def _fake_fx(frm, to, date=None):
    rates = {("EUR", "USD"): 1.10, ("USD", "EUR"): 1 / 1.10,
             ("USD", "USD"): 1.0}
    if frm == to:
        return 1.0
    return rates.get((frm, to))


def test_eur_market_value_converts_to_usd():
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx):
        assert pt._fx_local_to_usd("EU") == 1.10            # USD per EUR
        qty, price_eur = 5.0, 100.0
        mv_usd = qty * price_eur * pt._fx_local_to_usd("EU")
        assert abs(mv_usd - 550.0) < 1e-9                   # 5*100 EUR -> $550


def test_eur_buy_share_count_uses_usd_to_local():
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx):
        amount_usd, price_eur = 1100.0, 100.0
        u2l = pt._fx_usd_to_local("EU")                     # 1/1.10 EUR per USD
        qty = (amount_usd * u2l) / price_eur
        # $1100 / 1.10 = 1000 EUR budget; /100 EUR price = 10 shares
        assert abs(qty - 10.0) < 1e-9


# ── attribution: local_pnl + fx_pnl == MV_usd - cost_usd (no residual) ──
def test_attribution_usd_has_zero_fx_component():
    qty, pe, pc = 10.0, 50.0, 60.0
    local_pnl, fx_pnl = pt.fx_pnl_attribution(qty, pe, pc, 1.0, 1.0)
    assert fx_pnl == 0.0
    assert local_pnl == round(qty * (pc - pe), 2)           # 100.0


def test_attribution_eur_sums_to_total_usd_pnl():
    qty, pe, pc = 10.0, 100.0, 110.0   # local EUR prices
    fe, fc = 1.10, 1.20                # USD per EUR moved 1.10 -> 1.20
    local_pnl, fx_pnl = pt.fx_pnl_attribution(qty, pe, pc, fe, fc)
    mv_usd = qty * pc * fc             # 10*110*1.20 = 1320
    cost_usd = qty * pe * fe           # 10*100*1.10 = 1100
    # local_pnl + fx_pnl must equal MV_usd - cost_usd exactly (no residual)
    assert abs((local_pnl + fx_pnl) - (mv_usd - cost_usd)) < 0.01
    assert local_pnl == round(qty * (pc - pe) * fe, 2)      # price move at entry FX = 110
    assert fx_pnl == round(qty * pc * (fc - fe), 2)         # FX move = 110


# ── phase-56.1 (findings F-2/55.1-B1/B2): trade ROWS persist USD ─────
# Regression-proof KRW fixtures: on pre-56.1 code the stored trade-row
# total_value (BUY+SELL) and SELL transaction_cost were LOCAL currency
# (KRW magnitudes ~1500x USD) -- these tests FAIL there and PASS post-fix.
# FX mocked (KRW->USD 0.000655 ~= away-week real rate); no network/BQ.
from unittest.mock import MagicMock

_KRW_USD = 0.000655


def _fake_fx_krw(frm, to, date=None):
    rates = {("KRW", "USD"): _KRW_USD, ("USD", "KRW"): 1 / _KRW_USD,
             ("EUR", "USD"): 1.10, ("USD", "EUR"): 1 / 1.10}
    if frm == to:
        return 1.0
    return rates.get((frm, to))


class _FakeFill:
    def __init__(self, price):
        self.fill_price = price
        self.source = "bq_sim"


def _mk_trader(cash=10_000.0, position=None):
    """PaperTrader with a fully-mocked BQ client (no network)."""
    from backend.config.settings import get_settings
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default", "starting_capital": 10_000.0,
        "current_cash": cash, "total_nav": cash, "total_pnl_pct": 0.0,
        "benchmark_return_pct": 0.0, "inception_date": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = [position] if position else []
    bq.get_paper_position.return_value = position
    bq.get_paper_trades_for_ticker_since.return_value = []
    trader = pt.PaperTrader(get_settings(), bq)
    trader._maybe_notify_trade = lambda trade: None
    return trader, bq


def test_krw_buy_row_persists_usd_total_value():
    """F-2 regression (paper_trader.py:265): KR BUY trade-row total_value must
    be USD (~amount_usd), NOT the KRW notional (~1500x larger)."""
    trader, _ = _mk_trader()
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx_krw), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _FakeFill(248_000.0)
        trade = trader.execute_buy(
            ticker="066570.KS", amount_usd=164.0, price=248_000.0, market="KR",
        )
    assert trade is not None
    # magnitude guard: KRW notional would be ~250,000; USD is ~164
    assert trade["total_value"] < 1_000.0, (
        f"total_value={trade['total_value']} looks like LOCAL currency (KRW), not USD"
    )
    assert abs(trade["total_value"] - 164.0) < 2.0          # ~= amount_usd
    # BUY fee was ALREADY USD pre-fix (computed on amount_usd) -- must stay USD
    assert abs(trade["transaction_cost"] - 164.0 * 0.001) < 0.01


def test_krw_sell_row_persists_usd_total_value_and_fee():
    """F-2 regression (paper_trader.py:413-414): KR SELL trade-row total_value
    AND transaction_cost must be USD. Cash credit must NOT double-convert."""
    position = {
        "position_id": "rt-test-1", "ticker": "066570.KS",
        "quantity": 1.468448, "avg_entry_price": 248_000.0,
        "cost_basis": 238.40, "current_price": 248_000.0,
        "market_value": 238.40, "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0,
        "entry_date": "2026-06-09T18:12:39+00:00", "last_analysis_date": "",
        "recommendation": "", "risk_judge_position_pct": 1.0,
        "stop_loss_price": 228_160.0, "market": "KR", "base_currency": "USD",
        "mfe_pct": 0.0, "mae_pct": 0.0,
    }
    trader, bq = _mk_trader(cash=1_000.0, position=position)
    captured_cash = {}
    trader._update_portfolio_cash = lambda c: captured_cash.update(cash=c)
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx_krw), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _FakeFill(248_000.0)
        trade = trader.execute_sell(ticker="066570.KS", price=248_000.0, reason="test")
    assert trade is not None
    sell_value_local = 1.468448 * 248_000.0                  # 364,175.10 KRW
    fee_local = sell_value_local * 0.001                     # 364.18 KRW
    expect_tv_usd = round(sell_value_local * _KRW_USD, 2)    # ~238.53
    expect_fee_usd = round(fee_local * _KRW_USD, 2)          # ~0.24
    assert trade["total_value"] < 1_000.0, (
        f"total_value={trade['total_value']} looks like LOCAL currency (KRW), not USD"
    )
    assert abs(trade["total_value"] - expect_tv_usd) < 0.5
    assert trade["transaction_cost"] < 1.0, (
        f"transaction_cost={trade['transaction_cost']} looks like LOCAL (KRW) fee"
    )
    assert abs(trade["transaction_cost"] - expect_fee_usd) < 0.05
    # four-point consistency: the cash credit path (already correct pre-fix)
    # must remain net_proceeds*fx -- NOT double-converted by the row fix.
    expect_cash = 1_000.0 + (sell_value_local - fee_local) * _KRW_USD
    assert abs(captured_cash["cash"] - expect_cash) < 0.5


def test_us_buy_row_byte_identical():
    """Do-no-harm: the US path's stored row is unchanged by the 56.1 fix
    (fx==1.0 multiply-identity)."""
    trader, _ = _mk_trader()
    with patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _FakeFill(100.0)
        trade = trader.execute_buy(
            ticker="DELL", amount_usd=1_000.0, price=100.0, market="US",
        )
    assert trade is not None
    assert trade["quantity"] == 10.0
    assert trade["total_value"] == 1_000.0                  # exactly qty*price
    assert trade["transaction_cost"] == 1.0                 # 0.1% of 1000
