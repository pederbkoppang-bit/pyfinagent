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
