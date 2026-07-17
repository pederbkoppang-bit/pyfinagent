"""phase-70.3 (P1, S3 + money-path): atomic cross-sector swap + non-US avg-entry fix.

Deterministic (network-free) proofs of the four immutable criteria:
  1. atomic swap -- a SELL can never persist while its paired BUY drops (net -1),
     the swap BUY is cash-bounded + honors the $50 floor.
  2. cross-sector rotation is flag-gated + fail-safe; OFF -> same-sector-only.
  3. add-on avg_entry_price is LOCAL-consistent for non-US tickers when the fix flag
     is ON (byte-identical for US).
  4. every lever default-OFF -> byte-identical; no risk threshold moved.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.services import paper_trader as pt
from backend.services import fx_rates
from backend.services import portfolio_manager as pm
from backend.services import autonomous_loop as al


# ─────────────────────────────── criterion 1: atomic swap execution ───────────
class _FakeTrader:
    """Minimal trader for _execute_swap_pair: records positions + calls."""
    def __init__(self):
        self.positions = {"WEAK": {"ticker": "WEAK", "market": "US", "market_value": 1000.0,
                                   "quantity": 10.0, "avg_entry_price": 100.0}}
        self.bq = MagicMock()
        self.buy_returns = {"trade_id": "b1"}   # override to None to simulate a BUY drop
        self.sell_returns = {"trade_id": "s1"}
        self.calls = []

    def get_position(self, t):
        return self.positions.get(t)

    def execute_buy(self, **kw):
        self.calls.append(("buy", kw.get("ticker")))
        return self.buy_returns

    def execute_sell(self, **kw):
        self.calls.append(("sell", kw.get("ticker")))
        if self.sell_returns:
            self.positions.pop(kw.get("ticker"), None)
        return self.sell_returns


def _orders():
    sell = pm.TradeOrder(ticker="WEAK", action="SELL", reason="swap_for_higher_conviction",
                         price=100.0, swap_group_id="g1")
    buy = pm.TradeOrder(ticker="STRONG", action="BUY", amount_usd=900.0, reason="swap_buy",
                        price=50.0, price_at_analysis=50.0, market="US", swap_group_id="g1")
    return sell, buy


def test_atomic_swap_buy_drops_does_not_sell():
    """RED->GREEN: if the paired BUY drops, the SELL is NEVER executed -> book unchanged
    (pre-fix flat SELL-then-BUY would have sold then dropped the buy = net -1)."""
    trader = _FakeTrader()
    trader.buy_returns = None                      # BUY drops
    sell, buy = _orders()
    with patch.object(pt, "_get_live_price", return_value=50.0):
        res = al._execute_swap_pair(trader, sell, buy)
    assert res["sold"] is False and res["bought"] is False
    assert "WEAK" in trader.positions              # the SELL leg still held (not sold)
    assert ("sell", "WEAK") not in trader.calls    # SELL was never attempted


def test_atomic_swap_contrast_old_flat_path_would_lose_position():
    """Documents the bug the new helper fixes: the OLD flat path (SELL then BUY-drops)
    removes the position (count -1); the NEW helper leaves it intact."""
    trader = _FakeTrader()
    # simulate OLD flat path: sell first...
    trader.execute_sell(ticker="WEAK")
    assert "WEAK" not in trader.positions          # RED: position gone
    # ...then the BUY drops (nothing restores it) -> net -1. The new helper (above) avoids this.


def test_atomic_swap_happy_path_both_legs():
    trader = _FakeTrader()
    sell, buy = _orders()
    with patch.object(pt, "_get_live_price", return_value=50.0):
        res = al._execute_swap_pair(trader, sell, buy)
    assert res["sold"] is True and res["bought"] is True
    assert "WEAK" not in trader.positions          # sold
    assert trader.calls[0][0] == "buy" and trader.calls[1][0] == "sell"   # BUY-first


def test_atomic_swap_sell_infeasible_drops_both():
    trader = _FakeTrader()
    trader.positions = {}                          # SELL leg has no position
    sell, buy = _orders()
    with patch.object(pt, "_get_live_price", return_value=50.0):
        res = al._execute_swap_pair(trader, sell, buy)
    assert res["sold"] is False and res["bought"] is False
    assert trader.calls == []                      # BUY never attempted


# ─────────────────────── criterion 1: cash-bound + $50 floor + group_id ───────
def _swap_settings(**kw):
    base = dict(paper_swap_enabled=True, paper_swap_min_delta_pct=25.0, paper_swap_max_per_cycle=2,
                paper_swap_churn_fix_enabled=True, paper_atomic_swap_enabled=False,
                paper_cross_sector_rotation_enabled=False, paper_max_per_sector=2,
                paper_max_per_sector_nav_pct=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _swap_inputs(cand_pct=10.0):
    cand = {"ticker": "STRONG", "sector": "Technology", "final_score": 9.0,
            "position_pct": cand_pct, "recommendation": "BUY"}
    holding = {"ticker": "WEAK", "sector": "Technology", "final_score": 5.0}
    positions = [{"ticker": "WEAK", "sector": "Technology", "market_value": 1000.0,
                  "current_price": 100.0}]
    return [cand], positions, {"WEAK": holding}


def test_swap_off_no_group_id_and_legacy_sizing():
    cand, positions, hl = _swap_inputs()
    orders = pm._compute_swap_candidates(
        sector_blocked=cand, current_positions=positions, holding_lookup=hl,
        sector_counts={"Technology": 2}, sector_market_values={"Technology": 1000.0},
        selling_tickers=set(), settings=_swap_settings(paper_atomic_swap_enabled=False),
        nav=10000.0, available_cash=200.0,
    )
    buy = next(o for o in orders if o.action == "BUY")
    assert buy.swap_group_id is None               # OFF -> untagged (byte-identical)
    assert buy.amount_usd == 1000.0                # legacy nav*pct = 10000*0.10, NOT cash-bounded


def test_swap_atomic_cash_bounded_and_grouped():
    cand, positions, hl = _swap_inputs()
    orders = pm._compute_swap_candidates(
        sector_blocked=cand, current_positions=positions, holding_lookup=hl,
        sector_counts={"Technology": 2}, sector_market_values={"Technology": 1000.0},
        selling_tickers=set(), settings=_swap_settings(paper_atomic_swap_enabled=True),
        nav=10000.0, available_cash=200.0,
    )
    sell = next(o for o in orders if o.action == "SELL")
    buy = next(o for o in orders if o.action == "BUY")
    # cash-bounded: min(nav*pct=1000, available 200 + freed 1000 = 1200) = 1000
    assert buy.amount_usd == 1000.0
    # both legs share one swap_group_id
    assert sell.swap_group_id and sell.swap_group_id == buy.swap_group_id


def test_swap_atomic_50_floor_drops_pair():
    cand, positions, hl = _swap_inputs(cand_pct=0.4)   # 10000*0.004 = $40 < $50 floor
    orders = pm._compute_swap_candidates(
        sector_blocked=cand, current_positions=positions, holding_lookup=hl,
        sector_counts={"Technology": 2}, sector_market_values={"Technology": 1000.0},
        selling_tickers=set(), settings=_swap_settings(paper_atomic_swap_enabled=True),
        nav=10000.0, available_cash=0.0,   # freed=1000 but nav*pct=40 -> min=40 < 50
    )
    assert orders == []                            # whole pair dropped, no lone SELL


# ─────────────────────── criterion 2: cross-sector rotation gate ──────────────
def test_cross_rotation_safe_blocks_count_cap_breach():
    # cand_sector already at cap (2) -> +1 would breach -> blocked (fail-safe)
    assert pm._cross_rotation_safe(
        cand_sector="Technology", weakest={"sector": "Energy", "market_value": 1000.0},
        sector_counts={"Technology": 2}, sector_market_values={"Technology": 5000.0, "Energy": 1000.0},
        buy_amount=1000.0, nav=10000.0, settings=_swap_settings(paper_max_per_sector=2),
    ) is False


def test_cross_rotation_safe_allows_hhi_drop_within_caps():
    # move value FROM the concentrated Technology TO under-weight Health -> HHI drops,
    # destination (Health) count 0 -> +1 within cap -> allowed
    ok = pm._cross_rotation_safe(
        cand_sector="Health", weakest={"sector": "Technology", "market_value": 3000.0},
        sector_counts={"Technology": 3, "Health": 0},
        sector_market_values={"Technology": 8000.0, "Health": 0.0, "Energy": 2000.0},
        buy_amount=3000.0, nav=10000.0, settings=_swap_settings(paper_max_per_sector=5),
    )
    assert ok is True


# ─────────────────────── criterion 3: non-US avg_entry FX fix ─────────────────
_KRW_USD = 0.000655


def _fake_fx_krw(frm, to, date=None):
    r = {("KRW", "USD"): _KRW_USD, ("USD", "KRW"): 1 / _KRW_USD}
    return 1.0 if frm == to else r.get((frm, to))


class _Fill:
    def __init__(self, p): self.fill_price, self.source = p, "bq_sim"


def _kr_trader(fix_on: bool):
    from backend.config.settings import get_settings
    s = get_settings().model_copy(update={"paper_avg_entry_fx_fix_enabled": fix_on})
    bq = MagicMock()
    pos = {"ticker": "005930.KS", "quantity": 1.0, "avg_entry_price": 70000.0,  # LOCAL (KRW)
           "cost_basis": round(1.0 * 70000.0 * _KRW_USD, 2), "current_price": 70000.0,
           "market_value": round(70000.0 * _KRW_USD, 2), "market": "KR", "base_currency": "USD",
           "entry_date": "2026-01-01T00:00:00+00:00", "position_id": "kr1", "sector": "Technology"}
    bq.get_paper_portfolio.return_value = {"portfolio_id": "d", "starting_capital": 10000.0,
        "current_cash": 10000.0, "total_nav": 10000.0, "total_pnl_pct": 0.0,
        "benchmark_return_pct": 0.0, "inception_date": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00"}
    bq.get_paper_positions.return_value = [pos]
    bq.get_paper_position.return_value = pos
    bq.get_paper_trades_for_ticker_since.return_value = []
    trader = pt.PaperTrader(s, bq)
    trader._maybe_notify_trade = lambda trade: None
    captured = {}
    bq.save_paper_position.side_effect = lambda row: captured.update(row=row)
    return trader, captured


def test_avg_entry_fx_fix_local_consistent_for_kr():
    """ON: add-on avg_entry stays LOCAL (KRW ~70000). OFF: legacy USD-mix (~a few USD)."""
    trader, cap = _kr_trader(fix_on=True)
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx_krw), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _Fill(70000.0)
        trader.execute_buy(ticker="005930.KS", amount_usd=45.85, price=70000.0, market="KR")
    avg = cap["row"]["avg_entry_price"]
    assert abs(avg - 70000.0) < 500.0, f"ON avg_entry should be LOCAL ~70000, got {avg}"

    trader2, cap2 = _kr_trader(fix_on=False)
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx_krw), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _Fill(70000.0)
        trader2.execute_buy(ticker="005930.KS", amount_usd=45.85, price=70000.0, market="KR")
    avg_legacy = cap2["row"]["avg_entry_price"]
    assert avg_legacy < 1000.0, f"OFF (legacy) mixes USD/local -> tiny avg, got {avg_legacy}"


# ─────────────────────── criterion 4: flags default OFF ───────────────────────
def test_flags_present_and_default_off():
    from backend.config.settings import Settings
    for f in ("paper_atomic_swap_enabled", "paper_cross_sector_rotation_enabled",
              "paper_avg_entry_fx_fix_enabled"):
        assert f in Settings.model_fields
        assert Settings.model_fields[f].default is False
    assert "reserved_cash" in __import__("inspect").signature(pt.PaperTrader.execute_buy).parameters
