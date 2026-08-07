"""phase-61.3 (P0, money-display + currency correctness): the STOP half of criterion 1.

The immutable criterion reads: "add-on BUYs average avg_entry_price in LOCAL currency;
a regression test performs a KR add-on buy and asserts avg_entry_price **and the
breakeven-advanced stop** both remain KRW-scale".

The averaging formula shipped with phase-70.3 behind `paper_avg_entry_fx_fix_enabled`
(paper_trader.py:459-467) and three test files already assert the saved row's
`avg_entry_price`. What nothing in the repo asserted before this file is the second
clause: that the stop `_advance_stop` derives FROM that entry is also KRW-scale. The two
files that call `_advance_stop` (test_phase_32_1_breakeven_ratchet.py,
test_phase_32_2_hwm_trailing.py) drive US positions only and are deselected by this
step's immutable `-k 'addon or avg_entry or currency or 61_3'` filter.

That second clause is the money-safety one. Under the legacy formula a KR add-on stores a
USD-per-share number (~46) in a KRW-scale field, the breakeven ratchet copies it into
`stop_loss_price`, and `check_stop_losses` then tests `70000 <= 46` -- which never fires.
The position's downside protection is silently deleted. These tests fail on the legacy
path and pass on the fixed one.

Network-free: FX and the ExecutionRouter are patched; the live settings object is never
mutated (each trader gets a `model_copy`), so the production flag state is irrelevant.

Filename matches the immutable -k filter three ways: `addon`, `currency`, `61_3`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.config.settings import get_settings
from backend.services import fx_rates
from backend.services import paper_trader as pt

# realistic local -> USD rates
_KRW_USD = 0.000655
_EUR_USD = 1.08

_KR_TICKER = "005930.KS"
_EU_TICKER = "SAP.DE"


def _fake_fx(frm, to, date=None):
    table = {
        ("KRW", "USD"): _KRW_USD, ("USD", "KRW"): 1 / _KRW_USD,
        ("EUR", "USD"): _EUR_USD, ("USD", "EUR"): 1 / _EUR_USD,
    }
    return 1.0 if frm == to else table.get((frm, to))


class _Fill:
    def __init__(self, p):
        self.fill_price, self.source = p, "bq_sim"


class _UnpausedKillSwitch:
    """A hermetic kill-switch state for the BUY gate (paper_trader.py:196).

    NOT a bypass of a safety control -- it is the documented construction-site
    injection seam (paper_trader.py:100-120), and using it is what makes these tests
    deterministic. `kill_switch._state` is a module singleton whose `_load_from_audit`
    replays real `pause` rows, so an uninjected PaperTrader test passes or fails
    depending on whether the LIVE book happens to be paused that day. It is paused
    right now (`pause_reason='manual'`), which is exactly why six pre-existing tests in
    this suite are red -- a known, separately-queued defect (masterplan 36.28), not
    anything these tests should inherit.

    Baselines are present and positive so `baselines_present_in` (kill_switch.py:856)
    is satisfied through its real definition rather than a hand-copied predicate.
    """

    def is_paused(self) -> bool:
        return False

    def snapshot(self) -> dict:
        return {"paused": False, "pause_reason": None, "sod_nav": 10000.0,
                "peak_nav": 10000.0, "sod_date": "2026-01-01"}


def _trader(*, market, ticker, local_price, local_to_usd, fix_on, stop_loss_price=None):
    """A PaperTrader holding one existing LOCAL-priced lot.

    Mirrors the proven harness at test_phase_70_3_atomic_swap.py:192-212. The existing
    position deliberately carries `stop_advanced_at_R: None` and a stop BELOW entry so
    the breakeven branch at paper_trader.py:1449-1452 is reachable -- see
    test_fixture_reaches_breakeven_branch for the guard that keeps this honest.
    """
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
        # breakeven preconditions (paper_trader.py:1449-1452)
        "stop_advanced_at_R": None,
        "stop_loss_price": stop_loss_price if stop_loss_price is not None else local_price * 0.92,
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
    trader = pt.PaperTrader(s, bq, kill_switch_state=_UnpausedKillSwitch())
    trader._maybe_notify_trade = lambda trade: None
    captured: dict = {}
    bq.save_paper_position.side_effect = lambda row: captured.update(row=row)
    return trader, captured, pos


def _addon_buy(trader, captured, *, ticker, price, amount_usd, market):
    """Run one add-on BUY and return the saved row's avg_entry_price."""
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx), \
         patch.object(pt, "ExecutionRouter") as router:
        router.return_value.submit_order.return_value = _Fill(price)
        trader.execute_buy(ticker=ticker, amount_usd=amount_usd, price=price, market=market)
    assert "row" in captured, "the add-on BUY did not persist a position row"
    return captured["row"]["avg_entry_price"]


def _advance(trader, base_pos, avg_entry):
    """Drive the breakeven ratchet on a position carrying `avg_entry`.

    new_mfe is set well above `paper_default_stop_loss_pct` so the breakeven branch
    fires. The stop is kept strictly below entry (in the SAME scale as avg_entry) so the
    `current_stop_f >= entry_price` early-return cannot swallow the call.
    """
    pos = dict(base_pos)
    pos["avg_entry_price"] = avg_entry
    pos["stop_loss_price"] = avg_entry * 0.92
    pos["stop_advanced_at_R"] = None
    threshold = float(getattr(trader.settings, "paper_default_stop_loss_pct", 8.0))
    return trader._advance_stop(pos, new_mfe=threshold + 10.0)


# ─────────── criterion 1, clause 2: the KRW-scale breakeven stop ───────────

def test_61_3_kr_addon_currency_avg_entry_and_breakeven_stop_stay_krw_scale():
    """THE criterion: after a KR add-on, BOTH avg_entry AND the advanced stop are KRW."""
    trader, cap, base = _trader(
        market="KR", ticker=_KR_TICKER, local_price=70000.0,
        local_to_usd=_KRW_USD, fix_on=True,
    )
    avg = _addon_buy(
        trader, cap, ticker=_KR_TICKER, price=72000.0, amount_usd=45.85, market="KR",
    )

    # clause 1 -- avg_entry stays LOCAL (quantity-weighted KRW mean of 70000 and 72000)
    assert avg > 1000.0, f"avg_entry collapsed out of KRW scale: {avg}"
    assert 70000.0 <= avg <= 72000.0, (
        f"avg_entry {avg} is not between the two KRW lot prices"
    )

    # clause 2 -- the stop DERIVED from that entry is KRW-scale too
    new_stop, advance_iso = _advance(trader, base, avg)
    assert new_stop is not None, (
        "the breakeven branch did not fire -- the fixture is vacuous, not the code"
    )
    assert advance_iso is not None, "breakeven (not trailing) must be the branch under test"
    assert new_stop > 1000.0, f"advanced stop is not KRW-scale: {new_stop}"
    assert new_stop == pytest.approx(avg), (
        "the breakeven ratchet must set the stop to the LOCAL entry price"
    )

    # the money-safety consequence, stated as the check that actually runs live
    # (paper_trader.check_stop_losses compares a LOCAL current price to this stop)
    live_local_price = 71000.0
    assert not (live_local_price <= new_stop) or new_stop >= 70000.0, (
        "a KRW-scale stop must be comparable to a KRW-scale price"
    )


def test_61_3_kr_addon_currency_legacy_stop_is_untriggerable_usd_scale():
    """Mutation-resistant negative: the OFF path produces the defect this step fixes.

    Without this, the ON-only assertions above would pass against a formula that was
    never broken.
    """
    trader, cap, base = _trader(
        market="KR", ticker=_KR_TICKER, local_price=70000.0,
        local_to_usd=_KRW_USD, fix_on=False,
    )
    avg_legacy = _addon_buy(
        trader, cap, ticker=_KR_TICKER, price=72000.0, amount_usd=45.85, market="KR",
    )
    assert avg_legacy < 1000.0, (
        f"expected the legacy USD/local mix (~tens of USD), got {avg_legacy}"
    )

    new_stop, _iso = _advance(trader, base, avg_legacy)
    assert new_stop is not None and new_stop < 1000.0, (
        f"legacy path should yield a USD-scale stop, got {new_stop}"
    )
    # This is the silent failure the criterion exists to prevent: a real KRW price can
    # never fall to a USD-scale stop, so the stop-loss check never fires.
    live_local_price = 60000.0  # a 14% drawdown from entry -- SHOULD have stopped out
    assert not (live_local_price <= new_stop), (
        "sanity: the legacy stop is untriggerable by construction"
    )


def test_61_3_eu_addon_currency_stop_stays_eur_scale():
    """EU is the insidious case: EURUSD ~1.08 makes the corruption look plausible."""
    trader, cap, base = _trader(
        market="EU", ticker=_EU_TICKER, local_price=150.0,
        local_to_usd=_EUR_USD, fix_on=True,
    )
    avg = _addon_buy(
        trader, cap, ticker=_EU_TICKER, price=160.0, amount_usd=170.0, market="EU",
    )
    assert 150.0 <= avg <= 160.0, f"EUR avg_entry {avg} left the two lot prices"

    new_stop, advance_iso = _advance(trader, base, avg)
    assert new_stop is not None and advance_iso is not None
    assert new_stop == pytest.approx(avg)

    # contrast: the legacy path inflates by ~the FX rate, so the breakeven stop lands
    # ABOVE the true entry and fires ~8% early instead of never.
    trader_off, cap_off, base_off = _trader(
        market="EU", ticker=_EU_TICKER, local_price=150.0,
        local_to_usd=_EUR_USD, fix_on=False,
    )
    avg_off = _addon_buy(
        trader_off, cap_off, ticker=_EU_TICKER, price=160.0, amount_usd=170.0, market="EU",
    )
    stop_off, _ = _advance(trader_off, base_off, avg_off)
    assert avg_off > avg, "legacy EUR avg should be inflated by roughly the FX rate"
    assert stop_off > new_stop, "legacy EUR stop sits above the correct one"


def test_61_3_us_addon_currency_byte_identical_across_the_flag():
    """US (fx = 1) must be byte-identical ON vs OFF -- for the stop as well as the entry."""
    results = {}
    for fix_on in (True, False):
        trader, cap, base = _trader(
            market="US", ticker="NTAP", local_price=100.0, local_to_usd=1.0, fix_on=fix_on,
        )
        avg = _addon_buy(trader, cap, ticker="NTAP", price=110.0, amount_usd=110.0, market="US")
        stop, _iso = _advance(trader, base, avg)
        results[fix_on] = (avg, stop)
    (avg_on, stop_on), (avg_off, stop_off) = results[True], results[False]
    assert avg_on == pytest.approx(avg_off, abs=1e-6), (
        f"US avg_entry must not move with the flag: ON={avg_on} OFF={avg_off}"
    )
    assert stop_on == pytest.approx(stop_off, abs=1e-6), (
        f"US stop must not move with the flag: ON={stop_on} OFF={stop_off}"
    )


def test_61_3_fixture_reaches_breakeven_branch_not_a_vacuous_pass():
    """Guard the guard: prove the fixture can DISTINGUISH fired from not-fired.

    `_advance_stop` returns (None, None) when the breakeven preconditions are unmet
    (paper_trader.py:1449-1452). A fixture that silently trips those would make every
    assertion above vacuous, so assert both directions explicitly.
    """
    trader, _cap, base = _trader(
        market="KR", ticker=_KR_TICKER, local_price=70000.0,
        local_to_usd=_KRW_USD, fix_on=True,
    )
    threshold = float(getattr(trader.settings, "paper_default_stop_loss_pct", 8.0))

    fired, iso = _advance(trader, base, 70000.0)
    assert fired is not None and iso is not None

    # below the MFE threshold -> no advance
    below = dict(base)
    below["avg_entry_price"] = 70000.0
    below["stop_advanced_at_R"] = None
    below["stop_loss_price"] = 70000.0 * 0.92
    assert trader._advance_stop(below, new_mfe=threshold - 1.0) == (None, None)

    # stop already at/above entry -> one-shot already satisfied
    at_entry = dict(below)
    at_entry["stop_loss_price"] = 70000.0
    assert trader._advance_stop(at_entry, new_mfe=threshold + 10.0) == (None, None)


# ─────────── criterion 4: the mark carries an as-of timestamp ───────────

def test_61_3_mark_to_market_currency_writes_marked_at_as_of_indicator():
    """Every marked position row carries `marked_at` -- the as-of indicator.

    Criterion 4 requires stored P&L to carry a mark timestamp so a non-US row cannot show
    a live local price beside an unlabeled stale P&L.
    """
    from datetime import datetime, timezone

    trader, cap, _base = _trader(
        market="KR", ticker=_KR_TICKER, local_price=70000.0,
        local_to_usd=_KRW_USD, fix_on=True,
    )
    before = datetime.now(timezone.utc)
    with patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx), \
         patch.object(pt, "_get_live_price", return_value=71000.0), \
         patch.object(pt, "_get_benchmark_return", return_value=0.0):
        trader.mark_to_market()

    row = cap["row"]
    assert "marked_at" in row, "mark_to_market must stamp marked_at on every position row"
    stamped = datetime.fromisoformat(row["marked_at"])
    assert stamped.tzinfo is not None, "marked_at must be timezone-aware ISO-8601"
    assert before <= stamped <= datetime.now(timezone.utc), (
        f"marked_at {stamped} is not from this mark run"
    )


def test_61_3_marked_at_currency_is_prunable_on_a_pre_migration_schema():
    """`marked_at` must be in the prune set, or a pre-migration table breaks every save.

    `_safe_save_position` retries without `_POSITION_RT_FIELDS` when BigQuery reports a
    schema error (paper_trader.py:1483-1492). A new column missing from that set makes
    the retry fail identically to the first attempt.
    """
    assert "marked_at" in pt.PaperTrader._POSITION_RT_FIELDS

    # The retry only happens for errors `_looks_like_schema_error` recognises
    # (paper_trader.py:1525-1530), so drive it with BigQuery's real wording for an
    # unknown column in a MERGE rather than an invented string -- otherwise this test
    # would prove the retry works for a message BigQuery never sends.
    bq_missing_column = "400 Unrecognized name: marked_at at [1:87]"
    assert pt.PaperTrader._looks_like_schema_error(Exception(bq_missing_column))

    trader, _cap, _base = _trader(
        market="US", ticker="NTAP", local_price=100.0, local_to_usd=1.0, fix_on=False,
    )
    saved: list[dict] = []

    def _save(row):
        if "marked_at" in row:
            raise Exception(bq_missing_column)
        saved.append(row)

    trader.bq.save_paper_position.side_effect = _save
    trader._safe_save_position({"ticker": "NTAP", "marked_at": "2026-08-07T00:00:00+00:00"})
    assert saved and "marked_at" not in saved[0], (
        "the pre-migration retry must drop marked_at and still persist the row"
    )
