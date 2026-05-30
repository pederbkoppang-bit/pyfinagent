"""phase-50.4: market-calendar gating -- per-exchange trading-day correctness.

All dates verified against the installed exchange_calendars==4.13.2 + the
published Xetra/KRX 2026 calendars. The previous is_trading_day was a latent
no-op (cal.days removed in 4.0 -> bare except -> always True); these tests pin
the rewritten cal.is_session behaviour + the independent per-market gating.
"""
from backend.backtest import markets as m


def test_us_weekday_and_weekend():
    assert m.is_trading_day("2026-06-15", "US") is True    # Monday
    assert m.is_trading_day("2026-06-13", "US") is False   # Saturday


def test_eu_labour_day_closed_us_open():
    # 2026-05-01 (Fri) = Labour Day: Xetra CLOSED, NYSE OPEN -> independent gating
    assert m.is_trading_day("2026-05-01", "EU") is False
    assert m.is_trading_day("2026-05-01", "US") is True


def test_kr_lunar_holiday_closed_us_open():
    # 2026-02-17 (Tue) = Seollal (lunar new year): KRX CLOSED, NYSE OPEN
    assert m.is_trading_day("2026-02-17", "KR") is False
    assert m.is_trading_day("2026-02-17", "US") is True
    # 2026-09-25 (Fri) = Chuseok: KRX CLOSED, NYSE OPEN
    assert m.is_trading_day("2026-09-25", "KR") is False
    assert m.is_trading_day("2026-09-25", "US") is True


def test_normal_weekday_all_open():
    for mk in ("US", "EU", "KR"):
        assert m.is_trading_day("2026-06-15", mk) is True


def test_failopen_unknown_market():
    # unknown market -> get_market_config defaults to US -> a US trading day
    assert m.is_trading_day("2026-06-15", "ZZ") is True


def test_not_always_true_regression_guard():
    # the old bug returned True for EVERYTHING; prove it now returns False somewhere
    assert m.is_trading_day("2026-06-13", "US") is False   # weekend
    assert m.is_trading_day("2026-05-01", "EU") is False   # holiday


def test_market_derivation_for_gating():
    # the live gate keys off market_for_symbol; US is never gated
    assert m.market_for_symbol("AAPL") == "US"
    assert m.market_for_symbol("SAP.DE") == "EU"
    assert m.market_for_symbol("005930.KS") == "KR"
