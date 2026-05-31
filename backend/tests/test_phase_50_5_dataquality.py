"""phase-50.5: data-quality gate -- US byte-identical no-op + intl bad-bar detection.

Offline (constructed DataFrames). Pins: US fast-path returns input unchanged;
unambiguous bad intl bars DROPPED (identical-OHLC+zero-vol, impossible OHLC,
>50% move); merely-large real moves FLAGGED not dropped; clean series passes.
"""
import pandas as pd
import pytest

from backend.tools.price_quality import validate_ohlcv, is_bad_bar


def _df(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def _clean_series(n=20, start=100.0):
    rows = []
    px = start
    for i in range(n):
        px = px * (1.0 + (0.005 if i % 2 == 0 else -0.004))  # gentle drift
        rows.append([px * 0.99, px * 1.01, px * 0.985, px, 1_000_000 + i])
    return _df(rows)


def test_us_is_byte_identical_noop():
    df = _clean_series()
    out, rep = validate_ohlcv(df, market="US")
    assert out is df                       # exact same object, untouched
    assert rep["dropped"] == 0 and rep["flagged"] == 0


def test_clean_intl_series_passes():
    df = _clean_series()
    out, rep = validate_ohlcv(df, market="EU")
    assert rep["dropped"] == 0
    assert len(out) == len(df)


def test_identical_ohlc_zero_vol_dropped():
    df = _clean_series(10)
    # inject a flat bar with zero volume (the documented yfinance bad-bar)
    df.loc[5] = [200.0, 200.0, 200.0, 200.0, 0]
    out, rep = validate_ohlcv(df, market="EU")
    assert rep["dropped"] >= 1
    assert 200.0 not in out["Close"].values


def test_impossible_ohlc_dropped():
    df = _clean_series(10)
    df.loc[4] = [100.0, 90.0, 110.0, 100.0, 5000]   # high<low, high<open -> impossible
    out, rep = validate_ohlcv(df, market="EU")
    assert rep["dropped"] >= 1


def test_extreme_move_dropped_but_real_move_flagged():
    df = _clean_series(12)
    # a 60% single-day jump in close -> data glitch -> DROP
    df.loc[6, "Close"] = df.loc[5, "Close"] * 1.60
    df.loc[6, "High"] = df.loc[6, "Close"] * 1.001
    out, rep = validate_ohlcv(df, market="EU")
    assert rep["dropped"] >= 1


def test_is_bad_bar_single():
    assert is_bad_bar(200, 200, 200, 200, volume=0) is True       # identical + zero vol
    assert is_bad_bar(100, 90, 110, 100, volume=5000) is True     # impossible OHLC
    assert is_bad_bar(-1, 1, -2, 0.5, volume=100) is True         # non-positive
    assert is_bad_bar(99, 101, 98.5, 100, volume=1_000_000) is False  # normal bar
    assert is_bad_bar(100, 100, 100, 100, volume=500000) is False     # flat but real volume -> keep


# ---- phase-50.5 criteria #1: per-market benchmark + FX-converted returns ----

import numpy as np  # noqa: E402

from backend.backtest.analytics import compute_baseline_strategies  # noqa: E402
from backend.backtest.markets import get_market_config, MARKET_CONFIG  # noqa: E402


def test_market_config_has_benchmarks():
    assert get_market_config("US")["benchmark"] == "SPY"
    assert get_market_config("EU")["benchmark"] == "^GDAXI"
    assert get_market_config("KR")["benchmark"] == "^KS11"
    assert get_market_config("NO")["benchmark"] == "^OSEAX"
    assert get_market_config("CA")["benchmark"] == "^GSPTSE"
    # every configured market must carry a benchmark (no silent gap)
    assert all("benchmark" in cfg for cfg in MARKET_CONFIG.values())


def _linear_prices_fn(_ticker, _start, _end):
    # deterministic 100 -> 120 series (=> +20% local return, telescoping product)
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, 30)})


def test_baseline_us_byte_identical_passthrough():
    """US: benchmark defaults to SPY, local==base==USD -> fx_ratio EXACTLY 1.0
    and returns pass through with no float drift (criteria #3: US unchanged)."""
    out = compute_baseline_strategies(
        _linear_prices_fn, "2026-01-01", "2026-04-01", ["AAA", "BBB", "CCC", "DDD"],
    )
    assert out["benchmark"] == "SPY"
    assert out["fx_ratio"] == 1.0
    assert out["base_currency"] == "USD"
    # BYTE-IDENTITY: fx_ratio==1.0 -> the benchmark return is the EXACT float the
    # pre-phase-50.5 code produced (raw close[-1]/close[0]-1)*100, no FX drift.
    ref = _linear_prices_fn("SPY", "2026-01-01", "2026-04-01")["close"]
    raw = float((ref.iloc[-1] / ref.iloc[0] - 1) * 100)
    assert out["spy_return_pct"] == raw                 # exact equality, not approx
    assert out["equal_weight_return_pct"] == pytest.approx(20.0, abs=1e-9)


def test_baseline_eu_fx_converted(monkeypatch):
    """EU: benchmark ^GDAXI, local EUR -> headline returns FX-converted to USD
    via endpoint fx (criteria #1). 100->120 local (+20%) with fx 1.0->1.10
    (ratio 1.10) => ((1.20)*1.10 - 1)*100 = 32.0% in USD."""
    def fake_fx(from_ccy, to_ccy, date=None):
        assert (from_ccy, to_ccy) == ("EUR", "USD")
        return 1.0 if date == "2026-01-01" else 1.10
    monkeypatch.setattr("backend.services.fx_rates.get_fx_rate", fake_fx)

    out = compute_baseline_strategies(
        _linear_prices_fn, "2026-01-01", "2026-04-01", ["AAA", "BBB", "CCC", "DDD"],
        benchmark="^GDAXI", base_currency="USD", local_currency="EUR",
    )
    assert out["benchmark"] == "^GDAXI"
    assert abs(out["fx_ratio"] - 1.10) < 1e-9
    assert abs(out["spy_return_pct"] - 32.0) < 1e-6        # FX-converted, != 20.0 local
    assert abs(out["equal_weight_return_pct"] - 32.0) < 1e-6
