"""phase-64.3: per-market screener / data-quality gap tests (pure pandas).

price_quality.validate_ohlcv is US-byte-identical (fast-path no-op) and
applies R1-R3 drop rules to non-US markets; markets.market_for_symbol derives
the market code from the yfinance suffix.
"""
from __future__ import annotations

import pandas as pd

from backend.backtest.markets import market_for_symbol
from backend.tools.price_quality import validate_ohlcv


def _df(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_64_3_screener_market_us_fast_path_no_op():
    df = _df([[100, 105, 95, 100, 1000]])
    clean, report = validate_ohlcv(df, market="US", ticker="AAPL")
    assert report["dropped"] == 0
    assert clean is df  # byte-identical: same object returned


def test_64_3_screener_market_r1_impossible_bar_dropped():
    # high(90) < low(110) -> impossible OHLC -> R1 drop.
    df = _df([[100, 90, 110, 100, 1000]])
    _clean, report = validate_ohlcv(df, market="EU", ticker="SAP.DE")
    assert report["dropped"] == 1
    assert any("R1" in r for r in report["reasons"])


def test_64_3_screener_market_r2_identical_zero_vol_dropped_vs_flagged():
    # identical OHLC + zero volume -> drop.
    drop_df = _df([[100, 100, 100, 100, 0]])
    _c, drop_rep = validate_ohlcv(drop_df, market="KR", ticker="005930.KS")
    assert drop_rep["dropped"] == 1
    # identical OHLC WITH volume -> flagged, NOT dropped.
    flag_df = _df([[100, 100, 100, 100, 5000]])
    clean, flag_rep = validate_ohlcv(flag_df, market="KR", ticker="005930.KS")
    assert flag_rep["dropped"] == 0
    assert flag_rep["flagged"] >= 1
    assert len(clean) == 1


def test_64_3_screener_market_r3_extreme_move_dropped():
    # row2 close jumps +200% vs row1 -> R3 drop (single-day extreme return).
    df = _df([[100, 105, 95, 100, 1000], [100, 310, 100, 300, 1000]])
    clean, report = validate_ohlcv(df, market="EU", ticker="SAP.DE")
    assert report["dropped"] == 1
    assert len(clean) == 1


def test_64_3_screener_market_market_for_symbol():
    assert market_for_symbol("005930.KS") == "KR"
    assert market_for_symbol("SAP.DE") == "EU"
    assert market_for_symbol("AAPL") == "US"
