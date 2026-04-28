"""phase-23.1.13: screen_universe accepts sector_lookup and attaches sector to candidates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.tools.screener import screen_universe


def _fake_yf_dataframe(tickers: list[str]) -> pd.DataFrame:
    """Build a multi-index DataFrame matching yfinance.download output."""
    rows = []
    for t in tickers:
        rows.append({"ticker": t})
    idx = pd.date_range(end="2026-04-28", periods=180, freq="B")
    cols = pd.MultiIndex.from_product([tickers, ["Close", "Volume"]])
    data = {}
    for t in tickers:
        data[(t, "Close")] = [100 + i * 0.5 for i in range(180)]
        data[(t, "Volume")] = [200_000 for _ in range(180)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_screen_universe_accepts_sector_lookup_kwarg():
    """Passing sector_lookup=dict produces results with sector field."""
    df = _fake_yf_dataframe(["AAPL", "XOM"])
    lookup = {
        "AAPL": {"sector": "Information Technology", "company_name": "Apple Inc."},
        "XOM": {"sector": "Energy", "company_name": "Exxon Mobil"},
    }
    with patch("yfinance.download", return_value=df):
        results = screen_universe(
            tickers=["AAPL", "XOM"], period="6mo", sector_lookup=lookup,
        )
    sectors = {r["ticker"]: r.get("sector") for r in results}
    assert sectors.get("AAPL") == "Information Technology"
    assert sectors.get("XOM") == "Energy"


def test_screen_universe_without_lookup_omits_sector_backward_compat():
    """No sector_lookup -> results lack sector field (backward compat)."""
    df = _fake_yf_dataframe(["AAPL"])
    with patch("yfinance.download", return_value=df):
        results = screen_universe(tickers=["AAPL"], period="6mo")
    if results:
        # sector key absent OR None -- both acceptable for backward compat
        assert "sector" not in results[0] or results[0].get("sector") in (None, "")


def test_screen_universe_lookup_handles_missing_ticker():
    """sector_lookup missing a ticker -> that result has no sector (graceful)."""
    df = _fake_yf_dataframe(["AAPL", "XOM"])
    lookup = {"AAPL": {"sector": "Information Technology"}}  # XOM absent
    with patch("yfinance.download", return_value=df):
        results = screen_universe(
            tickers=["AAPL", "XOM"], period="6mo", sector_lookup=lookup,
        )
    by_ticker = {r["ticker"]: r for r in results}
    assert by_ticker.get("AAPL", {}).get("sector") == "Information Technology"
    # XOM absent from lookup -> no sector field set
    assert "sector" not in by_ticker.get("XOM", {})


def test_screen_universe_lookup_accepts_string_value():
    """Some callers may pass {ticker: 'Sector'} (string) instead of dict."""
    df = _fake_yf_dataframe(["AAPL"])
    lookup = {"AAPL": "Information Technology"}
    with patch("yfinance.download", return_value=df):
        results = screen_universe(
            tickers=["AAPL"], period="6mo", sector_lookup=lookup,
        )
    if results:
        assert results[0].get("sector") == "Information Technology"
