"""phase-23.1.10: ticker-meta endpoint + BQ-first / yfinance-fallback resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.api.paper_trading import (
    _fetch_ticker_meta,
    _yfinance_ticker_info,
)


def test_yfinance_info_handles_exception_gracefully():
    """When yf.Ticker raises, fallback dict is returned."""
    with patch("yfinance.Ticker", side_effect=RuntimeError("boom")):
        out = _yfinance_ticker_info("ZZZZ")
    assert out == {"company_name": "ZZZZ", "sector": "", "source": "error"}


def test_yfinance_info_uses_short_name_when_present():
    fake_ticker = MagicMock()
    fake_ticker.info = {"shortName": "Apple Inc.", "sector": "Information Technology"}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        out = _yfinance_ticker_info("AAPL")
    assert out["company_name"] == "Apple Inc."
    assert out["sector"] == "Information Technology"
    assert out["source"] == "yfinance"


def test_yfinance_info_falls_through_to_long_name():
    fake_ticker = MagicMock()
    fake_ticker.info = {"shortName": None, "longName": "Apple Inc.", "sector": "Tech"}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        out = _yfinance_ticker_info("AAPL")
    assert out["company_name"] == "Apple Inc."


def test_yfinance_info_falls_through_to_ticker_when_no_name():
    fake_ticker = MagicMock()
    fake_ticker.info = {}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        out = _yfinance_ticker_info("XYZ")
    assert out["company_name"] == "XYZ"
    assert out["sector"] == ""


def test_fetch_ticker_meta_empty_input_returns_empty():
    out = _fetch_ticker_meta([], settings=MagicMock(), bq=MagicMock())
    assert out == {"meta": {}, "ttl_sec": 86400, "count": 0}


def test_fetch_ticker_meta_bq_hit_skips_yfinance():
    """When BQ returns both company_name AND sector, yfinance is NOT called."""
    fake_settings = MagicMock(gcp_project_id="proj", bq_dataset_reports="ds")
    bq_row = MagicMock()
    bq_row.__getitem__ = lambda self, k: {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Information Technology",
    }[k]
    fake_query_result = [bq_row]

    fake_bq_client = MagicMock()
    fake_bq_client.client.query.return_value.result.return_value = fake_query_result

    with patch("yfinance.Ticker") as yf_mock:
        out = _fetch_ticker_meta(["AAPL"], settings=fake_settings, bq=fake_bq_client)

    yf_mock.assert_not_called()
    assert out["count"] == 1
    assert out["meta"]["AAPL"]["company_name"] == "Apple Inc."
    assert out["meta"]["AAPL"]["sector"] == "Information Technology"
    assert out["meta"]["AAPL"]["source"] == "bq"


def test_fetch_ticker_meta_bq_missing_sector_falls_back_to_yfinance():
    """When BQ has company_name but sector is empty/null, yfinance fills the gap."""
    fake_settings = MagicMock(gcp_project_id="proj", bq_dataset_reports="ds")
    bq_row = MagicMock()
    bq_row.__getitem__ = lambda self, k: {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": None,
    }[k]
    fake_bq_client = MagicMock()
    fake_bq_client.client.query.return_value.result.return_value = [bq_row]

    fake_yf_ticker = MagicMock()
    fake_yf_ticker.info = {"shortName": "Apple Inc.", "sector": "Information Technology"}

    with patch("yfinance.Ticker", return_value=fake_yf_ticker):
        out = _fetch_ticker_meta(["AAPL"], settings=fake_settings, bq=fake_bq_client)

    # BQ name retained; yfinance sector added; source upgraded to bq+yf
    assert out["meta"]["AAPL"]["company_name"] == "Apple Inc."
    assert out["meta"]["AAPL"]["sector"] == "Information Technology"
    assert out["meta"]["AAPL"]["source"] == "bq+yf"


def test_fetch_ticker_meta_bq_query_failure_falls_back_to_yfinance():
    """BQ down -> all tickers resolved via yfinance only."""
    fake_settings = MagicMock(gcp_project_id="proj", bq_dataset_reports="ds")
    fake_bq_client = MagicMock()
    fake_bq_client.client.query.side_effect = RuntimeError("BQ down")

    fake_yf_ticker = MagicMock()
    fake_yf_ticker.info = {"shortName": "Apple Inc.", "sector": "Tech"}

    with patch("yfinance.Ticker", return_value=fake_yf_ticker):
        out = _fetch_ticker_meta(["AAPL"], settings=fake_settings, bq=fake_bq_client)

    assert out["meta"]["AAPL"]["source"] == "yfinance"
    assert out["meta"]["AAPL"]["company_name"] == "Apple Inc."


def test_fetch_ticker_meta_response_shape():
    fake_settings = MagicMock(gcp_project_id="proj", bq_dataset_reports="ds")
    fake_bq_client = MagicMock()
    fake_bq_client.client.query.return_value.result.return_value = []

    fake_yf_ticker = MagicMock()
    fake_yf_ticker.info = {"shortName": "X Corp", "sector": "Energy"}

    with patch("yfinance.Ticker", return_value=fake_yf_ticker):
        out = _fetch_ticker_meta(["AAPL", "MSFT"], settings=fake_settings, bq=fake_bq_client)

    assert set(out.keys()) == {"meta", "ttl_sec", "count"}
    assert out["ttl_sec"] == 86400
    assert out["count"] == 2
    for t in ("AAPL", "MSFT"):
        assert "company_name" in out["meta"][t]
        assert "sector" in out["meta"][t]
        assert "source" in out["meta"][t]
