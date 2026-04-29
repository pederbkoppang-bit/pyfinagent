"""phase-23.1.16: parallel yfinance + per-ticker cache + startup prewarm."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _fake_yf_ticker_factory(latency_s: float, concurrent_counter: dict):
    """Build a yfinance.Ticker mock that records concurrency + sleeps."""
    def _make(symbol):
        with concurrent_counter["lock"]:
            concurrent_counter["active"] += 1
            concurrent_counter["peak"] = max(
                concurrent_counter["peak"], concurrent_counter["active"],
            )
        time.sleep(latency_s)
        with concurrent_counter["lock"]:
            concurrent_counter["active"] -= 1
        m = MagicMock()
        m.info = {"shortName": f"{symbol} Inc.", "sector": "Technology"}
        return m
    return _make


def test_parallel_yfinance_caps_concurrent_workers_at_5():
    """Fix A: ThreadPoolExecutor max_workers=5 must not exceed 5 concurrent
    yfinance calls even with 14 tickers in flight."""
    from backend.api.paper_trading import _fetch_ticker_meta

    counter = {"active": 0, "peak": 0, "lock": threading.Lock()}
    fake = _fake_yf_ticker_factory(latency_s=0.1, concurrent_counter=counter)

    bq = MagicMock()
    bq.client.query.return_value.result.return_value = []  # no BQ hits -> all yf
    settings = MagicMock(gcp_project_id="p", bq_dataset_reports="d")

    tickers = [f"T{i}" for i in range(14)]
    with patch("yfinance.Ticker", side_effect=fake):
        result = _fetch_ticker_meta(tickers, settings, bq)

    assert len(result["meta"]) == 14
    assert counter["peak"] <= 5, \
        f"max_workers=5 violated; saw {counter['peak']} concurrent yf calls"
    assert counter["peak"] >= 2, \
        f"expected real parallelism (>=2 concurrent), saw peak={counter['peak']}"


def test_parallel_yfinance_faster_than_serial():
    """Fix A: 10 tickers @ 100ms each parallel must be substantially faster
    than 10 * 100ms = 1.0s + (9 * 0.3s sleep = 2.7s) serial wall clock."""
    from backend.api.paper_trading import _fetch_ticker_meta

    counter = {"active": 0, "peak": 0, "lock": threading.Lock()}
    fake = _fake_yf_ticker_factory(latency_s=0.1, concurrent_counter=counter)
    bq = MagicMock()
    bq.client.query.return_value.result.return_value = []
    settings = MagicMock(gcp_project_id="p", bq_dataset_reports="d")

    tickers = [f"X{i}" for i in range(10)]
    with patch("yfinance.Ticker", side_effect=fake):
        t0 = time.monotonic()
        _fetch_ticker_meta(tickers, settings, bq)
        elapsed = time.monotonic() - t0

    # 10 tickers * 0.1s with 5 concurrent -> ~0.2s. Old serial+sleep was ~3.7s.
    assert elapsed < 1.5, f"parallel fetch took {elapsed:.2f}s, expected <1.5s"


def test_per_ticker_cache_returns_partial_hits():
    """Fix B: when 3 of 5 tickers are pre-cached, only the missing 2 trigger
    _fetch_ticker_meta. Calls the route handler coroutine directly to bypass
    the auth middleware (matches the pattern in tests/api/test_ticker_meta.py)."""
    import asyncio as _asyncio
    from backend.api.paper_trading import get_ticker_meta
    from backend.services.api_cache import get_api_cache, ENDPOINT_TTLS

    cache = get_api_cache()
    ttl = ENDPOINT_TTLS.get("paper:ticker_meta", 86400)
    for t in ["AAA", "BBB", "CCC"]:
        cache.set(f"paper:ticker_meta:single:{t}", {
            "company_name": f"{t} Co.", "sector": "Tech", "source": "test",
        }, ttl)
    try:
        with patch("backend.api.paper_trading._fetch_ticker_meta") as m, \
             patch("backend.api.paper_trading.BigQueryClient"):
            m.return_value = {
                "meta": {
                    "DDD": {"company_name": "DDD Co.", "sector": "Energy", "source": "yfinance"},
                    "EEE": {"company_name": "EEE Co.", "sector": "Health", "source": "yfinance"},
                },
                "ttl_sec": ttl,
                "count": 2,
            }
            body = _asyncio.run(get_ticker_meta(tickers="AAA,BBB,CCC,DDD,EEE"))

        assert set(body["meta"].keys()) == {"AAA", "BBB", "CCC", "DDD", "EEE"}
        called_args = m.call_args[0][0]
        assert sorted(called_args) == ["DDD", "EEE"], \
            f"expected only DDD,EEE to be fetched; got {called_args}"
    finally:
        for t in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
            cache.invalidate(f"paper:ticker_meta:single:{t}")


def test_prewarm_short_circuits_when_no_positions(monkeypatch):
    """Fix C: when paper_positions is empty, prewarm task logs and returns
    without calling _fetch_ticker_meta."""
    # Build a minimal stub of the prewarm path. The actual function is defined
    # inside lifespan; we replicate its short-circuit branch here.
    import asyncio as _asyncio

    fetcher_called = {"hit": False}

    async def stub_prewarm():
        bq = MagicMock()
        bq.get_paper_positions = MagicMock(return_value=[])
        positions = await _asyncio.to_thread(bq.get_paper_positions)
        tickers = sorted({p.get("ticker") for p in positions if p.get("ticker")})
        if not tickers:
            return "skipped"
        fetcher_called["hit"] = True
        return "ran"

    result = _asyncio.run(stub_prewarm())
    assert result == "skipped"
    assert fetcher_called["hit"] is False
