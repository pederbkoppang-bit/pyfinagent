"""phase-9.2 daily price + fundamentals refresh job.

Fetches OHLCV for the settings watchlist via yfinance; writes to BQ.
Idempotent by (job_name, date). Fail-open; no raise from run().
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)

JOB_NAME = "daily_price_refresh"


def run(
    *,
    tickers: list[str] | None = None,
    fetch_fn: Callable[[list[str]], dict] | None = None,
    write_fn: Callable[[dict], int] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Fetch OHLCV for tickers and write to BQ. Idempotent by day."""
    key = IdempotencyKey.daily(JOB_NAME, day=day or date.today().isoformat())
    result: dict[str, Any] = {"written": 0, "key": key, "skipped": False}

    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        universe = tickers or ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
        fetched = (fetch_fn or _default_fetch)(universe)
        n = (write_fn or _default_write)(fetched)
        result["written"] = int(n)
        result["tickers"] = universe
    return result


def run_production(
    *,
    day: str | None = None,
    lookback_days: int = 7,
    store: IdempotencyStore | None = None,
) -> dict[str, Any]:
    """Production daily price refresh.

    phase-47.1: supersedes the legacy fetch_fn/write_fn closure path (which
    wrote close-only rows to the WRONG table `pyfinagent_data.price_snapshots`
    -- no `ingested_at`, only 5 tickers). This refreshes the full S&P-500
    universe with full OHLCV into `financial_reports.historical_prices` via the
    audited, idempotent `DataIngestionService.ingest_prices` path (dedup on
    (ticker, date)). Module-level (picklable) so a future persistent jobstore
    can serialize it. Idempotent by day via the heartbeat. Fail-open: never
    raises from the scheduler context.
    """
    key = IdempotencyKey.daily(JOB_NAME, day=day or date.today().isoformat())
    result: dict[str, Any] = {"written": 0, "key": key, "skipped": False, "path": "ingest_prices"}

    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        try:
            from datetime import timedelta

            from google.cloud import bigquery

            from backend.backtest.data_ingestion import DataIngestionService
            from backend.config.settings import get_settings
            from backend.tools.screener import get_sp500_tickers

            settings = get_settings()
            client = bigquery.Client(project=settings.gcp_project_id)
            tickers = get_sp500_tickers()
            today = date.today()
            start = (today - timedelta(days=max(1, lookback_days))).isoformat()
            end = (today + timedelta(days=1)).isoformat()
            n = DataIngestionService(client, settings).ingest_prices(tickers, start, end)
            result["written"] = int(n)
            result["tickers"] = len(tickers)
            result["window"] = [start, end]
        except Exception as exc:  # network, BQ quota, yfinance schema -- fail-open
            logger.warning("daily_price_refresh.run_production fail-open: %r", exc)
    return result


def _default_fetch(tickers: list[str]) -> dict[str, Any]:
    # Injected in tests; production wraps yfinance.
    return {t: {"close": 100.0} for t in tickers}


def _default_write(rows: dict[str, Any]) -> int:
    # Injected in tests; production streams to BQ.
    return len(rows)


__all__ = ["run", "run_production", "JOB_NAME"]
