"""
BQ query cache — in-memory LRU cache for historical data queries.
Prevents redundant BQ round-trips during a single backtest run
where expanding windows re-read overlapping date ranges.
"""

import functools
import logging
from typing import Optional

import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# Module-level caches (cleared between backtest runs)
_prices_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
_fundamentals_cache: dict[tuple[str, str], dict] = {}
_macro_cache: dict[str, dict] = {}

_bq_client: bigquery.Client | None = None
_project: str = ""
_dataset: str = ""


def init_cache(bq_client: bigquery.Client, project: str, dataset: str):
    """Initialize the cache with a BQ client reference."""
    global _bq_client, _project, _dataset
    _bq_client = bq_client
    _project = project
    _dataset = dataset


def _table(name: str) -> str:
    return f"{_project}.{_dataset}.{name}"


def clear_cache():
    """Clear all caches between backtest runs."""
    _prices_cache.clear()
    _fundamentals_cache.clear()
    _macro_cache.clear()
    logger.info("BQ cache cleared")


def cached_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data from cache or BQ."""
    key = (ticker, start_date, end_date)
    if key in _prices_cache:
        return _prices_cache[key]

    query = f"""
        SELECT date, open, high, low, close, volume
        FROM `{_table("historical_prices")}`
        WHERE ticker = @ticker AND date >= @start AND date <= @end
        ORDER BY date ASC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        bigquery.ScalarQueryParameter("start", "STRING", start_date),
        bigquery.ScalarQueryParameter("end", "STRING", end_date),
    ])
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    rows = list(_bq_client.query(query, job_config=job_config).result())
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    _prices_cache[key] = df
    return df


def cached_fundamentals(ticker: str, cutoff_date: str) -> dict:
    """Get most recent quarterly fundamentals as-of cutoff_date."""
    key = (ticker, cutoff_date)
    if key in _fundamentals_cache:
        return _fundamentals_cache[key]

    query = f"""
        SELECT *
        FROM `{_table("historical_fundamentals")}`
        WHERE ticker = @ticker AND report_date <= @cutoff
        ORDER BY report_date DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        bigquery.ScalarQueryParameter("cutoff", "STRING", cutoff_date),
    ])
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    rows = list(_bq_client.query(query, job_config=job_config).result())
    result = dict(rows[0]) if rows else {}

    _fundamentals_cache[key] = result
    return result


def cached_macro(cutoff_date: str) -> dict:
    """Get most recent FRED macro values as-of cutoff_date."""
    if cutoff_date in _macro_cache:
        return _macro_cache[cutoff_date]

    query = f"""
        SELECT series_id, value, date
        FROM (
            SELECT series_id, value, date,
                   ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC) as rn
            FROM `{_table("historical_macro")}`
            WHERE date <= @cutoff
        )
        WHERE rn = 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "STRING", cutoff_date),
    ])
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    rows = list(_bq_client.query(query, job_config=job_config).result())
    result = {r["series_id"]: {"value": r["value"], "date": r["date"]} for r in rows}

    _macro_cache[cutoff_date] = result
    return result
