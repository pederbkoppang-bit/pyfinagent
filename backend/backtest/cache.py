"""
BQ query cache — in-memory cache for historical data queries.
Supports bulk preloading (2-3 BQ queries for entire backtest) with
per-ticker slicing, plus exact-match fallback for ad-hoc queries.

Phase 2.9: market parameter added to preloaders (passthrough for now,
all data is US. Phase 5 will filter by market column in BQ tables).
"""

import logging
from typing import Optional

import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# Default market — used when no market parameter is passed
_DEFAULT_MARKET = "US"

# ── Preloaded full-range data (keyed by ticker only, sliced on read) ─
_prices_full: dict[str, pd.DataFrame] = {}
_fundamentals_full: dict[str, list[dict]] = {}
_macro_full: dict[str, list[dict]] = {}  # series_id -> [{value, date}, ...] sorted by date DESC

# ── Exact-match caches (fallback for non-preloaded queries) ──────────
_prices_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
_fundamentals_cache: dict[tuple[str, str], list[dict]] = {}
_macro_cache: dict[str, dict] = {}

# ── Cache statistics ─────────────────────────────────────────────────
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}

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


def get_cache_stats() -> dict[str, int]:
    """Return current cache hit/miss counters."""
    return dict(_cache_stats)


def clear_cache():
    """Clear all caches between backtest runs."""
    _prices_full.clear()
    _fundamentals_full.clear()
    _macro_full.clear()
    _prices_cache.clear()
    _fundamentals_cache.clear()
    _macro_cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    logger.info("BQ cache cleared")


# ── Bulk Preloaders ──────────────────────────────────────────────────

def preload_prices(tickers: list[str], start_date: str, end_date: str, market: str = _DEFAULT_MARKET) -> int:
    """Bulk-load all price data for tickers in a single BQ query.

    Stores per-ticker DataFrames in _prices_full.  Subsequent calls to
    cached_prices() slice from these rather than hitting BQ individually.
    Returns the total number of rows loaded.
    
    Phase 2.9: market param accepted but not filtered yet (all data is US).
    Phase 5: will add WHERE market = @market to the query.
    """
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    if not tickers:
        return 0

    query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM `{_table("historical_prices")}`
        WHERE ticker IN UNNEST(@tickers)
          AND date >= @start AND date <= @end
        ORDER BY ticker, date ASC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("start", "STRING", start_date),
        bigquery.ScalarQueryParameter("end", "STRING", end_date),
    ])
    rows = list(_bq_client.query(query, job_config=job_config).result())

    if not rows:
        logger.warning("preload_prices: 0 rows returned for %d tickers", len(tickers))
        return 0

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    total_rows = len(df)

    for ticker, group in df.groupby("ticker"):
        _prices_full[str(ticker)] = (
            group.drop(columns=["ticker"]).set_index("date").sort_index()
        )

    logger.info(
        "Preloaded prices for %d tickers (%s rows) in single BQ query",
        len(_prices_full), f"{total_rows:,}",
    )
    return total_rows


def preload_fundamentals(tickers: list[str], market: str = _DEFAULT_MARKET) -> int:
    """Bulk-load all fundamental data for tickers in a single BQ query.

    Stores per-ticker lists (sorted by report_date DESC) in _fundamentals_full.
    Returns the total number of rows loaded.
    
    Phase 2.9: market param accepted but not filtered yet.
    """
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    if not tickers:
        return 0

    query = f"""
        SELECT *
        FROM `{_table("historical_fundamentals")}`
        WHERE ticker IN UNNEST(@tickers)
        ORDER BY ticker, report_date DESC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
    ])
    rows = list(_bq_client.query(query, job_config=job_config).result())

    if not rows:
        logger.warning("preload_fundamentals: 0 rows returned for %d tickers", len(tickers))
        return 0

    total_rows = len(rows)
    # Group by ticker, maintain DESC order
    current_ticker: str | None = None
    current_list: list[dict] = []
    for r in rows:
        row = dict(r)
        t = row.get("ticker", "")
        if t != current_ticker:
            if current_ticker is not None:
                _fundamentals_full[current_ticker] = current_list
            current_ticker = t
            current_list = []
        current_list.append(row)
    if current_ticker is not None:
        _fundamentals_full[current_ticker] = current_list

    logger.info(
        "Preloaded fundamentals for %d tickers (%s rows) in single BQ query",
        len(_fundamentals_full), f"{total_rows:,}",
    )
    return total_rows


def preload_macro() -> int:
    """Bulk-load all macro data in a single BQ query.

    Stores per-series lists (sorted by date DESC) in _macro_full.
    Returns the total number of rows loaded.
    """
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"

    query = f"""
        SELECT series_id, value, date
        FROM `{_table("historical_macro")}`
        ORDER BY series_id, date DESC
    """
    rows = list(_bq_client.query(query).result())

    if not rows:
        logger.warning("preload_macro: 0 rows returned")
        return 0

    total_rows = len(rows)
    current_series: str | None = None
    current_list: list[dict] = []
    for r in rows:
        row = dict(r)
        s = row.get("series_id", "")
        if s != current_series:
            if current_series is not None:
                _macro_full[current_series] = current_list
            current_series = s
            current_list = []
        current_list.append({"value": row["value"], "date": row["date"]})
    if current_series is not None:
        _macro_full[current_series] = current_list

    logger.info(
        "Preloaded macro for %d series (%s rows) in single BQ query",
        len(_macro_full), f"{total_rows:,}",
    )
    return total_rows


# ── Cached Accessors (preload-aware) ─────────────────────────────────

def cached_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data — slices from preloaded data if available, else BQ."""
    # 1. Try preloaded full-range data (fast path)
    if ticker in _prices_full:
        _cache_stats["hits"] += 1
        full = _prices_full[ticker]
        sliced = full.loc[start_date:end_date]
        if not sliced.empty:
            return sliced
        # Preloaded but no data in this range — return empty
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # 2. Try exact-match cache
    key = (ticker, start_date, end_date)
    if key in _prices_cache:
        _cache_stats["hits"] += 1
        return _prices_cache[key]

    # 3. Fall back to individual BQ query (with timeout)
    _cache_stats["misses"] += 1
    logger.debug("BQ fallback: prices for %s (%s to %s)", ticker, start_date, end_date)
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
    try:
        rows = list(_bq_client.query(query, job_config=job_config).result(timeout=30))
    except Exception as e:
        logger.warning("BQ prices query timed out for %s: %s", ticker, e)
        rows = []
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    _prices_cache[key] = df
    return df


def cached_fundamentals(ticker: str, cutoff_date: str) -> list[dict]:
    """Get up to 5 most recent quarterly fundamentals as-of cutoff_date.

    Returns a list ordered by report_date DESC (index 0 = most recent).
    Multiple quarters are needed for YoY revenue growth computation.
    """
    # 1. Try preloaded full data
    if ticker in _fundamentals_full:
        _cache_stats["hits"] += 1
        all_rows = _fundamentals_full[ticker]
        filtered = [r for r in all_rows if str(r.get("report_date", "")) <= cutoff_date]
        return filtered[:5]

    # 2. Try exact-match cache
    key = (ticker, cutoff_date)
    if key in _fundamentals_cache:
        _cache_stats["hits"] += 1
        return _fundamentals_cache[key]

    # 3. Fall back to individual BQ query (with timeout)
    _cache_stats["misses"] += 1
    logger.debug("BQ fallback: fundamentals for %s (cutoff %s)", ticker, cutoff_date)
    query = f"""
        SELECT *
        FROM `{_table("historical_fundamentals")}`
        WHERE ticker = @ticker AND report_date <= @cutoff
        ORDER BY report_date DESC
        LIMIT 5
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        bigquery.ScalarQueryParameter("cutoff", "STRING", cutoff_date),
    ])
    assert _bq_client is not None, "Cache not initialized — call init_cache() first"
    try:
        rows = list(_bq_client.query(query, job_config=job_config).result(timeout=30))
    except Exception as e:
        logger.warning("BQ fundamentals query timed out for %s: %s", ticker, e)
        rows = []
    result = [dict(r) for r in rows]

    _fundamentals_cache[key] = result
    return result


def cached_macro(cutoff_date: str) -> dict:
    """Get most recent FRED macro values as-of cutoff_date."""
    if cutoff_date in _macro_cache:
        _cache_stats["hits"] += 1
        return _macro_cache[cutoff_date]

    # 1. Try preloaded full data (fast path — binary search per series)
    if _macro_full:
        _cache_stats["hits"] += 1
        result = {}
        for series_id, entries in _macro_full.items():
            # entries sorted by date DESC — find first entry <= cutoff_date
            for entry in entries:
                if str(entry["date"]) <= cutoff_date:
                    result[series_id] = entry
                    break
        _macro_cache[cutoff_date] = result
        return result

    # 2. Fall back to individual BQ query (with timeout)
    _cache_stats["misses"] += 1
    logger.debug("BQ fallback: macro (cutoff %s)", cutoff_date)
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
    try:
        rows = list(_bq_client.query(query, job_config=job_config).result(timeout=30))
    except Exception as e:
        logger.warning("BQ macro query timed out: %s", e)
        rows = []
    result = {r["series_id"]: {"value": r["value"], "date": r["date"]} for r in rows}

    _macro_cache[cutoff_date] = result
    return result
