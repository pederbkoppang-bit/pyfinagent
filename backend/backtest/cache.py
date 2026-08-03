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

# phase-25.D7: max acceptable age for the most-recent macro row.
# 35 days covers FRED-monthly series (publication lag ~10-15 days
# after month end) with grace. Older than this and `preload_macro`
# refuses to cache + emits a WARNING -- prevents silently powering
# backtests with stale macro factors. Closes audit bucket 24.7 F-5.
#
# phase-82.0: RETAINED as the fallback for any series absent from
# MACRO_SERIES_MAX_AGE_DAYS below, but it is NO LONGER the gate. Two
# independent defects made the flat global threshold unfit:
#   (a) it was applied to a GLOBAL max across all series, so daily DGS10 /
#       T10Y2Y masked a completely dead GDP -- the guard could not detect
#       the very failure it was written for; and
#   (b) FRED dates monthly series to the month START and quarterly series
#       to the quarter START, so a perfectly HEALTHY GDP newest row is
#       routinely ~211 days old and a healthy CPIAUCSL ~72 days. A flat
#       35-day bound is therefore unsatisfiable per-series -- it would
#       condemn a fully current table.
MACRO_MAX_AGE_DAYS = 35

# phase-82.0: per-series freshness SLA, in days, measured against the
# series' own newest observation DATE (not ingested_at). Values account for
# FRED's start-of-period dating plus real publication lag.
#
# phase-82.0 cycle-3 (Q/A finding 3): the daily bounds were widened 5 -> 12.
# Arming this gate makes `return 0` REACHABLE FOR THE FIRST TIME -- the guard
# was vacuous before, so preload_macro always cached. At 5 days the measured
# live headroom was ONE day (DGS10 newest 2026-07-30 = 4d), and FRED daily
# series do not publish on weekends or US market holidays. A long weekend plus
# a single missed cron run would have tripped the gate on a perfectly healthy
# feed, and because backtest_engine.py:308 DISCARDS preload_macro's return
# value, the only symptom would be a silent fall-through to the per-cutoff BQ
# query in cached_macro -- CLAUDE.md's ~40-minute path. 12 days spans a
# holiday weekend plus several missed runs while still catching a genuinely
# dead feed by three orders of magnitude (the observed failure was 212 days).
MACRO_SERIES_MAX_AGE_DAYS: dict[str, int] = {
    "DGS10": 12,      # daily (business days only; holiday-tolerant)
    "T10Y2Y": 12,     # daily
    "FEDFUNDS": 70,   # monthly, dated to month start
    "UMCSENT": 70,    # monthly
    "UNRATE": 75,     # monthly
    "CPIAUCSL": 80,   # monthly, longer release lag
    "GDP": 225,       # quarterly, dated to quarter start
}

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

    # Skip BQ query if all requested tickers are already preloaded
    if _prices_full and all(t in _prices_full for t in tickers):
        total = sum(len(df) for df in _prices_full.values())
        logger.info("Prices already preloaded (%d tickers, %d rows), skipping BQ query", len(_prices_full), total)
        return total

    # TODO (Phase 2.9): Add market filter when expanding to multi-market
    # e.g., WHERE ticker IN UNNEST(@tickers) AND market = @market AND ...
    query = f"""
        SELECT ticker, date, open, high, low, close, volume, market
        FROM `{_table("historical_prices")}`
        WHERE ticker IN UNNEST(@tickers)
          AND date >= @start AND date <= @end
          AND market = 'US'
        ORDER BY ticker, date ASC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        bigquery.ScalarQueryParameter("start", "STRING", start_date),
        bigquery.ScalarQueryParameter("end", "STRING", end_date),
    ])
    rows = list(_bq_client.query(query, job_config=job_config).result(timeout=120))

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

    # Skip BQ query if all requested tickers are already preloaded
    if _fundamentals_full and all(t in _fundamentals_full for t in tickers):
        total = sum(len(v) for v in _fundamentals_full.values())
        logger.info("Fundamentals already preloaded (%d tickers, %d rows), skipping BQ query", len(_fundamentals_full), total)
        return total

    # phase-53.3: explicit projection (not SELECT *) -- BQ bills for every column
    # scanned. These 12 are the only columns any consumer reads
    # (backend/backtest/historical_data.py .get()s + the ticker grouping key +
    # the report_date ORDER BY). Dropping the 4 never-read columns (filing_date,
    # ingested_at, market, currency) cuts bytes-scanned ~21% (655,079 -> 515,937,
    # dry-run measured) with byte-identical results. (A tighter 10-col set scans less
    # but drops sector/industry, which historical_data.py consumes -- a result change;
    # this 12-col set is the results-preserving projection.) historical_fundamentals is NOT
    # partitioned, so a WHERE/date filter would not prune -- projection is the only
    # correctness-preserving lever.
    query = f"""
        SELECT ticker, report_date, total_revenue, net_income, total_debt,
               total_equity, total_assets, operating_cash_flow, shares_outstanding,
               sector, industry, dividends_per_share
        FROM `{_table("historical_fundamentals")}`
        WHERE ticker IN UNNEST(@tickers)
        ORDER BY ticker, report_date DESC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
    ])
    rows = list(_bq_client.query(query, job_config=job_config).result(timeout=120))

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

    # Skip BQ query if macro data is already preloaded
    if _macro_full:
        total = sum(len(v) for v in _macro_full.values())
        logger.info("Macro already preloaded (%d series, %d rows), skipping BQ query", len(_macro_full), total)
        return total

    query = f"""
        SELECT series_id, value, date
        FROM `{_table("historical_macro")}`
        ORDER BY series_id, date DESC
    """
    rows = list(_bq_client.query(query).result(timeout=60))

    if not rows:
        logger.warning("preload_macro: 0 rows returned")
        return 0

    # phase-25.D7: refuse to cache stale macro -- prevents silent backtest
    # corruption when FRED ingestion stalls.
    # phase-82.0: evaluated PER SERIES. The previous implementation took a
    # single global max across every series, which meant one live daily series
    # (DGS10 updates every business day) satisfied the gate on behalf of a
    # series that had been dead for years. Any stale series now fails the gate.
    #
    # phase-82.0 cycle-2 (Q/A FAIL): `historical_macro.date` is a **STRING**
    # column, not DATE -- BQ schema ('date','STRING','REQUIRED'), declared at
    # scripts/migrations/migrate_backtest_data.py:68, and live rows come back as
    # e.g. '2023-07-03' (type str). The pre-existing phase-25.D7 gate tested
    # `isinstance(rd, date)`, which is FALSE for every production row, so
    # `max_date` stayed None, the staleness branch was skipped entirely, and
    # preload_macro cached whatever it was given. **That guard has never fired
    # in production since the day it was written.** It did not "refuse stale
    # macro"; it silently served it, which is the worse failure. The cycle-1
    # per-series rewrite inherited the same isinstance bug and was equally
    # vacuous. Dates are now COERCED before comparison, so the gate works on
    # the type the query actually returns.
    from datetime import date as _date, datetime as _dt

    def _coerce_date(v) -> Optional[_date]:
        """BQ returns this column as str; tests and other callers may pass
        date/datetime. Accept all three; anything unparseable is skipped."""
        if isinstance(v, _dt):
            return v.date()
        if isinstance(v, _date):
            return v
        if isinstance(v, str):
            try:
                return _date.fromisoformat(v[:10])
            except ValueError:
                return None
        return None

    today = _date.today()
    per_series_max: dict[str, _date] = {}
    unparsed = 0
    for r in rows:
        row = r if isinstance(r, dict) else dict(r)
        sid = row.get("series_id", "")
        rd = _coerce_date(row.get("date"))
        if rd is None:
            unparsed += 1
            continue
        cur = per_series_max.get(sid)
        if cur is None or rd > cur:
            per_series_max[sid] = rd

    # Fail CLOSED on a table we cannot evaluate. Previously an unreadable date
    # column silently disabled the entire gate; refusing is the safe direction
    # because the caller treats 0 as "no macro cached" and surfaces it.
    if not per_series_max and rows:
        logger.warning(
            "preload_macro: refusing to cache -- could not parse a usable date "
            "from any of %d rows (%d unparsed); the freshness gate cannot be "
            "evaluated, so the data is not trusted",
            len(rows),
            unparsed,
        )
        return 0

    stale: list[str] = []
    for sid, newest in sorted(per_series_max.items()):
        limit = MACRO_SERIES_MAX_AGE_DAYS.get(sid, MACRO_MAX_AGE_DAYS)
        age_days = (today - newest).days
        if age_days > limit:
            stale.append(f"{sid}(newest={newest.isoformat()} age={age_days}d limit={limit}d)")

    if stale:
        logger.warning(
            "preload_macro: stale data, refusing to cache -- %d of %d series "
            "past their per-series SLA: %s",
            len(stale),
            len(per_series_max),
            "; ".join(stale),
        )
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

def get_preloaded_tickers() -> set[str]:
    """Return set of tickers that were bulk-preloaded (fast path only)."""
    return set(_prices_full.keys())


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
    # phase-53.3: explicit projection (same 12 consumed columns as preload_fundamentals).
    # Correctness-preserving; the 30s timeout + cutoff WHERE + LIMIT 5 are unchanged.
    query = f"""
        SELECT ticker, report_date, total_revenue, net_income, total_debt,
               total_equity, total_assets, operating_cash_flow, shares_outstanding,
               sector, industry, dividends_per_share
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
