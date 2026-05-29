"""
fx_rates -- FX rate layer for multi-currency portfolio valuation (phase-50.1).

The FREE foundation for all multi-currency work (50.2 accounting, 50.5 backtest).
Sources EUR/USD + KRW/USD (and other market currencies) from yfinance, with a
FRED fallback, an api_cache TTL for the live daily mark, and a BigQuery
`historical_fx_rates` table for point-in-time / backtest reads.

DIRECTION (locked by research; KRW inversion is the #1 pitfall):
  - yfinance `EURUSD=X` = USD per 1 EUR  (~1.16)  -> usd_value(EUR) directly
  - yfinance `KRW=X`    = KRW per 1 USD  (~1300)  -> usd_value(KRW) = 1/rate
  - FRED `DEXUSEU` = "USD to One Euro" (same dir as EURUSD=X)
  - FRED `DEXKOUS` = "Won to One USD"  (same dir as KRW=X)
  NEVER use `KRWUSD=X` (inverse).

CANONICAL MODEL: usd_value(ccy) = USD value of 1 unit of ccy (USD=1.0).
  get_fx_rate(from, to, date) = usd_value(from) / usd_value(to).
  from == to  ->  1.0  (keeps the US-only / USD-only path byte-identical).

Storage: BQ row pair="{CCY}USD" holds usd_value(CCY) (USD per 1 unit), `date`
as STRING, point-in-time as-of read (`WHERE date<=? ORDER BY date DESC LIMIT 1`,
forward-fills weekend/holiday gaps). Daily close = mid rate (NAV standard).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from backend.backtest import markets

logger = logging.getLogger(__name__)

# currency -> (yfinance ticker, invert) where invert=True means the ticker
# quotes CCY-per-USD so usd_value = 1/rate; invert=False means USD-per-CCY.
_CCY_YF: dict[str, tuple[str, bool]] = {
    "EUR": ("EURUSD=X", False),   # USD per EUR
    "GBP": ("GBPUSD=X", False),   # USD per GBP
    "KRW": ("KRW=X", True),       # KRW per USD
    "NOK": ("NOK=X", True),       # NOK per USD
    "CAD": ("CAD=X", True),       # CAD per USD
}
# currency -> (FRED series, invert) for the robustness fallback.
_CCY_FRED: dict[str, tuple[str, bool]] = {
    "EUR": ("DEXUSEU", False),    # USD per EUR
    "KRW": ("DEXKOUS", True),     # KRW per USD
    "CAD": ("DEXCAUS", True),     # CAD per USD
    "NOK": ("DEXNOUS", True),     # NOK per USD
}

_CACHE_TTL = 6 * 60 * 60  # 6h live-mark cache; FX moves slowly intraday for a daily mark


def market_currency(market: str) -> str:
    """ISO currency for a market code (delegates to markets.MARKET_CONFIG)."""
    return markets.get_market_config(market)["currency"]


def _bq():
    """Lazy BigQuery client; None on any failure (fx_rates degrades to live fetch)."""
    try:
        from backend.config.settings import get_settings
        from google.cloud import bigquery
        s = get_settings()
        return bigquery.Client(project=s.gcp_project_id), s.bq_dataset_reports
    except Exception as e:
        logger.warning("fx_rates: BQ client unavailable: %s", e)
        return None, None


def _pair(ccy: str) -> str:
    return f"{ccy.upper()}USD"


# ── live + historical usd_value(ccy) ────────────────────────────────
def _usd_value_live(ccy: str) -> Optional[float]:
    """USD value of 1 unit of ccy, fetched live. yfinance primary, FRED fallback."""
    ccy = ccy.upper()
    if ccy == "USD":
        return 1.0
    # cache
    try:
        from backend.services.api_cache import get_api_cache
        cache = get_api_cache()
        ck = f"fx:usd_value:{ccy}"
        hit = cache.get(ck)
        if hit is not None:
            return hit
    except Exception:
        cache, ck = None, None
    val = _fetch_yf(ccy)
    if val is None:
        val = _fetch_fred(ccy)
    if val is not None:
        if cache and ck:
            try:
                cache.set(ck, val, _CACHE_TTL)
            except Exception:
                pass
        # write-through: today's live mark becomes tomorrow's history
        _persist(ccy, datetime.now(timezone.utc).date().isoformat(), val, "yfinance/fred-live")
    return val


def _fetch_yf(ccy: str) -> Optional[float]:
    spec = _CCY_YF.get(ccy)
    if not spec:
        return None
    ticker, invert = spec
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty:
            return None
        rate = float(hist["Close"].dropna().iloc[-1])
        if rate <= 0:
            return None
        return (1.0 / rate) if invert else rate
    except Exception as e:
        logger.warning("fx_rates: yfinance fetch %s failed: %s", ccy, e)
        return None


def _fetch_fred(ccy: str) -> Optional[float]:
    spec = _CCY_FRED.get(ccy)
    if not spec:
        return None
    series, invert = spec
    key = os.getenv("FRED_API_KEY")
    if not key:
        return None
    try:
        import requests
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series, "api_key": key, "file_type": "json",
                  "sort_order": "desc", "limit": "5"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        for obs in r.json().get("observations", []):
            v = obs.get("value")
            if v and v != ".":
                rate = float(v)
                if rate > 0:
                    return (1.0 / rate) if invert else rate
        return None
    except Exception as e:
        logger.warning("fx_rates: FRED fetch %s failed: %s", ccy, e)
        return None


def _usd_value_asof(ccy: str, date: str) -> Optional[float]:
    """USD value of 1 unit of ccy as of `date` (point-in-time, no look-ahead)."""
    ccy = ccy.upper()
    if ccy == "USD":
        return 1.0
    client, dataset = _bq()
    if client is None:
        return _usd_value_live(ccy)  # degrade
    try:
        from backend.config.settings import get_settings
        proj = get_settings().gcp_project_id
        sql = (
            f"SELECT rate FROM `{proj}.{dataset}.historical_fx_rates` "
            f"WHERE pair=@pair AND date<=@d ORDER BY date DESC LIMIT 1"
        )
        from google.cloud import bigquery
        job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("pair", "STRING", _pair(ccy)),
            bigquery.ScalarQueryParameter("d", "STRING", date),
        ]))
        rows = list(job.result(timeout=30))
        if rows:
            return float(rows[0]["rate"])
        return _usd_value_live(ccy)  # not yet backfilled -> live
    except Exception as e:
        logger.warning("fx_rates: as-of query %s@%s failed: %s", ccy, date, e)
        return _usd_value_live(ccy)


def get_fx_rate(from_ccy: str, to_ccy: str, date: Optional[str] = None) -> Optional[float]:
    """FX rate to convert 1 unit of from_ccy into to_ccy.

    date=None -> live daily mark; date=ISO str -> point-in-time as-of.
    from_ccy == to_ccy -> 1.0 (US-only/USD path byte-identical). Returns None
    only if a non-trivial rate genuinely cannot be sourced.
    """
    f, t = from_ccy.upper(), to_ccy.upper()
    if f == t:
        return 1.0
    uf = _usd_value_live(f) if date is None else _usd_value_asof(f, date)
    ut = _usd_value_live(t) if date is None else _usd_value_asof(t, date)
    if uf is None or ut is None or ut == 0:
        return None
    return uf / ut


# ── persistence + backfill ──────────────────────────────────────────
def _persist(ccy: str, date: str, usd_value: float, source: str) -> None:
    client, dataset = _bq()
    if client is None:
        return
    try:
        from backend.config.settings import get_settings
        proj = get_settings().gcp_project_id
        table = f"{proj}.{dataset}.historical_fx_rates"
        client.insert_rows_json(table, [{
            "pair": _pair(ccy), "date": date, "rate": float(usd_value), "source": source,
        }])
    except Exception as e:
        logger.warning("fx_rates: persist %s@%s failed: %s", ccy, date, e)


def backfill_fx(currencies: list[str], start: str, end: str) -> int:
    """Backfill usd_value history for `currencies` over [start, end] from yfinance.
    Writes one row per (pair, date). Returns the row count written."""
    written = 0
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning("fx_rates: yfinance import failed: %s", e)
        return 0
    client, dataset = _bq()
    if client is None:
        return 0
    from backend.config.settings import get_settings
    proj = get_settings().gcp_project_id
    table = f"{proj}.{dataset}.historical_fx_rates"
    for ccy in currencies:
        ccy = ccy.upper()
        if ccy == "USD":
            continue
        spec = _CCY_YF.get(ccy)
        if not spec:
            logger.warning("fx_rates: no yf ticker for %s -- skipping backfill", ccy)
            continue
        ticker, invert = spec
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            close = df["Close"]
            # single-ticker yf.download can return a 1-col (MultiIndex) DataFrame;
            # squeeze to a Series so .items() yields (date_index, value), not (col, series).
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            rows = []
            for idx, raw in close.dropna().items():
                rate = float(raw)
                if rate <= 0:
                    continue
                usd_value = (1.0 / rate) if invert else rate
                d = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                rows.append({"pair": _pair(ccy), "date": d, "rate": usd_value, "source": "yfinance-backfill"})
            if rows:
                client.insert_rows_json(table, rows)
                written += len(rows)
                logger.info("fx_rates: backfilled %d %s rows", len(rows), _pair(ccy))
        except Exception as e:
            logger.warning("fx_rates: backfill %s failed: %s", ccy, e)
    return written
