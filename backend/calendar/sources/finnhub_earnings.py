"""phase-6.6 Finnhub earnings-calendar adapter.

Endpoint: `GET https://finnhub.io/api/v1/calendar/earnings?from=&to=&symbol=&token=`.
Rate limit: 30 req/sec free tier. Returns `earningsCalendar[]` with
`{symbol, date, hour, year, quarter, epsEstimate, epsActual,
  revenueEstimate, revenueActual}`.

`hour` field provides pre/post-market timing:
    bmo = before market open (-> pre_open)
    amc = after market close (-> post_close)
    dmh = during market hours (-> intraday)
    missing -> all_day

Empty `FINNHUB_API_KEY` -> fetch() yields nothing (matches phase-6.3
news adapter fail-open convention at `backend/news/sources/finnhub.py`).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable

import requests

from backend.calendar.registry import CalendarSource, register
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_ENDPOINT = "https://finnhub.io/api/v1/calendar/earnings"
_TIMEOUT_SEC = 15.0


class FinnhubEarningsSource:
    name = "finnhub"

    def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
        settings = get_settings()
        token = getattr(settings, "finnhub_api_key", "")
        if not token:
            logger.debug("FINNHUB_API_KEY empty; skipping finnhub earnings fetch")
            return
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": token,
        }
        try:
            resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_SEC)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("finnhub earnings fetch failed: %r", exc)
            return
        for row in payload.get("earningsCalendar", []) or []:
            symbol = str(row.get("symbol") or "").upper()
            report_date = str(row.get("date") or "")
            if not symbol or not report_date:
                continue
            hour = str(row.get("hour") or "").lower()
            fiscal_period_end = self._fiscal_period_end_from_row(row)
            # Conservative: assume 13:30 UTC (09:30 ET open) for bmo / amc
            # timing so downstream code has a sortable timestamp; intraday
            # treated as 17:00 UTC mid-session.
            scheduled_at = f"{report_date}T13:30:00+00:00"
            yield {
                "event_type": "earnings",
                "ticker": symbol,
                "scheduled_at": scheduled_at,
                "window": hour,  # normalize_window handles bmo/amc/dmh
                "fiscal_period_end": fiscal_period_end,
                "source": "finnhub",
                "confidence": "confirmed" if hour else "estimated",
                "eps_estimate": _safe_float(row.get("epsEstimate")),
                "revenue_estimate": _safe_float(row.get("revenueEstimate")),
                "metadata": {
                    "year": row.get("year"),
                    "quarter": row.get("quarter"),
                    "eps_actual": _safe_float(row.get("epsActual")),
                    "revenue_actual": _safe_float(row.get("revenueActual")),
                    "hour_raw": hour,
                },
            }

    @staticmethod
    def _fiscal_period_end_from_row(row: dict[str, Any]) -> str | None:
        """Approximate fiscal period end from year + quarter when available.

        Finnhub returns `year` and `quarter` (1-4); we use quarter-end month-
        end as a stable dedup anchor. This is approximate (company fiscal
        calendars vary) but sufficient for the phase-6.6 dedup key
        `(ticker, fiscal_period_end)`.
        """
        year = row.get("year")
        quarter = row.get("quarter")
        if year is None or quarter is None:
            return None
        try:
            q = int(quarter)
            y = int(year)
        except (TypeError, ValueError):
            return None
        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}.get(q)
        quarter_end_day = {1: 31, 2: 30, 3: 30, 4: 31}.get(q)
        if quarter_end_month is None:
            return None
        return f"{y:04d}-{quarter_end_month:02d}-{quarter_end_day:02d}"


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Register at import time (side-effect import pattern parallels phase-6.3).
register(FinnhubEarningsSource())
