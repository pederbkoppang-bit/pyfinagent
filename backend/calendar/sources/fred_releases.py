"""phase-6.6 FRED macro-release calendar adapter.

Endpoint: `GET https://api.stlouisfed.org/fred/releases/dates?file_type=json
  &api_key=<key>&realtime_start=YYYY-MM-DD&realtime_end=YYYY-MM-DD`.

Response: `release_dates` array with `{release_id, release_name, date}`.
We filter to the economic releases that matter for event-driven trading:
CPI, PPI, Employment Situation (NFP + U-rate), GDP, Retail Sales. The
release_id whitelist is documented below and can be extended without
code changes elsewhere.

Empty `FRED_API_KEY` -> fetch() yields nothing.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Iterable

import requests

from backend.calendar.registry import register
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.stlouisfed.org/fred/releases/dates"
_TIMEOUT_SEC = 15.0

# FRED release_id -> (event_type, display_name)
# Sourced from fred.stlouisfed.org/releases with IDs verified via the public
# web UI. NOT exhaustive -- extend as needed without changing schema.
_RELEASE_WHITELIST: dict[int, tuple[str, str]] = {
    10: ("cpi", "Consumer Price Index"),
    20: ("ppi", "Producer Price Index"),
    50: ("nfp", "Employment Situation"),  # includes NFP + U-rate
    53: ("gdp", "Gross Domestic Product"),
    82: ("retail_sales", "Advance Monthly Sales for Retail and Food Services"),
    91: ("unemployment", "Weekly Unemployment Insurance Claims"),
}


class FredReleasesSource:
    name = "fred_releases"

    def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
        settings = get_settings()
        api_key = getattr(settings, "fred_api_key", "")
        if not api_key:
            logger.debug("FRED_API_KEY empty; skipping fred releases fetch")
            return
        params = {
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": from_date.isoformat(),
            "realtime_end": to_date.isoformat(),
            "limit": 1000,
        }
        try:
            resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_SEC)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("fred releases fetch failed: %r", exc)
            return

        for row in payload.get("release_dates", []) or []:
            try:
                rid = int(row.get("release_id", -1))
            except (TypeError, ValueError):
                continue
            mapping = _RELEASE_WHITELIST.get(rid)
            if mapping is None:
                continue
            event_type, display_name = mapping
            release_date_str = str(row.get("date") or "")
            if not release_date_str:
                continue
            try:
                d = datetime.fromisoformat(release_date_str).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            # FRED dates are calendar days, no time-of-day. BLS releases at
            # 8:30 AM ET -> 13:30 UTC (during EST) / 12:30 UTC (during EDT).
            # We standardize at 13:30 UTC as the conservative choice.
            scheduled = d.replace(hour=13, minute=30)
            yield {
                "event_type": event_type,
                "ticker": None,
                "scheduled_at": scheduled.isoformat(),
                "window": "pre_open",  # BLS prints are pre-open US cash
                "fiscal_period_end": None,
                "source": "fred_releases",
                "confidence": "confirmed",
                "metadata": {
                    "release_id": rid,
                    "release_name": display_name,
                    "fred_release_name": row.get("release_name"),
                },
            }


register(FredReleasesSource())
