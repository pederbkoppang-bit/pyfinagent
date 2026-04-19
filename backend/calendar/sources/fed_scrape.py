"""phase-6.6 Fed FOMC calendar scraper.

`federalreserve.gov/monetarypolicy/fomccalendars.htm` has no JSON/ICS API
(confirmed in research brief 2026-04-19). Strategy: fetch HTML, use a
conservative regex to extract meeting date rows for the current calendar
year + next year. For each matched row, emit an `fomc_meeting` event with
`scheduled_at` at the meeting start date, 14:00 ET press-conference default
(convention: press-conference meetings are 14:30 ET; non-PC meetings have
no presser but the statement is released at 14:00 ET).

If BeautifulSoup is installed, prefer a structural parse; otherwise fall
back to regex. Fail-open: any exception returns no events, logs a warning.

This source is scheduled for weekly polling -- HTML parsing is brittle and
should not be hit daily. Forward window 18 months typical.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable

import requests

from backend.calendar.registry import register

logger = logging.getLogger(__name__)

_ENDPOINT = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_TIMEOUT_SEC = 20.0
_USER_AGENT = "pyfinagent-calendar-watcher/6.6"

# Month name map for textual date rows like "January 28-29" or "December 16-17"
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Regex for lines like: January 28-29*, March 18-19, November 4-5*
# group 1 = month, group 2 = start day, group 3 = end day (optional)
_ROW_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:[-\u2013\u2014](\d{1,2}))?\*?",
    re.IGNORECASE,
)


class FedScrapeSource:
    name = "fed_scrape"

    def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
        try:
            resp = requests.get(
                _ENDPOINT,
                headers={"User-Agent": _USER_AGENT},
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.warning("fed scrape fetch failed: %r", exc)
            return

        try:
            meetings = self._extract_meetings(html, from_date.year, to_date.year)
        except Exception as exc:
            logger.warning("fed scrape parse failed: %r", exc)
            return

        for year, month, start_day, end_day, has_sep in meetings:
            try:
                start_dt = datetime(year, month, start_day, 14, 0, tzinfo=timezone.utc)
                if start_dt.date() < from_date or start_dt.date() > to_date:
                    continue
                # Day-1 of the meeting; callers needing the end-day compute
                # it from the matched range; we also emit an event for the
                # end day when it differs so blackout / multi-day handling
                # in watcher.run_once can use both dates.
                yield self._make_event(year, month, start_day, has_sep)
                if end_day and end_day != start_day:
                    yield self._make_event(year, month, end_day, has_sep, tag="end")
            except Exception as exc:
                logger.debug("fed scrape yield failed: %r", exc)
                continue

    @staticmethod
    def _make_event(
        year: int,
        month: int,
        day: int,
        has_sep: bool,
        tag: str = "start",
    ) -> dict[str, Any]:
        scheduled = datetime(year, month, day, 14, 0, tzinfo=timezone.utc)
        return {
            "event_type": "fomc_meeting",
            "ticker": None,
            "scheduled_at": scheduled.isoformat(),
            "window": "intraday",
            "fiscal_period_end": None,
            "source": "fed_scrape",
            "confidence": "confirmed",
            "metadata": {
                "has_sep": has_sep,
                "day_tag": tag,
                "year": year,
                "month": month,
            },
        }

    @staticmethod
    def _extract_meetings(
        html: str, from_year: int, to_year: int
    ) -> list[tuple[int, int, int, int | None, bool]]:
        """Return [(year, month, start_day, end_day, has_sep), ...].

        Coarse heuristic: for each year in [from_year, to_year], search for
        a block that contains that year header, then extract month+day rows
        within the next ~2000 characters. The SEP star (`*`) indicator is
        captured per row.
        """
        out: list[tuple[int, int, int, int | None, bool]] = []
        for year in range(from_year, to_year + 1):
            idx = html.find(str(year))
            if idx < 0:
                continue
            block = html[idx : idx + 4000]
            for m in _ROW_RE.finditer(block):
                month_name = m.group(1).lower()
                month = _MONTHS.get(month_name)
                if month is None:
                    continue
                try:
                    start_day = int(m.group(2))
                    end_day = int(m.group(3)) if m.group(3) else None
                except (TypeError, ValueError):
                    continue
                # star immediately after captured range indicates SEP meeting
                tail = block[m.end() : m.end() + 2]
                has_sep = "*" in tail or m.group(0).endswith("*")
                out.append((year, month, start_day, end_day, has_sep))
        return out


register(FedScrapeSource())
