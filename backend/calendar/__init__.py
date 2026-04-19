"""phase-6.6 calendar watcher -- package entry.

Parallel to `backend/news/` but with a domain-specific data model
(`CalendarEvent` not `NormalizedArticle`) and Protocol (`CalendarSource`
not `NewsSource`). See `handoff/archive/phase-6.6/` for the research
brief that motivated the separate-module architecture.
"""
from backend.calendar.registry import (
    CalendarSource,
    register,
    get_sources,
    clear_registry,
)
from backend.calendar.normalize import (
    compute_event_id,
    normalize_window,
)
from backend.calendar.blackout import compute_fomc_blackout
from backend.calendar.watcher import (
    CalendarEvent,
    CalendarFetchReport,
    normalize_event,
    run_once,
)

# side-effect import registers finnhub_earnings / fed_scrape / fred_releases
from backend.calendar import sources as _sources  # noqa: F401

__all__ = [
    "CalendarEvent",
    "CalendarSource",
    "CalendarFetchReport",
    "register",
    "get_sources",
    "clear_registry",
    "compute_event_id",
    "normalize_window",
    "compute_fomc_blackout",
    "normalize_event",
    "run_once",
]
