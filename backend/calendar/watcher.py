"""phase-6.6 calendar watcher -- orchestrator + CalendarEvent shape.

Keeps news pipeline's data model isolated (`NewsSource` / `NormalizedArticle`
in `backend/news/`). The calendar pipeline has its own Protocol, Registry,
and event TypedDict because event semantics do not match article semantics
(see research_brief: `backend/news/registry.py:31-41` Protocol is a
structural mismatch for calendar sources).

`run_once(days_forward=90, days_backward=0, dry_run=False)` -- orchestrates
all registered sources over the given date range, dedups by `event_id`,
applies FOMC blackout windows, returns a `CalendarFetchReport`. Fail-open:
one source raising does not stop the others.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, TypedDict

from backend.calendar.registry import get_sources
from backend.calendar.normalize import (
    compute_event_id,
    normalize_window,
)
from backend.calendar.blackout import compute_fomc_blackout

logger = logging.getLogger(__name__)


class CalendarEvent(TypedDict, total=False):
    """BQ `pyfinagent_data.calendar_events` row shape.

    Fields match the schema from `scripts/migrations/add_calendar_events_schema.py`.
    `total=False` so sources can omit optional fields; `normalize_event()`
    fills defaults.
    """

    event_id: str
    event_type: str  # fomc_meeting | earnings | cpi | ppi | nfp | ...
    ticker: str | None
    scheduled_at: str  # ISO-8601 UTC
    window: str  # pre_open | post_close | intraday | all_day
    fiscal_period_end: str | None  # ISO date str
    source: str
    confidence: str  # confirmed | estimated | unscheduled
    blackout_start: str | None
    blackout_end: str | None
    eps_estimate: float | None
    revenue_estimate: float | None
    fetched_at: str
    metadata: dict[str, Any]


_FOMC_EVENT_TYPES = frozenset(
    {"fomc_meeting", "fomc_minutes", "fomc_sep", "fed_speech"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_event(raw: dict[str, Any]) -> CalendarEvent:
    """Fill defaults + compute event_id if missing + normalize window."""
    event_type = str(raw.get("event_type") or "unknown")
    ticker = raw.get("ticker")
    scheduled_at = str(raw.get("scheduled_at") or _now_iso())
    fiscal_period_end = raw.get("fiscal_period_end")
    window = normalize_window(raw.get("window") or raw.get("hour"))

    eid = raw.get("event_id") or compute_event_id(
        event_type=event_type,
        ticker=ticker,
        fiscal_period_end=fiscal_period_end,
        scheduled_at=scheduled_at,
    )

    return CalendarEvent(
        event_id=str(eid),
        event_type=event_type,
        ticker=ticker if ticker is not None else None,
        scheduled_at=scheduled_at,
        window=window,
        fiscal_period_end=fiscal_period_end,
        source=str(raw.get("source") or "unknown"),
        confidence=str(raw.get("confidence") or "estimated"),
        blackout_start=raw.get("blackout_start"),
        blackout_end=raw.get("blackout_end"),
        eps_estimate=raw.get("eps_estimate"),
        revenue_estimate=raw.get("revenue_estimate"),
        fetched_at=_now_iso(),
        metadata=dict(raw.get("metadata") or {}),
    )


@dataclass
class CalendarFetchReport:
    n_events: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    events: list[CalendarEvent] = field(default_factory=list)
    dry_run: bool = False


def _apply_blackouts(events: list[CalendarEvent]) -> None:
    """For FOMC meeting pairs, compute blackout window in-place."""
    meetings_by_date: dict[str, CalendarEvent] = {}
    for ev in events:
        if ev.get("event_type") == "fomc_meeting":
            meetings_by_date[str(ev.get("scheduled_at", ""))[:10]] = ev
    sorted_keys = sorted(meetings_by_date.keys())
    for i, k in enumerate(sorted_keys):
        ev = meetings_by_date[k]
        try:
            start_dt = datetime.fromisoformat(str(ev["scheduled_at"]))
            end_dt = start_dt
            # heuristic: FOMC meetings are often 2-day Tue-Wed. If the next
            # entry in the list is the following calendar day, treat as the
            # same meeting's second day and skip its own blackout computation.
            if i + 1 < len(sorted_keys):
                next_start = datetime.fromisoformat(
                    str(meetings_by_date[sorted_keys[i + 1]]["scheduled_at"])
                )
                if (next_start - start_dt) <= timedelta(days=1):
                    end_dt = next_start
            bs, be = compute_fomc_blackout(start_dt, end_dt)
            ev["blackout_start"] = bs.isoformat()
            ev["blackout_end"] = be.isoformat()
        except Exception as exc:  # pragma: no cover
            logger.debug("blackout compute failed for %s: %r", k, exc)


def run_once(
    days_forward: int = 90,
    days_backward: int = 0,
    dry_run: bool = False,
    sources: Iterable[str] | None = None,
) -> CalendarFetchReport:
    """Run all registered calendar sources and merge results.

    Dedups by `event_id`. Applies FOMC blackout windows. Fail-open per source.
    `dry_run=True` does not skip source.fetch() -- the sources themselves
    honor their own dry-run if implemented; this flag is reflected in the
    returned report for downstream BQ writers.
    """
    report = CalendarFetchReport(dry_run=dry_run)
    today = date.today()
    from_date = today - timedelta(days=days_backward)
    to_date = today + timedelta(days=days_forward)

    by_id: dict[str, CalendarEvent] = {}
    registered = get_sources()
    requested = set(sources) if sources is not None else None

    for source_obj in registered:
        if requested is not None and source_obj.name not in requested:
            continue
        try:
            raws = list(source_obj.fetch(from_date, to_date))
        except Exception as exc:
            report.errors.append({"source": source_obj.name, "error": repr(exc)})
            logger.warning("calendar source %s failed: %r", source_obj.name, exc)
            continue
        for raw in raws:
            try:
                ev = normalize_event(raw)
            except Exception as exc:
                report.errors.append(
                    {"source": source_obj.name, "error": f"normalize: {exc!r}"}
                )
                continue
            eid = str(ev["event_id"])
            if eid in by_id:
                by_id[eid]["metadata"].update(ev.get("metadata", {}))
                by_id[eid]["metadata"].setdefault("merged_from", []).append(
                    source_obj.name
                )
            else:
                by_id[eid] = ev

    events = list(by_id.values())
    _apply_blackouts(events)

    report.events = events
    report.n_events = len(events)
    for ev in events:
        report.by_type[ev.get("event_type", "?")] = (
            report.by_type.get(ev.get("event_type", "?"), 0) + 1
        )
        report.by_source[ev.get("source", "?")] = (
            report.by_source.get(ev.get("source", "?"), 0) + 1
        )

    logger.info(
        "calendar.run_once n_events=%d by_type=%s errors=%d",
        report.n_events,
        report.by_type,
        len(report.errors),
    )
    return report
