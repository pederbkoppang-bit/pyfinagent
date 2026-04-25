"""phase-6.6 calendar source Protocol + decorator registry.

Structurally parallel to `backend/news/registry.py:31-41` but with a
different Protocol because `NewsSource.fetch()` returns article dicts
and `CalendarSource.fetch(from_date, to_date)` returns event dicts
within a date window.

Keeping the registries separate avoids entangling pipelines: the archive
hook, the news cron, and the calendar cron each operate on their own
domain without cross-talk.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class CalendarSource(Protocol):
    name: str

    def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
        """Yield raw event dicts in [from_date, to_date]. No normalization required."""
        ...


_REGISTRY: list[CalendarSource] = []


def register(source: CalendarSource) -> CalendarSource:
    """Decorator / function: register a CalendarSource instance.

    Also usable on a class (must be instantiated at module import time):
        `register(MyCalendarSource())`. Returns the source unchanged so it
    can be chained or used inline.
    """
    if not isinstance(source, CalendarSource):  # pragma: no cover
        raise TypeError(
            f"register() expected CalendarSource, got {type(source).__name__}"
        )
    if any(s.name == source.name for s in _REGISTRY):
        # idempotent: second register of the same-named source replaces the first
        _REGISTRY[:] = [s for s in _REGISTRY if s.name != source.name]
    _REGISTRY.append(source)
    return source


def get_sources() -> list[CalendarSource]:
    return list(_REGISTRY)


def clear_registry() -> None:
    """Test helper: empty the registry. Do not call from production code."""
    _REGISTRY.clear()


__all__ = ["CalendarSource", "register", "get_sources", "clear_registry"]
