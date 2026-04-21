"""phase-9.3 weekly FRED macro refresh. Idempotent by iso week."""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "weekly_fred_refresh"
_DEFAULT_SERIES = ["DGS10", "DGS2", "VIXCLS", "DFF", "UNRATE", "CPIAUCSL"]


def run(
    *,
    series: list[str] | None = None,
    fetch_fn: Callable[[list[str]], dict] | None = None,
    write_fn: Callable[[dict], int] | None = None,
    store: IdempotencyStore | None = None,
    iso_year_week: str | None = None,
) -> dict[str, Any]:
    key = IdempotencyKey.weekly(JOB_NAME, iso_year_week=iso_year_week)
    result: dict[str, Any] = {"written": 0, "key": key, "skipped": False}
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        universe = series or _DEFAULT_SERIES
        fetched = (fetch_fn or _default_fetch)(universe)
        n = (write_fn or _default_write)(fetched)
        result["written"] = int(n)
        result["series"] = universe
    return result


def _default_fetch(series: list[str]) -> dict[str, Any]:
    return {s: [] for s in series}  # injected in tests; production wraps fredapi


def _default_write(rows: dict[str, Any]) -> int:
    return len(rows)


__all__ = ["run", "JOB_NAME"]
