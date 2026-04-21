"""phase-9.1 Job-runtime primitives: heartbeat + idempotency.

Heartbeat: a context manager that emits a 'started' marker, records
duration, and swallows/logs exceptions while marking the run as failed.
Writes to an injectable sink (default: logger; production wires to BQ
`job_heartbeat` or a Slack channel).

Idempotency: `IdempotencyKey.seen(key)` / `.mark(key)` backed by an
injectable store (default: in-memory set). Job keys are typically
`{job_name}:{iso_date}` for daily or `{job_name}:{iso_week}` for weekly.

Pure. Fail-open. ASCII-only.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyStore:
    """In-memory default store. Production wires to BQ or Redis."""

    _seen: set[str] = field(default_factory=set)

    def seen(self, key: str) -> bool:
        return key in self._seen

    def mark(self, key: str) -> None:
        self._seen.add(key)


_GLOBAL_STORE = IdempotencyStore()


class IdempotencyKey:
    """Namespace for idempotency-key helpers."""

    @staticmethod
    def daily(job_name: str, day: str | None = None) -> str:
        d = day or datetime.now(timezone.utc).date().isoformat()
        return f"{job_name}:{d}"

    @staticmethod
    def weekly(job_name: str, iso_year_week: str | None = None) -> str:
        if iso_year_week is None:
            now = datetime.now(timezone.utc)
            iso = now.isocalendar()
            iso_year_week = f"{iso.year}-W{iso.week:02d}"
        return f"{job_name}:{iso_year_week}"

    @staticmethod
    def hourly(job_name: str, iso_hour: str | None = None) -> str:
        if iso_hour is None:
            now = datetime.now(timezone.utc)
            iso_hour = now.strftime("%Y-%m-%dT%H")
        return f"{job_name}:{iso_hour}"


@contextmanager
def heartbeat(
    job_name: str,
    *,
    idempotency_key: str | None = None,
    store: IdempotencyStore | None = None,
    sink: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Emit a 'started' event, run the block, emit 'ok' or 'failed'.

    If `idempotency_key` is set AND already marked in `store`, the block
    is skipped (yielding a dict with `skipped=True`).

    `sink` receives event dicts: `{job, status, duration_s, error?, idempotency_key?}`.
    Default sink = logger.info.
    """
    s = store or _GLOBAL_STORE
    sink_fn: Callable[[dict[str, Any]], None] = sink or (lambda evt: logger.info("job: %s", evt))

    state: dict[str, Any] = {
        "job": job_name,
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": idempotency_key,
        "skipped": False,
    }
    if idempotency_key is not None and s.seen(idempotency_key):
        state["status"] = "skipped_idempotent"
        state["skipped"] = True
        state["duration_s"] = 0.0
        sink_fn(dict(state))
        yield state
        return

    sink_fn(dict(state))
    t0 = time.monotonic()
    try:
        yield state
        state["status"] = "ok"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        logger.warning("job: %s failed: %r", job_name, exc)
    finally:
        state["duration_s"] = time.monotonic() - t0
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        if state["status"] == "ok" and idempotency_key is not None:
            s.mark(idempotency_key)
        sink_fn(dict(state))


__all__ = ["IdempotencyStore", "IdempotencyKey", "heartbeat"]
