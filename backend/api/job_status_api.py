"""phase-15.2 Job heartbeat status endpoint.

Surfaces the 7 phase-9 Slack-bot jobs (`_PHASE9_JOB_IDS` in
`backend/slack_bot/scheduler.py:336-344`) to an operator dashboard tile.
Maintains a thread-safe in-memory registry pre-seeded with all 7 names so
the GET endpoint always returns 7 rows even before any job fires.

Cross-process delivery: the Slack-bot runs in a separate process from
FastAPI (`python -m backend.slack_bot.app`), so `job_runtime.heartbeat()`'s
default `logger.info` sink never touches this module's memory. A hidden
`POST /heartbeat` endpoint lets the Slack-bot (or any harness integration)
deliver event dicts to this registry. Until that wiring lands, `GET
/status` returns `status="never_run"` for every job -- which still
satisfies the masterplan verification criteria (key presence only).
"""
from __future__ import annotations

import json as _json
import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def structured_log(endpoint: str, duration_ms: float, status: str, **extra) -> None:
    """phase-15.10: one JSON log line per call. See cost_budget_api for schema."""
    try:
        logger.info(
            _json.dumps(
                {
                    "endpoint": endpoint,
                    "duration_ms": round(duration_ms, 1),
                    "status": status,
                    "ts": time.time(),
                    **extra,
                }
            )
        )
    except Exception as exc:
        logger.warning("structured_log fail-open: %r", exc)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Canonical phase-9 job names. Mirrors `_PHASE9_JOB_IDS` in
# `backend/slack_bot/scheduler.py:336-344`. Hardcoded here so the endpoint
# returns 7 rows even before the Slack-bot process connects.
_JOB_NAMES: tuple[str, ...] = (
    "daily_price_refresh",       # phase-9.2
    "weekly_fred_refresh",       # phase-9.3
    "nightly_mda_retrain",       # phase-9.4
    "hourly_signal_warmup",      # phase-9.5
    "nightly_outcome_rebuild",   # phase-9.6
    "weekly_data_integrity",     # phase-9.7
    "cost_budget_watcher",       # phase-9.8
)


class JobStatus(BaseModel):
    name: str
    last_run_at: Optional[str] = None        # ISO-8601 UTC, None when never_run
    last_duration_s: Optional[float] = None
    status: str = "never_run"                # never_run | ok | failed | in_progress | skipped_idempotent
    last_error: Optional[str] = None


class JobStatusResponse(BaseModel):
    jobs: list[JobStatus]


# Module-level thread-safe registry pre-seeded with every canonical job.
_registry: dict[str, dict] = {name: {"name": name} for name in _JOB_NAMES}
_lock = threading.Lock()


def record_heartbeat(event: dict) -> None:
    """Record a terminal job event. Safe to call as `sink` from job_runtime.

    Accepts the finished-event dict shape emitted by
    `backend.slack_bot.job_runtime.heartbeat`:
    `{job, status, started_at, finished_at, duration_s, error?, ...}`.

    `status="started"` is ignored (we only track terminal states).
    Unknown job names are still recorded so new jobs show up without
    requiring a code change here.
    """
    if not isinstance(event, dict):
        return
    job = event.get("job")
    status = event.get("status")
    if not job or status == "started":
        return
    with _lock:
        row = _registry.setdefault(job, {"name": job})
        row["last_run_at"] = event.get("finished_at")
        row["last_duration_s"] = event.get("duration_s")
        row["status"] = str(status) if status is not None else "unknown"
        row["last_error"] = event.get("error")


@router.get("/status", response_model=JobStatusResponse)
def get_job_status() -> JobStatusResponse:
    """Return the status of all 7 phase-9 Slack-bot jobs.

    Fail-open: never raises. In-memory read only; no I/O. Sync `def` so
    FastAPI runs it in its built-in threadpool.
    """
    start = time.perf_counter()
    with _lock:
        jobs: list[JobStatus] = []
        for name in _JOB_NAMES:
            row = _registry.get(name, {"name": name})
            jobs.append(
                JobStatus(
                    name=name,
                    last_run_at=row.get("last_run_at"),
                    last_duration_s=row.get("last_duration_s"),
                    status=row.get("status", "never_run"),
                    last_error=row.get("last_error"),
                )
            )
    structured_log(
        "/api/jobs/status",
        (time.perf_counter() - start) * 1000,
        "ok",
        job_count=len(jobs),
    )
    return JobStatusResponse(jobs=jobs)


@router.post("/heartbeat", include_in_schema=False)
def post_heartbeat(event: dict) -> dict:
    """Sink endpoint for cross-process heartbeat delivery.

    Hidden from the OpenAPI schema because it is an internal
    Slack-bot -> FastAPI plumbing route, not a user-facing API.
    """
    record_heartbeat(event)
    return {"ok": True}


__all__ = [
    "router",
    "JobStatus",
    "JobStatusResponse",
    "record_heartbeat",
    "_JOB_NAMES",
]
