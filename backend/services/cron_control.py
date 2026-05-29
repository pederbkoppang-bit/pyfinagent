"""
cron_control -- operator pause / resume / (paper-job) trigger for the BACKEND's
in-process APScheduler jobs (phase-49.2, P7 "cron enable+trigger").

Only the 2 backend-owned jobs are controllable in-process via the existing
`cron_dashboard_api._RUNNING_SCHEDULERS` registry (no new scheduler plumbing):
  - paper_trading_daily       -> the "main" scheduler (the daily trade cycle)
  - ticket_queue_process_batch -> the "queue" scheduler
The 11 slack_bot jobs + 6 launchd jobs are cross-process (a static manifest in
cron_dashboard_api) -> the API rejects them with 404.

pause/resume use APScheduler 3.x `pause_job`/`resume_job` (sets/clears
`next_run_time`), which is REVERSIBLE and preserves the job + its trigger --
distinct from paper_trading `/stop`'s `remove_job` (which deletes the job).

Audit: every control action appends a JSON line to
handoff/cron_control_audit.jsonl, mirroring kill_switch.py / risk_overrides.py
(Hasura "audit every invocation" guidance).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "cron_control_audit.jsonl"
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

# job_id -> registered scheduler name (see cron_dashboard_api._RUNNING_SCHEDULERS,
# populated by main.py:264 "main" + :317 "queue").
CONTROLLABLE: dict[str, str] = {
    "paper_trading_daily": "main",
    "ticket_queue_process_batch": "queue",
}


class CronControlError(Exception):
    """Raised when a job id is not controllable in-process (cross-process /
    unknown) or its scheduler/job is not currently live. The API maps this to
    HTTP 404."""


def is_controllable(job_id: str) -> bool:
    return job_id in CONTROLLABLE


def _append_audit(action: str, job_id: str, **fields: Any) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "job_id": job_id,
        **fields,
    }
    try:
        with _AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        logger.warning("cron_control: audit write failed: %s", e)


def _resolve_scheduler(job_id: str):
    if job_id not in CONTROLLABLE:
        raise CronControlError(
            f"'{job_id}' is not controllable in-process (cross-process or unknown job). "
            f"Controllable jobs: {sorted(CONTROLLABLE)}"
        )
    # Lazy import to avoid any import-time coupling with the API layer.
    from backend.api.cron_dashboard_api import get_registered_schedulers

    sched = get_registered_schedulers().get(CONTROLLABLE[job_id])
    if sched is None:
        raise CronControlError(
            f"scheduler '{CONTROLLABLE[job_id]}' for job '{job_id}' is not registered/running"
        )
    return sched


def status(job_id: str) -> dict:
    """Current paused/next-run state for a controllable job."""
    sched = _resolve_scheduler(job_id)
    job = sched.get_job(job_id)
    if job is None:
        return {"job_id": job_id, "exists": False, "paused": None, "next_run": None}
    nrt = getattr(job, "next_run_time", None)
    return {
        "job_id": job_id,
        "exists": True,
        "paused": nrt is None,
        "next_run": nrt.isoformat() if nrt is not None else None,
    }


def record_trigger(job_id: str, reason: str = "manual", result: str = "ok") -> None:
    """Audit a manual trigger. The trigger itself is performed by the API layer
    (it reuses paper_trading /run-now's guarded path); this just records it."""
    _append_audit("trigger", job_id, reason=reason, result=result)
    logger.info("cron_control: TRIGGERED %s (reason=%s, result=%s)", job_id, reason, result)


def pause(job_id: str, reason: str = "manual") -> dict:
    sched = _resolve_scheduler(job_id)
    from apscheduler.jobstores.base import JobLookupError

    try:
        sched.pause_job(job_id)
    except JobLookupError:
        raise CronControlError(f"job '{job_id}' not found on scheduler '{CONTROLLABLE[job_id]}'")
    _append_audit("pause", job_id, reason=reason)
    logger.info("cron_control: PAUSED %s (reason=%s)", job_id, reason)
    return status(job_id)


def resume(job_id: str, reason: str = "manual") -> dict:
    sched = _resolve_scheduler(job_id)
    from apscheduler.jobstores.base import JobLookupError

    try:
        sched.resume_job(job_id)
    except JobLookupError:
        raise CronControlError(f"job '{job_id}' not found on scheduler '{CONTROLLABLE[job_id]}'")
    _append_audit("resume", job_id, reason=reason)
    logger.info("cron_control: RESUMED %s (reason=%s)", job_id, reason)
    return status(job_id)
