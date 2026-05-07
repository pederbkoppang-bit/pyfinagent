"""phase-23.2.23 Cron / Logs operator dashboard endpoints.

Two read-only endpoints powering frontend/src/app/cron/page.tsx:

- `GET /api/jobs/all` -- merges live APScheduler introspection with a
  static manifest of jobs that live in other processes (slack_bot) or
  in launchd. Cross-process IPC is overkill for a single-developer
  local app per the phase-23.2.23 research brief.
- `GET /api/logs/tail?log=<key>&lines=<n>` -- safe tail-read of an
  allowlisted log file. Path traversal is impossible because the
  client only ever passes a curated KEY; the server resolves it to a
  fixed Path. lines is clamped to [10, 1000].

Both endpoints are protected by the existing auth middleware (NOT
listed in `_PUBLIC_PATHS`).
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["cron"])

# ── Scheduler registry ─────────────────────────────────────────────
#
# `backend/main.py` lifespan populates this dict so /api/jobs/all can
# call .get_jobs() on the live AsyncIOScheduler instances without
# importing main.py (which would create a circular dependency at
# import time).

_RUNNING_SCHEDULERS: dict[str, Any] = {}


def register_scheduler(name: str, scheduler: Any) -> None:
    """Register a running APScheduler so /api/jobs/all can introspect it.

    Called from backend/main.py lifespan. Safe to call multiple times --
    last writer wins per name.
    """
    _RUNNING_SCHEDULERS[name] = scheduler


def get_registered_schedulers() -> dict[str, Any]:
    return dict(_RUNNING_SCHEDULERS)


# ── Static manifest for out-of-process jobs ────────────────────────
#
# slack_bot runs in a separate process (`python -m backend.slack_bot.app`).
# The MAIN backend cannot call .get_jobs() on its scheduler. Mirror the
# canonical job list from `backend/slack_bot/scheduler.py:31-90, 340-344`
# so the operator at least sees what SHOULD be running. last_run + status
# remain "unknown" -- a future phase can wire a heartbeat POST.

_SLACK_BOT_JOBS: tuple[dict[str, str], ...] = (
    {"id": "morning_digest",        "schedule": "cron daily morning_digest_hour:00 ET",
     "description": "Slack morning digest (top movers + holdings recap)"},
    {"id": "evening_digest",        "schedule": "cron daily evening_digest_hour:00 ET",
     "description": "Slack evening digest (P&L + closed trades)"},
    {"id": "watchdog_health_check", "schedule": "interval watchdog_interval_minutes",
     "description": "Slack-bot self-watchdog (alerts on backend unreachability)"},
    {"id": "prompt_leak_redteam",   "schedule": "cron daily 03:15 ET",
     "description": "Nightly red-team prompt-leak audit"},
    {"id": "daily_price_refresh",      "schedule": "phase-9.2 cron",
     "description": "Daily refresh of universe price snapshots"},
    {"id": "weekly_fred_refresh",      "schedule": "phase-9.3 cron",
     "description": "Weekly refresh of FRED macro series"},
    {"id": "nightly_mda_retrain",      "schedule": "phase-9.4 cron",
     "description": "Nightly MDA feature-importance retrain"},
    {"id": "hourly_signal_warmup",     "schedule": "phase-9.5 interval",
     "description": "Hourly cache warmup for enrichment signals"},
    {"id": "nightly_outcome_rebuild",  "schedule": "phase-9.6 cron",
     "description": "Nightly outcome-tracking refresh"},
    {"id": "weekly_data_integrity",    "schedule": "phase-9.7 cron",
     "description": "Weekly BQ data-integrity audit"},
    {"id": "cost_budget_watcher",      "schedule": "phase-9.8 interval",
     "description": "Cost-budget watcher + soft-cap alerts"},
)

# phase-23.3.4: extended from 1 to 6 entries after launchd audit. Plists
# for the 5 new services live in `~/Library/LaunchAgents/` (user-local,
# not in repo per local-only deployment doctrine). `claude-code-proxy`
# is intentionally omitted -- it is Claude Code's own service.
_LAUNCHD_JOBS: tuple[dict[str, str], ...] = (
    {"id": "com.pyfinagent.backend-watchdog", "schedule": "launchd interval 60s",
     "description": "External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)"},
    {"id": "com.pyfinagent.backend",          "schedule": "launchd KeepAlive RunAtLoad",
     "description": "FastAPI backend daemon (uvicorn :8000); auto-respawns on EXIT"},
    {"id": "com.pyfinagent.frontend",         "schedule": "launchd KeepAlive RunAtLoad",
     "description": "Next.js frontend dev server (:3000)"},
    {"id": "com.pyfinagent.mas-harness",      "schedule": "launchd interval 1800s",
     "description": "MAS harness optimizer cycle (every 30 min)"},
    {"id": "com.pyfinagent.ablation",         "schedule": "launchd cron 03:00 daily",
     "description": "Nightly feature ablation experiment"},
    {"id": "com.pyfinagent.autoresearch",     "schedule": "launchd cron 02:00 daily",
     "description": "Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"},
)


# ── Log allowlist ──────────────────────────────────────────────────
#
# Path traversal mitigation: the client passes a KEY, the server
# resolves it to a fixed Path. Unknown keys -> 400. The server NEVER
# echoes a raw path back to the client.

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _log_paths() -> dict[str, Path]:
    return {
        "backend":             _REPO_ROOT / "backend.log",
        "watchdog":            _REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log",
        "restart":             _REPO_ROOT / "handoff" / "logs" / "backend-restart.log",
        "harness":             _REPO_ROOT / "handoff" / "logs" / "mas-harness.log",
        "autoresearch":        _REPO_ROOT / "handoff" / "logs" / "autoresearch.log",
        "mas_harness_launchd": _REPO_ROOT / "handoff" / "logs" / "mas-harness.launchd.log",
    }


_LINES_MIN = 10
_LINES_MAX = 1000


# ── Helpers ────────────────────────────────────────────────────────


def _trigger_str(trigger: Any) -> str:
    """Render an APScheduler trigger to a short human-readable string."""
    try:
        return str(trigger)
    except Exception:
        return "unknown"


def _job_to_dict(job: Any, source: str) -> dict[str, Any]:
    next_run: Optional[str] = None
    nrt = getattr(job, "next_run_time", None)
    if nrt is not None:
        try:
            next_run = nrt.isoformat()
        except Exception:
            next_run = str(nrt)
    return {
        "id": getattr(job, "id", "?"),
        "source": source,
        "schedule": _trigger_str(getattr(job, "trigger", None)),
        "next_run": next_run,
        "last_run": None,  # APScheduler doesn't expose this; phase-2 if needed
        "status": "scheduled" if nrt is not None else "paused",
        "description": getattr(job, "name", None) or getattr(job, "id", "?"),
    }


def _static_to_dict(entry: dict[str, str], source: str) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "source": source,
        "schedule": entry.get("schedule", "?"),
        "next_run": None,
        "last_run": None,
        "status": "manifest",
        "description": entry.get("description", entry["id"]),
    }


# ── Endpoints ──────────────────────────────────────────────────────


@router.get("/jobs/all")
async def get_all_jobs() -> dict[str, Any]:
    """Unified job inventory: live APScheduler + static manifests."""
    jobs: list[dict[str, Any]] = []

    for name, scheduler in get_registered_schedulers().items():
        try:
            for job in scheduler.get_jobs():
                jobs.append(_job_to_dict(job, source="main_apscheduler"))
        except Exception as exc:
            logger.warning(
                "get_all_jobs: failed to introspect scheduler %s: %r", name, exc
            )

    for entry in _SLACK_BOT_JOBS:
        jobs.append(_static_to_dict(entry, source="slack_bot"))

    for entry in _LAUNCHD_JOBS:
        jobs.append(_static_to_dict(entry, source="launchd"))

    return {
        "jobs": jobs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(jobs),
    }


@router.get("/logs/tail")
async def get_log_tail(
    log: str = Query(..., description="Allowlisted log key"),
    lines: int = Query(200, ge=1, le=10000, description="Number of trailing lines"),
) -> dict[str, Any]:
    """Tail-read an allowlisted log file. Rejects unknown keys with 400."""
    paths = _log_paths()
    if log not in paths:
        raise HTTPException(
            status_code=400,
            detail=f"unknown log key: {log!r}; allowed: {sorted(paths.keys())}",
        )

    n = max(_LINES_MIN, min(_LINES_MAX, int(lines)))
    p = paths[log]

    if not p.exists():
        return {
            "log": log,
            "lines": [],
            "n_returned": 0,
            "total_size_bytes": 0,
            "exists": False,
        }

    total_size = p.stat().st_size
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            tail = list(deque(f, maxlen=n))
    except Exception as exc:
        logger.warning("get_log_tail(%s) failed: %r", log, exc)
        raise HTTPException(status_code=500, detail="log read failed")

    # Strip trailing newlines so the frontend can re-join with \n.
    cleaned = [ln.rstrip("\n") for ln in tail]

    return {
        "log": log,
        "lines": cleaned,
        "n_returned": len(cleaned),
        "total_size_bytes": total_size,
        "exists": True,
    }
