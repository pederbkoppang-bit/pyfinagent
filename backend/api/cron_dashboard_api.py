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
import os
import plistlib
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# phase-23.5.2.5: merge job_status_api's heartbeat registry into
# /api/jobs/all so slack_bot rows reflect real status + last_run + next_run
# instead of static "manifest" placeholders. Import is safe (job_status_api
# does not import from this module).
from backend.api import job_status_api
# phase-49.2: operator cron-control (pause/resume/trigger) for the 2
# backend-owned in-process jobs. cron_control lazy-imports this module's
# get_registered_schedulers(), so no import cycle here.
from backend.services import cron_control
from backend.services.api_cache import get_api_cache

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

# phase-23.6.2: schedule strings use APScheduler bracket notation matching
# `_trigger_str()` output for `main_apscheduler` rows -- so live + static
# rows on the /cron dashboard render in a single consistent shape. The 3
# settings-driven values encode their defaults (morning_digest_hour=8,
# evening_digest_hour=17, watchdog_interval_minutes=15); inline comments
# flag them so an operator who changes the setting also updates the label.
_SLACK_BOT_JOBS: tuple[dict[str, str], ...] = (
    {"id": "morning_digest",        "schedule": "cron[hour='8', minute='0']",  # configurable via morning_digest_hour
     "description": "Slack morning digest (top movers + holdings recap)"},
    {"id": "evening_digest",        "schedule": "cron[hour='17', minute='0']",  # configurable via evening_digest_hour
     "description": "Slack evening digest (P&L + closed trades)"},
    {"id": "watchdog_health_check", "schedule": "interval[0:15:00]",  # configurable via watchdog_interval_minutes
     "description": "Slack-bot self-watchdog (alerts on backend unreachability)"},
    {"id": "prompt_leak_redteam",   "schedule": "cron[hour='3', minute='15']",
     "description": "Nightly red-team prompt-leak audit"},
    {"id": "daily_price_refresh",      "schedule": "cron[hour='1']",
     "description": "Daily refresh of universe price snapshots"},
    {"id": "weekly_fred_refresh",      "schedule": "cron[day_of_week='sun', hour='2']",
     "description": "Weekly refresh of FRED macro series"},
    {"id": "nightly_mda_retrain",      "schedule": "cron[hour='3']",
     "description": "Nightly MDA feature-importance retrain"},
    {"id": "hourly_signal_warmup",     "schedule": "cron[minute='5']",
     "description": "Hourly cache warmup for enrichment signals"},
    {"id": "nightly_outcome_rebuild",  "schedule": "cron[hour='4']",
     "description": "Nightly outcome-tracking refresh"},
    {"id": "weekly_data_integrity",    "schedule": "cron[day_of_week='mon', hour='5']",
     "description": "Weekly BQ data-integrity audit"},
    {"id": "cost_budget_watcher",      "schedule": "cron[hour='6']",
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
     "description": "Nightly autoresearch memo (exit 1 -- partial .env fix applied; python entrypoint still failing -- see phase-23.5.19)"},
)


# ── Log allowlist ──────────────────────────────────────────────────
#
# Path traversal mitigation: the client passes a KEY, the server
# resolves it to a fixed Path. Unknown keys -> 400. The server NEVER
# echoes a raw path back to the client.

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _log_paths() -> dict[str, Path]:
    # phase-23.3.5: live launchd-managed logs write to `handoff/<x>.log` at
    # repo root, NOT `handoff/logs/<x>.log`. Pre-fix, /cron showed 18-day
    # stale duplicates because the allowlist pointed at the wrong dir for
    # mas-harness/autoresearch/.launchd.log. Backend, watchdog, restart
    # logs are correctly at handoff/logs/ (different writers).
    return {
        # FastAPI backend stdout (uvicorn), repo-root for legacy reasons.
        "backend":               _REPO_ROOT / "backend.log",
        # Backend watchdog shell script writes to handoff/logs/ correctly.
        "watchdog":              _REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log",
        # Backend-restart log is quiescent (only written on restart events).
        "restart":               _REPO_ROOT / "handoff" / "logs" / "backend-restart.log",
        # phase-23.3.5: MAS harness writes to repo-root via its launchd plist
        # StandardOutPath (NOT handoff/logs/). Live file is 38+ MB, growing.
        "harness":               _REPO_ROOT / "handoff" / "mas-harness.log",
        # phase-23.3.5: same correction for autoresearch.
        "autoresearch":          _REPO_ROOT / "handoff" / "autoresearch.log",
        # phase-23.3.5: launchd's stderr capture for mas-harness, repo-root.
        "mas_harness_launchd":   _REPO_ROOT / "handoff" / "mas-harness.launchd.log",
        # phase-23.3.5: NEW keys -- autoresearch + ablation launchd stderr.
        # These surface the .env exit-127 errors (line 24 ALPHAVANTAGE_API_KEY
        # for autoresearch, line 56 ANTHROPIC_API_KEY for ablation).
        "autoresearch_launchd":  _REPO_ROOT / "handoff" / "autoresearch.launchd.log",
        "ablation":              _REPO_ROOT / "handoff" / "ablation.log",
        "ablation_launchd":      _REPO_ROOT / "handoff" / "ablation.launchd.log",
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
        # phase-49.2: operator can pause/resume/trigger this job in-process.
        "controllable": cron_control.is_controllable(getattr(job, "id", "")),
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


# ── phase-23.5.13.2: launchctl-print bridge ────────────────────────
#
# Probe `launchctl print gui/<uid>/<label>` per launchd entry to surface
# real status (running / ok / failed / not_loaded / unknown). launchd does
# not expose next-fire-time or last-run timestamp via launchctl, so those
# columns stay null for launchd entries -- a known gap; surfacing real
# `status` is still a strict improvement over the prior hardcoded
# `status="manifest"` placeholder.
#
# Cached for 30s per-label to keep `/api/jobs/all` responsive (six
# subprocess forks per uncached request would add ~400ms; 30s TTL gives
# the operator at most 2 cache-miss cycles per minute).

_LAUNCHCTL_TTL_SECONDS = 30.0
_LAUNCHCTL_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_LAUNCHCTL_TIMEOUT_S = 5.0
_LAUNCHCTL_STATE_RE = re.compile(r"^\s*state\s*=\s*(.+?)\s*$", re.MULTILINE)
_LAUNCHCTL_EXIT_RE = re.compile(r"^\s*last exit code\s*=\s*(-?\d+)\s*$", re.MULTILINE)
_LAUNCHCTL_PID_RE = re.compile(r"^\s*pid\s*=\s*(\d+)\s*$", re.MULTILINE)
_LAUNCHCTL_RUNS_RE = re.compile(r"^\s*runs\s*=\s*(\d+)\s*$", re.MULTILINE)


def _classify_launchctl_state(state: str | None, exit_code: int | None) -> str:
    """Map (state, last_exit_code) to a dashboard status string.

    state="running"                                  -> "running"
    state="not running", no exit code                -> "ok"   (never fired or pre-exit)
    state="not running", exit_code in {0, -15}       -> "ok"   (clean / SIGTERM cycle)
    state="not running", exit_code != 0 and != -15   -> "failed"
    state=None (anything unparseable)                -> "unknown"
    """
    if state is None:
        return "unknown"
    if state == "running":
        return "running"
    if state == "not running":
        if exit_code is None or exit_code == 0 or exit_code == -15:
            return "ok"
        return "failed"
    return "unknown"


def _probe_launchctl(label: str) -> dict[str, Any]:
    """Run `launchctl print gui/<uid>/<label>` and parse the output.

    Returns a dict with keys: status, last_exit_code, pid, runs, next_run,
    last_run. Never raises -- subprocess failures map to status="not_loaded"
    (returncode != 0) or status="unknown" (timeout / OS error).
    """
    target = f"gui/{os.getuid()}/{label}"
    try:
        proc = subprocess.run(
            ["launchctl", "print", target],
            capture_output=True,
            text=True,
            timeout=_LAUNCHCTL_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as exc:
        logger.warning("launchctl print %s timed out / failed: %r", target, exc)
        return {
            "status": "unknown",
            "last_exit_code": None,
            "pid": None,
            "runs": None,
            "next_run": None,
            "last_run": None,
        }

    if proc.returncode != 0:
        return {
            "status": "not_loaded",
            "last_exit_code": None,
            "pid": None,
            "runs": None,
            "next_run": None,
            "last_run": None,
        }

    out = proc.stdout or ""
    state_m = _LAUNCHCTL_STATE_RE.search(out)
    exit_m = _LAUNCHCTL_EXIT_RE.search(out)
    pid_m = _LAUNCHCTL_PID_RE.search(out)
    runs_m = _LAUNCHCTL_RUNS_RE.search(out)
    state = state_m.group(1).strip() if state_m else None
    last_exit_code = int(exit_m.group(1)) if exit_m else None
    pid = int(pid_m.group(1)) if pid_m else None
    runs = int(runs_m.group(1)) if runs_m else None
    return {
        "status": _classify_launchctl_state(state, last_exit_code),
        "last_exit_code": last_exit_code,
        "pid": pid,
        "runs": runs,
        "next_run": None,  # launchctl doesn't expose this
        "last_run": None,  # launchctl doesn't expose this
    }


def _launchctl_state(label: str) -> dict[str, Any]:
    """Cached wrapper around `_probe_launchctl` (30s TTL)."""
    now = time.monotonic()
    cached = _LAUNCHCTL_CACHE.get(label)
    if cached is not None and (now - cached[1]) < _LAUNCHCTL_TTL_SECONDS:
        return cached[0]
    fresh = _probe_launchctl(label)
    _LAUNCHCTL_CACHE[label] = (fresh, now)
    return fresh


# ── phase-23.6.3: plist-derived next-fire-time ─────────────────────
#
# launchctl print does NOT expose next-fire-time, so the launchd merge
# block in `get_all_jobs()` previously left `next_run: null` for all 6
# launchd jobs. For StartCalendarInterval jobs the next-fire is fully
# determined by the on-disk plist + the system clock, so we can compute
# it locally. Two jobs are affected (com.pyfinagent.ablation Hour=3,
# com.pyfinagent.autoresearch Hour=2). Unsupported shapes (Weekday-only,
# array-of-dicts, StartInterval-only) return None.

_PLIST_TTL_SECONDS = 60.0
_PLIST_CACHE: dict[str, tuple[Optional[dict[str, Any]], float]] = {}
_PLIST_DIR = Path.home() / "Library" / "LaunchAgents"


def _load_plist(label: str) -> Optional[dict[str, Any]]:
    """Parse `~/Library/LaunchAgents/<label>.plist` with 60s in-process cache.

    Returns the parsed plist dict, or None on missing-file / parse-error.
    Never raises.
    """
    now = time.monotonic()
    cached = _PLIST_CACHE.get(label)
    if cached is not None and (now - cached[1]) < _PLIST_TTL_SECONDS:
        return cached[0]
    path = _PLIST_DIR / f"{label}.plist"
    parsed: Optional[dict[str, Any]] = None
    try:
        with path.open("rb") as fp:
            data = plistlib.load(fp)
        if isinstance(data, dict):
            parsed = data
    except (FileNotFoundError, OSError, plistlib.InvalidFileException, ValueError) as exc:
        logger.debug("_load_plist(%s): %r", label, exc)
        parsed = None
    except Exception as exc:
        logger.warning("_load_plist(%s) unexpected error: %r", label, exc)
        parsed = None
    _PLIST_CACHE[label] = (parsed, now)
    return parsed


def _plist_next_run(label: str) -> Optional[str]:
    """Compute the next StartCalendarInterval fire time as an aware ISO string.

    Only handles the `{Hour, Minute}` dict shape -- the shape used by both
    in-scope jobs (ablation, autoresearch). Returns None for:
    - missing plist or parse error
    - StartCalendarInterval as array-of-dicts
    - StartCalendarInterval with Weekday / Day / Month keys
    - StartInterval-only jobs (no StartCalendarInterval at all)
    - malformed Hour / Minute (out of range, non-integer)

    Emits local-tz-aware ISO 8601 string with offset, matching the format
    used by APScheduler `next_run_time.isoformat()` elsewhere on the dashboard.
    """
    try:
        plist = _load_plist(label)
        if not plist:
            return None
        sci = plist.get("StartCalendarInterval")
        if not isinstance(sci, dict):
            return None
        # Only the {Hour, Minute} shape is in scope -- reject anything that
        # also specifies Weekday / Day / Month so the verifier is forced to
        # tighten the helper when a new shape comes online.
        if any(k in sci for k in ("Weekday", "Day", "Month")):
            return None
        hour = sci.get("Hour")
        minute = sci.get("Minute", 0)
        if not isinstance(hour, int) or not isinstance(minute, int):
            return None
        if not (0 <= hour <= 23) or not (0 <= minute <= 59):
            return None
        now = datetime.now().astimezone()
        today_fire = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        next_fire = today_fire if now < today_fire else today_fire + timedelta(days=1)
        return next_fire.isoformat()
    except Exception as exc:  # defensive: never raise from the merge loop
        logger.warning("_plist_next_run(%s) failed: %r", label, exc)
        return None


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

    # phase-23.5.2.5: merge slack_bot manifest with the heartbeat registry
    # so each row carries real status / last_run / next_run pushed by the
    # slack-bot scheduler's listener + startup state-push. Fallback when
    # an entry has no registry row: status="never_run" (matches
    # job_status_api JobStatus default; see researcher brief on Prefect /
    # Airflow / Dagster vocabulary).
    snapshot = job_status_api.get_registry_snapshot()
    for entry in _SLACK_BOT_JOBS:
        row = snapshot.get(entry["id"], {})
        jobs.append(
            {
                "id": entry["id"],
                "source": "slack_bot",
                "schedule": entry.get("schedule", "?"),
                "next_run": row.get("next_run_time"),
                "last_run": row.get("last_run_at"),
                "status": row.get("status", "never_run"),
                "description": entry.get("description", entry["id"]),
            }
        )

    # phase-23.5.13.2: merge launchd manifest with live `launchctl print`
    # state per entry. Surfaces real status (running / ok / failed /
    # not_loaded / unknown) instead of the prior hardcoded "manifest"
    # placeholder. last_run stays null (launchctl does not expose it).
    # phase-23.6.3: for StartCalendarInterval jobs the next-fire-time is
    # fully derivable from the plist + system clock -- compute it locally
    # (ablation @ 03:00, autoresearch @ 02:00). Falls back to None for
    # StartInterval-only / array-of-dicts / Weekday-bearing / missing
    # plist cases.
    for entry in _LAUNCHD_JOBS:
        probe = _launchctl_state(entry["id"])
        next_run = probe.get("next_run") or _plist_next_run(entry["id"])
        jobs.append(
            {
                "id": entry["id"],
                "source": "launchd",
                "schedule": entry.get("schedule", "?"),
                "next_run": next_run,
                "last_run": probe.get("last_run"),
                "status": probe.get("status", "unknown"),
                "description": entry.get("description", entry["id"]),
            }
        )

    return {
        "jobs": jobs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(jobs),
    }


# ── phase-49.2: operator cron control (pause / resume / trigger) ──
# In-process control of the 2 backend-owned APScheduler jobs only
# (paper_trading_daily, ticket_queue_process_batch); cross-process
# slack_bot/launchd jobs -> 404. Confirmation-gated + audited. trigger
# reuses paper_trading /run-now's triple-guard so it can never double-fire.
class CronControlRequest(BaseModel):
    """Confirmation token (must equal the action verb) + optional reason."""
    confirmation: str
    reason: str = Field("manual", max_length=200)


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str, req: CronControlRequest):
    if req.confirmation != "PAUSE_JOB":
        raise HTTPException(400, "Confirmation must equal PAUSE_JOB")
    try:
        state = cron_control.pause(job_id, reason=req.reason)
    except cron_control.CronControlError as e:
        raise HTTPException(404, str(e))
    get_api_cache().invalidate("paper:*")
    return {"status": "paused", "job": state}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, req: CronControlRequest):
    if req.confirmation != "RESUME_JOB":
        raise HTTPException(400, "Confirmation must equal RESUME_JOB")
    try:
        state = cron_control.resume(job_id, reason=req.reason)
    except cron_control.CronControlError as e:
        raise HTTPException(404, str(e))
    get_api_cache().invalidate("paper:*")
    return {"status": "resumed", "job": state}


@router.post("/jobs/{job_id}/trigger")
async def trigger_job(job_id: str, req: CronControlRequest):
    if req.confirmation != "TRIGGER_JOB":
        raise HTTPException(400, "Confirmation must equal TRIGGER_JOB")
    if not cron_control.is_controllable(job_id):
        raise HTTPException(404, f"'{job_id}' is not controllable in-process (cross-process or unknown job)")
    if job_id == "paper_trading_daily":
        # Reuse /run-now's TRIPLE guard (running-check 409 + _running flag +
        # cycle_lock flock) -- never double-fire a cycle. Lazy import avoids
        # any import cycle with paper_trading.
        from backend.api.paper_trading import run_now
        result = await run_now()  # raises HTTPException(409) if a cycle is running
        cron_control.record_trigger(job_id, reason=req.reason, result="delegated_to_run_now")
        return {"status": "triggered", "job_id": job_id, "detail": result}
    # ticket_queue_process_batch fires every minute; a guarded manual trigger
    # is a low-value follow-on. pause/resume are supported for it.
    raise HTTPException(400, f"trigger not supported for '{job_id}' in phase-49.2 (pause/resume only)")


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
