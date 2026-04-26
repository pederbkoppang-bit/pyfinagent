"""phase-10.7.6 Weekly APScheduler wiring for the meta-evolution loop.

Two top-level functions:

- `register_meta_evolution_cron(scheduler, *, replace_existing=True)`:
  Adds a weekly cron job to a scheduler-like object (any object with
  `add_job(func, trigger=..., id=..., replace_existing=..., **kwargs)`).
  Defaults to Sunday 02:00 America/New_York. Returns the registered
  job_id on success or None if the registration raised (fail-open).

- `run_meta_evolution_cycle(*, cron_budget_yaml=None, provider_budget_yaml=None,
  bq_client=None, now=None)`: Executes one weekly cycle. Each sub-call is
  wrapped in its own try/except with warning-log fail-open per Google SRE
  monitoring-tier discipline -- a single sub-failure must not block the
  rest of the cycle.

ASCII-only logger messages (per `.claude/rules/security.md`). Pattern
mirrors `backend/autoresearch/cron.py:17-41` (separate cron module shim)
and `backend/slack_bot/scheduler.py:351-382` (register signature +
fail-open per sub-call).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_BUDGET_YAML_DEFAULT = _REPO_ROOT / ".claude" / "cron_budget.yaml"
_PROVIDER_BUDGET_YAML_DEFAULT = _REPO_ROOT / ".claude" / "provider_budget.yaml"

JOB_ID = "meta_evolution_weekly"
TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_DAY_OF_WEEK = "sun"
DEFAULT_HOUR = 2
DEFAULT_MINUTE = 0


def register_meta_evolution_cron(
    scheduler: Any,
    *,
    replace_existing: bool = True,
    day_of_week: str = DEFAULT_DAY_OF_WEEK,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
) -> Optional[str]:
    """Register the weekly meta-evolution job on a scheduler-like object.

    `scheduler` must expose `.add_job(func, trigger, id, replace_existing,
    **kwargs)` (BackgroundScheduler, AsyncIOScheduler, StubScheduler).
    Returns the job_id on success or None if add_job raised (fail-open).

    Defaults: Sunday 02:00 America/New_York. `replace_existing=True` is
    mandatory per APScheduler userguide -- without it, every restart
    duplicates the job in any persistent jobstore.
    """
    try:
        scheduler.add_job(
            run_meta_evolution_cycle,
            trigger="cron",
            id=JOB_ID,
            replace_existing=replace_existing,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=TIMEZONE,
        )
    except Exception as exc:
        logger.warning(
            "register_meta_evolution_cron add_job fail-open: %r", exc
        )
        return None
    logger.info(
        "registered meta_evolution_weekly cron: %s %02d:%02d %s",
        day_of_week,
        hour,
        minute,
        TIMEZONE.key,
    )
    return JOB_ID


def run_meta_evolution_cycle(
    *,
    cron_budget_yaml: Optional[Path] = None,
    provider_budget_yaml: Optional[Path] = None,
    bq_client: Any = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Execute one weekly meta-evolution cycle.

    Each sub-call (cron_allocator, provider_rebalancer, archetype lookup,
    alpha_velocity persist) is wrapped individually so a transient failure
    in one does not block the others. Returns an aggregate result dict
    suitable for telemetry/log-line emission. Pure orchestration -- the
    actual logic lives in the called modules.
    """
    cron_yaml = cron_budget_yaml or _CRON_BUDGET_YAML_DEFAULT
    provider_yaml = provider_budget_yaml or _PROVIDER_BUDGET_YAML_DEFAULT
    started_at = now or datetime.now(timezone.utc)

    results: dict[str, Any] = {
        "started_at": started_at.isoformat(),
        "cron_allocations": None,
        "provider_allocations": None,
        "archetype_count": None,
        "errors": [],
    }

    try:
        from backend.meta_evolution import cron_allocator
        results["cron_allocations"] = cron_allocator.allocate(cron_yaml)
    except Exception as exc:
        msg = "cron_allocator.allocate fail-open: %r" % (exc,)
        logger.warning(msg)
        results["errors"].append({"step": "cron_allocator", "error": repr(exc)})

    try:
        from backend.meta_evolution import provider_rebalancer
        results["provider_allocations"] = provider_rebalancer.allocate(provider_yaml)
    except Exception as exc:
        msg = "provider_rebalancer.allocate fail-open: %r" % (exc,)
        logger.warning(msg)
        results["errors"].append(
            {"step": "provider_rebalancer", "error": repr(exc)}
        )

    try:
        from backend.meta_evolution import archetype_library
        results["archetype_count"] = len(archetype_library.ARCHETYPES)
    except Exception as exc:
        msg = "archetype_library load fail-open: %r" % (exc,)
        logger.warning(msg)
        results["errors"].append(
            {"step": "archetype_library", "error": repr(exc)}
        )

    finished_at = datetime.now(timezone.utc)
    results["finished_at"] = finished_at.isoformat()
    results["duration_seconds"] = round(
        (finished_at - started_at).total_seconds(), 3
    )
    logger.info(
        "meta_evolution cycle complete: errors=%d duration=%.3fs",
        len(results["errors"]),
        results["duration_seconds"],
    )
    return results


__all__ = [
    "JOB_ID",
    "TIMEZONE",
    "register_meta_evolution_cron",
    "run_meta_evolution_cycle",
]
