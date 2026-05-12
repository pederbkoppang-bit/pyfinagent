"""
Scheduled jobs: morning digest, evening digest, and watchdog health check.
Uses APScheduler to run tasks within the Slack bot process.
"""

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest, format_evening_digest, format_escalation_alert

logger = logging.getLogger(__name__)

# phase-23.5.3.1: _BACKEND_URL is no longer referenced by any handler.
# Kept here for documentation -- this is the URL that would resolve under
# Docker-compose networking. All in-process callers (watchdog probe via
# _HEALTH_PROBE_URL, digests via _LOCAL_BACKEND_URL, heartbeats via
# _HEARTBEAT_URL) now use 127.0.0.1 directly because pyfinagent runs as
# a Mac host process and the `backend` DNS alias does not resolve.
_BACKEND_URL = "http://backend:8000"
# phase-23.3.2: heartbeat-push target. Pyfinagent is local-only on a single
# Mac per memory/project_local_only_deployment.md, so the slack-bot process
# can reach the main backend at localhost. Kept separate from _BACKEND_URL
# above (which uses the docker-network hostname) so a docker-compose
# resurrection wouldn't accidentally point heartbeats at a stale host.
_HEARTBEAT_URL = "http://127.0.0.1:8000/api/jobs/heartbeat"
# phase-23.5.2.6: separate health-probe URL pinned to localhost. _BACKEND_URL
# uses the Docker DNS alias which doesn't resolve on host-process slack-bot
# deployments -- causing the watchdog to spam Slack every 15 minutes with
# `Backend unreachable` alerts even when the backend was healthy.
_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"
# phase-23.5.3.1: shared base URL for digest handlers. Same rationale as
# _HEALTH_PROBE_URL -- localhost-pinned because the Docker alias above
# doesn't resolve from a Mac host process. Used by _send_morning_digest
# and _send_evening_digest below.
_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"
_scheduler: AsyncIOScheduler | None = None
# phase-23.5.2.6: track watchdog state across fires so we only post on
# transitions (None->False, True->False, False->True). Steady-state fires
# log only. Reset to None on daemon restart -- intentional; first-fire
# state is the post-restart baseline.
_watchdog_last_was_healthy: bool | None = None


def _aps_to_heartbeat(event) -> None:
    """phase-23.3.2: APScheduler event listener that POSTs each terminal
    job event to the main backend's /api/jobs/heartbeat. Fail-open: any
    exception is swallowed so the slack-bot scheduler never breaks.

    Wired in start_scheduler() below for EVENT_JOB_EXECUTED |
    EVENT_JOB_ERROR | EVENT_JOB_MISSED. Closes the gap that
    /api/jobs/status returned all 'never_run' for a month.
    """
    try:
        exc = getattr(event, "exception", None)
        status = "ok" if not exc else "failed"
        job_id = getattr(event, "job_id", "unknown")
        # phase-23.5.2.5: surface the NEXT fire's run-time so /api/jobs/all
        # can populate `next_run` without waiting on a separate state push
        # for jobs that fire often. Fail-open if get_job returns None
        # (possible mid-EVENT_JOB_MISSED transition).
        next_run_iso: str | None = None
        try:
            if _scheduler is not None:
                job_obj = _scheduler.get_job(job_id)
                if job_obj is not None and job_obj.next_run_time is not None:
                    next_run_iso = job_obj.next_run_time.isoformat()
        except Exception as inner:
            logger.warning("aps_to_heartbeat next_run lookup fail-open for %s: %r", job_id, inner)
        payload = {
            "job": job_id,
            "status": status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc) if exc else None,
            "next_run_time": next_run_iso,
        }
        # Sync httpx (the listener runs in APScheduler's executor, not
        # asyncio). Tight 3s timeout so a stuck main backend cannot
        # block the listener.
        with httpx.Client(timeout=3.0) as client:
            client.post(_HEARTBEAT_URL, json=payload)
    except Exception as e:
        logger.warning("aps_to_heartbeat fail-open: %r", e)


def _seed_next_run_registry() -> None:
    """phase-23.5.2.5: after the scheduler starts, push a one-time
    `status="scheduled"` heartbeat for every registered job so the
    backend's job-status registry has next_run_time populated BEFORE
    any job fires. Without this seed, /api/jobs/all surfaces
    `next_run=null` for jobs that haven't fired since the slack-bot
    daemon started.

    Fail-open per job: a single failed POST does not abort the loop.
    """
    if _scheduler is None:
        return
    try:
        jobs = list(_scheduler.get_jobs())
    except Exception as exc:
        logger.warning("_seed_next_run_registry get_jobs fail-open: %r", exc)
        return
    for j in jobs:
        try:
            payload = {
                "job": j.id,
                "status": "scheduled",
                "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            with httpx.Client(timeout=3.0) as client:
                client.post(_HEARTBEAT_URL, json=payload)
        except Exception as exc:
            logger.warning("_seed_next_run_registry fail-open for %s: %r", j.id, exc)


def start_scheduler(app: AsyncApp):
    """Start the APScheduler with daily digests and watchdog jobs."""
    global _scheduler
    settings = get_settings()

    if not settings.slack_channel_id:
        logger.warning("SLACK_CHANNEL_ID not set -- scheduled jobs disabled")
        return

    _scheduler = AsyncIOScheduler()

    # Morning digest — daily at configured hour (US Eastern, DST-aware)
    _scheduler.add_job(
        _send_morning_digest,
        "cron",
        hour=settings.morning_digest_hour,
        minute=0,
        timezone=ZoneInfo("America/New_York"),
        args=[app],
        id="morning_digest",
        replace_existing=True,
    )

    # Evening digest — daily at configured hour (US Eastern, DST-aware)
    _scheduler.add_job(
        _send_evening_digest,
        "cron",
        hour=settings.evening_digest_hour,
        minute=0,
        timezone=ZoneInfo("America/New_York"),
        args=[app],
        id="evening_digest",
        replace_existing=True,
    )

    # Watchdog health check — interval-based, alerts on failure only
    _scheduler.add_job(
        _watchdog_health_check,
        "interval",
        minutes=settings.watchdog_interval_minutes,
        args=[app],
        id="watchdog_health_check",
        replace_existing=True,
    )

    # phase-4.14.25: nightly prompt-leak red-team audit.
    # Runs the fixed attack suite against apply_leak_defenses and
    # appends results to handoff/prompt_leak_redteam_audit.jsonl.
    _scheduler.add_job(
        _nightly_prompt_leak_redteam,
        "cron",
        hour=3, minute=15,
        timezone=ZoneInfo("America/New_York"),
        args=[app],
        id="prompt_leak_redteam",
        replace_existing=True,
    )

    # phase-23.3.2: wire heartbeat-push so /api/jobs/status reflects
    # real fires for the 4 core jobs + 7 phase-9 jobs. Listener fires
    # on every terminal scheduler event; fail-open if backend down.
    _scheduler.add_listener(
        _aps_to_heartbeat,
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started: morning digest at %d:00, evening digest at %d:00, "
        "watchdog every %d min",
        settings.morning_digest_hour,
        settings.evening_digest_hour,
        settings.watchdog_interval_minutes,
    )

    # phase-23.3.3: register the 7 phase-9 jobs (previously dormant -- the
    # function was defined but never called). Fail-open so a bad import in
    # any single phase-9 module cannot break the 4 core jobs above.
    #
    # phase-23.6.1: capture the running event loop AND pass `app` so that
    # `register_phase9_jobs` can inject production fetch/write/alert fns
    # (via `_production_fns` factories). The loop is needed by the
    # `make_alert_fn_for_*` factories to bridge sync->async Slack posts via
    # `asyncio.run_coroutine_threadsafe`. `start_scheduler` is called from
    # `async def main()` in `app.py:46`, so a running loop is guaranteed.
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError as exc:
        logger.warning("start_scheduler: no running loop (production-fn wiring will be skipped): %r", exc)
        running_loop = None
    try:
        registered = register_phase9_jobs(_scheduler, app=app, loop=running_loop)
        logger.info("phase-9 jobs registered: %s", registered)
    except Exception as exc:
        logger.warning("register_phase9_jobs fail-open at startup: %r", exc)

    # phase-23.5.2.5: seed registry with next_run_time for every job so
    # /api/jobs/all has data before any job fires. Run AFTER phase-9
    # registration so all 11 jobs are visible to get_jobs().
    _seed_next_run_registry()


async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            reports_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/reports/?limit=5")
            reports_data = reports_res.json() if reports_res.status_code == 200 else []

        blocks = format_morning_digest(portfolio_data, reports_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Morning Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Morning digest sent")

    except Exception:
        logger.exception("Failed to send morning digest")


async def _send_evening_digest(app: AsyncApp):
    """Fetch end-of-day portfolio summary and post evening digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            trades_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=10")
            # phase-23.5.7.1: /api/paper-trading/trades returns the dict envelope
            # {"trades": [...], "count": N} (paper_trading.py:226). Unwrap at the
            # HTTP boundary so format_evening_digest's `trades_today[:10]` slice
            # gets a list, not a dict (which raises KeyError: slice(...)).
            _raw = trades_res.json() if trades_res.status_code == 200 else []
            trades_data = _raw.get("trades", []) if isinstance(_raw, dict) else _raw

        blocks = format_evening_digest(portfolio_data, trades_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Evening Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Evening digest sent")

    except Exception:
        logger.exception("Failed to send evening digest")


async def _watchdog_health_check(app: AsyncApp):
    """Probe backend health endpoint; post to Slack only on state transitions.

    phase-23.5.2.6: state-transition gating. Posts to Slack only on:
        None -> False  (first probe failed; alert)
        True -> False  (down: alert)
        False -> True  (recovery: alert)
    Steady-state (None->True, True->True, False->False) logs only.
    """
    global _watchdog_last_was_healthy

    settings = get_settings()
    is_healthy = False
    detail = ""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_HEALTH_PROBE_URL)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                is_healthy = True
                detail = f"HTTP {resp.status_code}"
            else:
                detail = f"HTTP {resp.status_code}, body did not have status=ok"
    except Exception as exc:
        detail = f"unreachable: {type(exc).__name__}"

    prior = _watchdog_last_was_healthy
    _watchdog_last_was_healthy = is_healthy

    # Decide whether to post.
    post: tuple[str, str] | None = None  # (emoji+text, fallback_text)
    if is_healthy:
        if prior is False:
            post = (
                f":white_check_mark: *Watchdog Recovery* -- Backend reachable again\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}",
                "Watchdog Recovery: backend reachable",
            )
            logger.info("Watchdog recovery -- %s", detail)
        else:
            # None->True (clean baseline) or True->True (steady) -- silent.
            logger.debug("Watchdog steady-healthy -- %s", detail)
    else:
        if prior is None or prior is True:
            # None->False (post-restart already broken) or True->False (transition).
            post = (
                f":rotating_light: *Watchdog Alert* -- Backend unreachable\n"
                f"Detail: {detail} at {datetime.now().strftime('%H:%M:%S')}",
                "Watchdog Alert: backend unreachable",
            )
            logger.warning("Watchdog unhealthy transition -- %s", detail)
        else:
            # False->False (steady-down) -- log only; do NOT spam.
            logger.warning("Watchdog steady-unhealthy -- %s", detail)

    if post is not None:
        try:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": post[0]},
                }],
                text=post[1],
            )
        except Exception:
            logger.exception("Watchdog Slack post failed")


def pause_signals() -> bool:
    """Shut down the scheduler, stopping all signal-related jobs.

    Returns True if the scheduler was running and is now stopped,
    False if it was already stopped or never started.
    This is the rollback command for Go-Live checklist item 4.4.6.4.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down -- all signal jobs paused (rollback 4.4.6.4)")
        return True
    logger.info("Scheduler was not running -- no action taken")
    return False


async def send_trading_escalation(
    app: AsyncApp,
    severity: str,
    title: str,
    details: dict,
    actions: list[str] | None = None,
):
    """Send a trading incident escalation to Slack and iMessage (P0 only).

    Escalation ladder:
      L1 (all severities): Post Block Kit alert to configured Slack channel.
      L2 (P0 only): Send iMessage to Peder via `imsg` CLI.

    Args:
        app: Slack Bolt app for posting.
        severity: "P0", "P1", or "P2".
        title: Short incident title.
        details: Key-value pairs for the alert body.
        actions: Optional recommended next steps.
    """
    settings = get_settings()
    severity = str(severity or "P1").upper()

    blocks = format_escalation_alert(severity, title, details, actions)
    text_fallback = f"[{severity}] {title}"

    # L1: Slack channel alert
    if settings.slack_channel_id:
        try:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=blocks,
                text=text_fallback,
            )
            logger.warning("Trading escalation posted to Slack: %s -- %s", severity, title)
        except Exception:
            logger.exception("Failed to post trading escalation to Slack")

    # L2: iMessage for P0 incidents
    if severity == "P0":
        _ESCALATION_PHONE = "+4794810537"
        imsg_text = (
            f"PYFINAGENT {severity}: {title}\n"
            + "\n".join(f"{k}: {v}" for k, v in list(details.items())[:5])
            + "\nImmediate attention required."
        )
        try:
            import subprocess
            subprocess.run(
                ["imsg", "send", "--to", _ESCALATION_PHONE, "--text", imsg_text],
                capture_output=True, text=True, timeout=10,
            )
            logger.warning("iMessage escalation sent for %s: %s", severity, title)
        except Exception:
            logger.exception("Failed to send iMessage escalation")


async def send_analysis_alert(app: AsyncApp, ticker: str, report: dict):
    """Post a proactive alert after analysis completes (called from orchestrator)."""
    settings = get_settings()
    if not settings.slack_channel_id:
        return

    try:
        score = report.get("final_weighted_score", 0)
        rec = report.get("recommendation", {})
        action = rec.get("action", "N/A") if isinstance(rec, dict) else str(rec)

        emoji = ":chart_with_upwards_trend:" if score >= 7 else ":chart_with_downwards_trend:" if score < 4 else ":bar_chart:"
        color = "#22c55e" if score >= 7 else "#ef4444" if score < 4 else "#f59e0b"

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *Analysis Complete: {ticker}*"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Score:* {score:.1f}/10"},
                    {"type": "mrkdwn", "text": f"*Recommendation:* {action}"},
                ],
            },
        ]

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"Analysis complete: {ticker} -- {action} ({score:.1f}/10)",
            attachments=[{"color": color, "blocks": []}],
        )
    except Exception:
        logger.exception(f"Failed to send alert for {ticker}")


async def _nightly_prompt_leak_redteam(app: AsyncApp):
    """phase-4.14.25: run the prompt-leak red-team audit once per day."""
    import subprocess
    from pathlib import Path
    settings = get_settings()
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "audit" / "prompt_leak_redteam.py"
    try:
        proc = subprocess.run(
            ["python", str(script), "--min-pass", "0.80"],
            capture_output=True, text=True, timeout=120, cwd=str(repo),
        )
        logger.info(
            "prompt_leak_redteam exit=%d stdout=%s",
            proc.returncode, proc.stdout[:200]
        )
        if proc.returncode != 0 and settings.slack_channel_id:
            try:
                await app.client.chat_postMessage(
                    channel=settings.slack_channel_id,
                    text=(
                        f"prompt-leak redteam audit FAILED (exit {proc.returncode}): "
                        f"{proc.stdout[:500]}"
                    ),
                )
            except Exception as post_err:
                logger.warning("redteam Slack alert failed: %s", post_err)
    except Exception as e:
        logger.error("prompt_leak_redteam job failed: %s", e)


# ============================================================
# phase-9.9 job registration
# ============================================================
#
# Wires the 7 phase-9 job modules into the APScheduler registered by
# start_scheduler(). Keeps backwards compat: existing morning/evening/
# watchdog/redteam jobs unchanged.
#
# Each job is idempotency-keyed (daily/weekly/hourly) via
# backend/slack_bot/job_runtime.py so double registration on reload is
# harmless.

_PHASE9_JOB_IDS: tuple[str, ...] = (
    "daily_price_refresh",      # 9.2
    "weekly_fred_refresh",      # 9.3
    "nightly_mda_retrain",      # 9.4
    "hourly_signal_warmup",     # 9.5
    "nightly_outcome_rebuild",  # 9.6
    "weekly_data_integrity",    # 9.7
    "cost_budget_watcher",      # 9.8
)


def register_phase9_jobs(
    scheduler,
    replace_existing: bool = True,
    *,
    app: "AsyncApp | None" = None,
    loop: "asyncio.AbstractEventLoop | None" = None,
) -> list[str]:
    """Register all phase-9 jobs on the passed scheduler. Returns the job IDs registered.

    `scheduler` is any object with `.add_job(func, trigger=..., id=..., replace_existing=...)`.
    Pass `replace_existing=True` so reloads do not raise `ConflictingIdError`.

    Fail-open per job: a missing job module does not block the others.

    phase-23.3.3: each entry now passes `misfire_grace_time` + `coalesce=True`
    so a slack-bot restart that crosses a scheduled tick does NOT immediately
    fire the missed tick. Grace times: 3600s daily, 7200s weekly, 600s hourly.

    phase-23.6.1: when `app` and `loop` are both provided, each registered
    callable is wrapped in `functools.partial(run, **prod_fns)` to inject
    real fetch/write/alert functions from `_production_fns`. When either is
    None (e.g. unit tests calling `register_phase9_jobs(StubScheduler())`),
    the bare `run` is registered — preserving existing test-injection paths.
    """
    import functools
    registered: list[str] = []

    # phase-23.6.1: build per-job production-fn dicts (only when caller wires
    # a real Slack app + asyncio loop). Factories live in `_production_fns`.
    prod_fns_per_job: dict[str, dict] = {}
    if app is not None and loop is not None:
        try:
            from backend.slack_bot.jobs import _production_fns as pf
            channel = get_settings().slack_channel_id or ""
            prod_fns_per_job = {
                "daily_price_refresh": {
                    "fetch_fn": pf.make_price_fetch_fn(),
                    "write_fn": pf.make_price_write_fn(),
                },
                "weekly_fred_refresh": {
                    "fetch_fn": pf.make_fred_fetch_fn(),
                    "write_fn": pf.make_fred_write_fn(),
                },
                "nightly_outcome_rebuild": {
                    "ledger_fetch_fn": pf.make_ledger_fetch_fn(),
                    "outcome_write_fn": pf.make_outcome_write_fn(),
                },
                "cost_budget_watcher": {
                    "alert_fn": pf.make_alert_fn_for_budget(app, loop, channel),
                },
                "weekly_data_integrity": {
                    "alert_fn": pf.make_alert_fn_for_integrity(app, loop, channel),
                },
            }
        except Exception as exc:
            logger.warning("register_phase9_jobs: production-fn wiring fail-open: %r", exc)
            prod_fns_per_job = {}
    # phase-23.3.3: include APScheduler safety params per researcher's brief.
    # `misfire_grace_time` prevents stale-tick fires on restart;
    # `coalesce=True` collapses missed ticks into one fire (defensive for
    # any future jobstore migration; harmless with default in-memory store).
    mapping = {
        "daily_price_refresh":     ("backend.slack_bot.jobs.daily_price_refresh", "cron",
                                    {"hour": 1, "misfire_grace_time": 3600, "coalesce": True}),
        "weekly_fred_refresh":     ("backend.slack_bot.jobs.weekly_fred_refresh", "cron",
                                    {"day_of_week": "sun", "hour": 2, "misfire_grace_time": 7200, "coalesce": True}),
        "nightly_mda_retrain":     ("backend.slack_bot.jobs.nightly_mda_retrain", "cron",
                                    {"hour": 3, "misfire_grace_time": 3600, "coalesce": True}),
        "hourly_signal_warmup":    ("backend.slack_bot.jobs.hourly_signal_warmup", "cron",
                                    {"minute": 5, "misfire_grace_time": 600, "coalesce": True}),
        "nightly_outcome_rebuild": ("backend.slack_bot.jobs.nightly_outcome_rebuild", "cron",
                                    {"hour": 4, "misfire_grace_time": 3600, "coalesce": True}),
        "weekly_data_integrity":   ("backend.slack_bot.jobs.weekly_data_integrity", "cron",
                                    {"day_of_week": "mon", "hour": 5, "misfire_grace_time": 7200, "coalesce": True}),
        "cost_budget_watcher":     ("backend.slack_bot.jobs.cost_budget_watcher", "cron",
                                    {"hour": 6, "misfire_grace_time": 3600, "coalesce": True}),
    }
    for job_id, (module_path, trigger, kwargs) in mapping.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            run_fn = getattr(mod, "run")
        except Exception as exc:
            logger.warning("register_phase9_jobs: %s import fail-open: %r", job_id, exc)
            continue
        # phase-23.6.1: partial-apply production fns when wired; fall back to
        # bare run when caller didn't pass app+loop (preserves test injection).
        prod_fns = prod_fns_per_job.get(job_id, {})
        func = functools.partial(run_fn, **prod_fns) if prod_fns else run_fn
        try:
            scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=replace_existing, **kwargs)
            registered.append(job_id)
        except Exception as exc:
            logger.warning("register_phase9_jobs: %s add_job fail-open: %r", job_id, exc)
    return registered
