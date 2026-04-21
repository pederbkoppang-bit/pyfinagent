"""
Scheduled jobs: morning digest, evening digest, and watchdog health check.
Uses APScheduler to run tasks within the Slack bot process.
"""

import logging
from datetime import datetime

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest, format_evening_digest, format_escalation_alert

logger = logging.getLogger(__name__)

_BACKEND_URL = "http://backend:8000"
_scheduler: AsyncIOScheduler | None = None


def start_scheduler(app: AsyncApp):
    """Start the APScheduler with daily digests and watchdog jobs."""
    global _scheduler
    settings = get_settings()

    if not settings.slack_channel_id:
        logger.warning("SLACK_CHANNEL_ID not set -- scheduled jobs disabled")
        return

    _scheduler = AsyncIOScheduler()

    # Morning digest — daily at configured hour
    _scheduler.add_job(
        _send_morning_digest,
        "cron",
        hour=settings.morning_digest_hour,
        minute=0,
        args=[app],
        id="morning_digest",
        replace_existing=True,
    )

    # Evening digest — daily at configured hour
    _scheduler.add_job(
        _send_evening_digest,
        "cron",
        hour=settings.evening_digest_hour,
        minute=0,
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
        args=[app],
        id="prompt_leak_redteam",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started: morning digest at %d:00, evening digest at %d:00, "
        "watchdog every %d min",
        settings.morning_digest_hour,
        settings.evening_digest_hour,
        settings.watchdog_interval_minutes,
    )


async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            reports_res = await client.get(f"{_BACKEND_URL}/api/reports/?limit=5")
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
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            trades_res = await client.get(f"{_BACKEND_URL}/api/paper-trading/trades?limit=10")
            trades_data = trades_res.json() if trades_res.status_code == 200 else []

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
    """Probe backend health endpoint; post to Slack only on failure."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_BACKEND_URL}/api/health")
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.debug("Watchdog health check passed")
                return

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        ":warning: *Watchdog Alert* -- Backend health check failed\n"
                        f"Status: {resp.status_code} at {datetime.now().strftime('%H:%M:%S')}"
                    ),
                },
            }],
            text="Watchdog Alert: backend health check failed",
        )
        logger.warning("Watchdog health check failed -- status %d", resp.status_code)

    except Exception:
        try:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            ":rotating_light: *Watchdog Alert* -- Backend unreachable\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        ),
                    },
                }],
                text="Watchdog Alert: backend unreachable",
            )
        except Exception:
            pass
        logger.exception("Watchdog health check -- backend unreachable")


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


def register_phase9_jobs(scheduler, replace_existing: bool = True) -> list[str]:
    """Register all phase-9 jobs on the passed scheduler. Returns the job IDs registered.

    `scheduler` is any object with `.add_job(func, trigger=..., id=..., replace_existing=...)`.
    Pass `replace_existing=True` so reloads do not raise `ConflictingIdError`.

    Fail-open per job: a missing job module does not block the others.
    """
    registered: list[str] = []
    mapping = {
        "daily_price_refresh": ("backend.slack_bot.jobs.daily_price_refresh", "cron", {"hour": 1}),
        "weekly_fred_refresh": ("backend.slack_bot.jobs.weekly_fred_refresh", "cron", {"day_of_week": "sun", "hour": 2}),
        "nightly_mda_retrain": ("backend.slack_bot.jobs.nightly_mda_retrain", "cron", {"hour": 3}),
        "hourly_signal_warmup": ("backend.slack_bot.jobs.hourly_signal_warmup", "cron", {"minute": 5}),
        "nightly_outcome_rebuild": ("backend.slack_bot.jobs.nightly_outcome_rebuild", "cron", {"hour": 4}),
        "weekly_data_integrity": ("backend.slack_bot.jobs.weekly_data_integrity", "cron", {"day_of_week": "mon", "hour": 5}),
        "cost_budget_watcher": ("backend.slack_bot.jobs.cost_budget_watcher", "cron", {"hour": 6}),
    }
    for job_id, (module_path, trigger, kwargs) in mapping.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            func = getattr(mod, "run")
        except Exception as exc:
            logger.warning("register_phase9_jobs: %s import fail-open: %r", job_id, exc)
            continue
        try:
            scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=replace_existing, **kwargs)
            registered.append(job_id)
        except Exception as exc:
            logger.warning("register_phase9_jobs: %s add_job fail-open: %r", job_id, exc)
    return registered
