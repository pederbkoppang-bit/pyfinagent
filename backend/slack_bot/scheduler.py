"""
Scheduled jobs: morning digest, evening digest, and watchdog health check.
Uses APScheduler to run tasks within the Slack bot process.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
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


def _route_exception_to_p1(
    exc: BaseException,
    *,
    endpoint: str,
    source: str = "scheduler",
    extra: dict | None = None,
) -> None:
    """phase-25.O: route a high-severity caught exception to P1 Slack.

    Closes audit bucket 24.5 F-5(f). Designed to be called AFTER an existing
    `logger.exception(...)` site so the stacktrace is preserved in the log
    AND a deduplicated P1 alert reaches Slack.

    Dedup fingerprint follows Sentry/PagerDuty canonical pattern:
        `f"{type(exc).__name__}:{endpoint}"`
    so the same exception class at the same endpoint deduplicates within
    the AlertDeduper window, while a distinct combination opens a fresh
    incident.

    Fully fail-open: if alerting itself raises, we log at WARNING and swallow.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync
        fingerprint = f"{type(exc).__name__}:{endpoint}"
        details: dict = {
            "endpoint": endpoint,
            "exception_class": type(exc).__name__,
            "exception_repr": repr(exc)[:300],
        }
        if extra:
            for k, v in extra.items():
                details[str(k)] = str(v)[:300]
        raise_cron_alert_sync(
            source=source,
            error_type=fingerprint,
            severity="P1",
            title=f"Scheduler exception in {endpoint}",
            details=details,
        )
    except Exception as _alert_err:
        logger.warning(
            "_route_exception_to_p1 fail-open (endpoint=%s exc_class=%s): %r",
            endpoint,
            type(exc).__name__,
            _alert_err,
        )

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

# phase-30.1: sibling of _watchdog_last_was_healthy. Tracks the out-of-band
# autonomous-cycle heartbeat staleness state. State transitions:
#   None -> True   first probe found stale       -> P1 alert
#   False -> True  fresh -> stale transition     -> P1 alert
#   True -> False  stale -> fresh transition     -> P3 recovery
#   None -> False  first probe found fresh       -> silent baseline
#   True -> True   steady-stale                  -> silent (no spam)
#   False -> False steady-fresh                  -> silent
# Audit basis: handoff/archive/phase-30.0/experiment_results.md Anomaly C
# (65h 34m gap 2026-05-17 -> 2026-05-19 with no operator alert path).
_cycle_heartbeat_last_was_stale: bool | None = None

# phase-60.4 (criterion 2): state-transition gate for the inbound-ingestion
# silence alarm (the 6-week tickets.db outage class). None = unknown baseline.
_ingestion_silence_last_was_stale: bool | None = None


def _cycle_state_line() -> str:
    """phase-60.4 (criterion 3): busy-vs-down context for watchdog alerts.

    Reads the autonomous-cycle lockfile (backend/services/cycle_lock.py --
    importable cross-process; the lock lives on disk). Pure-read, never
    raises; the line is APPENDED to alerts, never used to suppress them.
    """
    try:
        from backend.services.cycle_lock import inspect_lock

        lock = inspect_lock()
        if lock is None:
            return "Cycle state: no cycle running (lockfile absent) -- backend looks DOWN, not busy."
        if lock.get("is_stale"):
            return "Cycle state: STALE lock (age > TTL or pid dead) -- backend looks DOWN, not busy."
        started = lock.get("started_at") or lock.get("ts") or "?"
        return (
            f"Cycle state: cycle IN PROGRESS (started {started}, age {lock.get('age_sec', 0):.0f}s, "
            f"pid alive) -- likely BUSY, not down; the away week's ReadTimeouts were this class."
        )
    except Exception as exc:
        return f"Cycle state: unavailable ({type(exc).__name__})"


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

    # phase-47.1: catch-up-on-start. The daily price ingest (hour=1 UTC) is
    # lost whenever the slack-bot is down/asleep at that tick (in-memory
    # jobstore). Schedule a one-off run shortly after startup so downtime
    # self-heals. ingest_prices is idempotent at the BQ level (dedup on
    # (ticker, date)), so a redundant catch-up after a same-day cron fire
    # inserts ~0 rows; within a single process lifetime the daily heartbeat
    # key also skips the duplicate (the store is per-process, in-memory).
    # Keeps historical_prices fresh across restarts WITHOUT a persistent
    # jobstore (which cannot pickle the production-fn closures -- see brief).
    try:
        from backend.slack_bot.jobs.daily_price_refresh import (
            run_production as _price_run_production,
        )

        _scheduler.add_job(
            _price_run_production,
            "date",
            run_date=datetime.now(timezone.utc) + timedelta(seconds=20),
            id="daily_price_refresh_catchup",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info(
            "phase-47.1: scheduled daily_price_refresh catch-up (+20s, idempotent by day)"
        )
    except Exception as exc:
        logger.warning("phase-47.1 price catch-up scheduling fail-open: %r", exc)


def _is_us_trading_day_now() -> bool:
    """phase-51.3: True iff TODAY (ET -- the digest cron tz) is a US trading
    session. Gates the morning/evening digests so they skip weekends AND market
    holidays -- they fired 7 days/week and re-sent the prior trading day's data on
    Sat/Sun (operator-reported). Fail-open: is_trading_day returns True if
    exchange_calendars is unavailable, so a calendar-lib error never suppresses a
    digest. APScheduler has no holiday support, so this in-body guard (not
    day_of_week='mon-fri') is required to cover holidays too."""
    from backend.backtest.markets import is_trading_day
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    return is_trading_day(et_today, "US")


async def _compute_cron_health(client: httpx.AsyncClient) -> str | None:
    """phase-54.2: derive a one-line cron-health summary from /api/jobs/all for the
    morning digest (operator-remote week). Fail-open: returns None on ANY error so a
    jobs-endpoint hiccup never blocks the operator's daily lifeline digest."""
    try:
        res = await client.get(f"{_LOCAL_BACKEND_URL}/api/jobs/all")
        if res.status_code != 200:
            return None
        jobs = res.json().get("jobs", [])
        if not jobs:
            return None
        bad = [j.get("id", "?") for j in jobs if j.get("status") == "failed"]
        n_green = sum(1 for j in jobs if j.get("status") in ("ok", "scheduled", "running"))
        line = (
            f":warning: *Crons:* {n_green}/{len(jobs)} healthy -- FAILED: {', '.join(bad)}"
            if bad else f":white_check_mark: *Crons:* {n_green}/{len(jobs)} healthy"
        )
        return line + _meta_scorer_state_line()
    except Exception:
        return None


def _meta_scorer_state_line() -> str:
    """phase-60.4 (criterion 4): surface the meta-scorer fallback state in
    the morning digest. The away week ran EVERY trade on 'conviction 10.00;
    fallback (LLM unavailable)' and the digest presented those convictions
    as LLM judgments. Reads the latest terminal cycle row from the cycle
    ledger (file-based, cross-process). Empty string when healthy/unknown
    (digest byte-identical)."""
    try:
        from backend.services.cycle_health import get_log

        rows = get_log().last_cycles(1)
        if rows and rows[0].get("meta_scorer_degraded"):
            return (
                "\n:warning: *Meta-scorer:* DEGRADED last cycle -- conviction values "
                "were no-LLM fallbacks (raw composite scores), not LLM judgments."
            )
    except Exception:
        pass
    return ""


async def _compute_system_state(client: httpx.AsyncClient) -> str | None:
    """phase-54.2 cycle-2: kill-switch + go-live-gate state for the morning digest --
    the most decision-relevant away-week signal ("is the system halted?"). Fail-open:
    returns None (or a partial line) on ANY error so it never blocks the lifeline."""
    lines: list[str] = []
    try:
        ks = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/kill-switch")
        if ks.status_code == 200:
            k = ks.json()
            br = k.get("breach", {}) or {}
            if k.get("paused"):
                reason = k.get("pause_reason")
                lines.append(":octagonal_sign: *Kill switch:* PAUSED"
                             + (f" -- {reason}" if reason else ""))
            elif br.get("any_breached"):
                lines.append(":red_circle: *Kill switch:* BREACH")
            else:
                lines.append(":large_green_circle: *Kill switch:* ACTIVE")
            d, t = br.get("daily_loss_pct"), br.get("trailing_dd_pct")
            dl, tl = br.get("daily_loss_limit_pct"), br.get("trailing_dd_limit_pct")
            if d is not None and t is not None and dl is not None and tl is not None:
                lines[-1] += f" (daily {d:+.1f}%/{dl:.0f}% | trail {t:+.1f}%/{tl:.0f}%)"
    except Exception:
        pass
    try:
        g = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/gate")
        if g.status_code == 200:
            gd = g.json()
            b = gd.get("booleans", {}) or {}
            npass = sum(1 for v in b.values() if v)
            elig = "ELIGIBLE" if gd.get("promote_eligible") else "NOT ELIGIBLE"
            lines.append(f"*Go-live gate:* {elig} ({npass}/{len(b)})")
    except Exception:
        pass
    return "\n".join(lines) if lines else None


def _steps_closed_from_log(lines, today: str) -> list[str]:
    """phase-62.8 cycle-2: steps CLOSED today = PASS cycles only. The first
    live read-back listed a CONDITIONAL step as closed (61.1) -- result=PASS
    is the closure marker; CONDITIONAL/FAIL cycles are in-flight, not closed."""
    steps = []
    for line in lines:
        if (line.startswith("## Cycle") and today in line
                and "phase=" in line and "result=PASS" in line):
            steps.append(line.split("phase=", 1)[1].split()[0])
    return steps


async def _gather_away_data(client: httpx.AsyncClient) -> dict:
    """phase-62.8: gather the away-mode digest inputs. Every source fail-open
    (a missing source renders its section's explicit empty state)."""
    import asyncio as _asyncio
    import json as _json
    import subprocess as _subprocess
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    from backend.slack_bot.formatters import _aggregate_trades_by_market

    repo = _Path(__file__).parent.parent.parent
    d: dict = {}

    # 1. trades by market (limit bumped: the digest default of 10 undercounts)
    try:
        r = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=200&since_today=true")
        raw = r.json() if r.status_code == 200 else {}
        trades = raw.get("trades", []) if isinstance(raw, dict) else raw
        d["trades_by_market"] = _aggregate_trades_by_market(trades)
    except Exception:
        logger.exception("away digest: trades gather failed (fail-open)")

    # 2. NAV/risk/kill-switch line (reuses the 54.2 helper)
    try:
        d["system_state_line"] = await _compute_system_state(client)
    except Exception:
        pass

    # 3. shipped today (committer-date local tz; --since-as-filter avoids early
    # traversal stop, git >= 2.37)
    def _git_today() -> list[str]:
        out = _subprocess.run(
            ["git", "log", "--since-as-filter=midnight", "--pretty=%h %s"],
            cwd=repo, capture_output=True, text=True, timeout=20)
        return [l for l in out.stdout.strip().splitlines() if l][:20]
    try:
        d["commits_today"] = await _asyncio.to_thread(_git_today)
    except Exception:
        pass
    try:
        today = _dt.now(_tz.utc).strftime("%Y-%m-%d")
        with open(repo / "handoff" / "harness_log.md", encoding="utf-8") as f:
            d["steps_flipped_today"] = _steps_closed_from_log(f, today)
    except Exception:
        pass

    # 4. pending operator asks (canonical file, 62.3)
    try:
        with open(repo / "handoff" / "away_ops" / "pending_tokens.json", encoding="utf-8") as f:
            asks = _json.load(f).get("asks", [])
        for a in asks:
            try:
                raised = _dt.fromisoformat(str(a.get("raised_at", ""))[:10])
                a["age_days"] = (_dt.now() - raised).days
            except Exception:
                a["age_days"] = None
        d["pending_asks"] = asks
    except Exception:
        pass

    # 5. last health line (62.5 writer; absent until it ships)
    try:
        lines = (repo / "handoff" / "away_ops" / "health.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if lines:
            d["health"] = _json.loads(lines[-1])
    except Exception:
        pass
    try:
        slog = (repo / "handoff" / "away_ops" / "session.log").read_text(encoding="utf-8").strip().splitlines()
        ends = [l for l in slog if "END session" in l]
        if ends:
            d["am_session_result"] = ends[-1]
    except Exception:
        pass

    # 6. defect-register delta (63.3 file; absent until it ships)
    try:
        reg = (repo / "handoff" / "away_ops" / "defect_register.md").read_text(encoding="utf-8")
        d["defect_counts"] = {
            "P0": reg.count("| P0 |"), "P1": reg.count("| P1 |"),
            "P2": reg.count("| P2 |"), "fixed": reg.count("FIXED"),
        }
    except Exception:
        pass

    return d


async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    # phase-51.3: skip on non-trading days (weekend/holiday) -- no fresh data, so
    # the digest would re-send the prior trading day's rows. Slack-bot only.
    if not _is_us_trading_day_now():
        logger.info("morning_digest skipped: %s ET is not a US trading day",
                    datetime.now(ZoneInfo("America/New_York")).date())
        return

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            reports_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/reports/?limit=5")
            reports_data = reports_res.json() if reports_res.status_code == 200 else []

            # phase-54.2: cron-health + system-state lines for the remote week (fail-open).
            cron_health = await _compute_cron_health(client)
            system_state = await _compute_system_state(client)

            # phase-62.8: compact away sections (asks + health), flag-gated.
            away_sections = None
            if settings.away_mode_enabled:
                try:
                    from backend.slack_bot.formatters import format_away_compact_sections
                    away_sections = format_away_compact_sections(await _gather_away_data(client))
                except Exception:
                    logger.exception("away compact sections failed (fail-open)")

        blocks = format_morning_digest(portfolio_data, reports_data, cron_health=cron_health, system_state=system_state, away_sections=away_sections)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Morning Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Morning digest sent")

    except Exception as exc:
        logger.exception("Failed to send morning digest")
        _route_exception_to_p1(exc, endpoint="morning_digest")


async def _send_evening_digest(app: AsyncApp):
    """Fetch end-of-day portfolio summary and post evening digest."""
    settings = get_settings()

    # phase-51.3: skip on non-trading days (weekend/holiday). Slack-bot only.
    if not _is_us_trading_day_now():
        logger.info("evening_digest skipped: %s ET is not a US trading day",
                    datetime.now(ZoneInfo("America/New_York")).date())
        return

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            # phase-71 cycle (2026-05-26): pass `since_today=true` so the
            # "Today's Trades" section actually reflects today's rows. The
            # default dateless query returned the latest N rows forever,
            # producing the same 10-trade list for 9 consecutive days. The
            # else-branch in format_evening_digest already prints "No trades
            # executed today." when the list is empty.
            trades_res = await client.get(
                f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=10&since_today=true"
            )
            # phase-23.5.7.1: /api/paper-trading/trades returns the dict envelope
            # {"trades": [...], "count": N} (paper_trading.py:226). Unwrap at the
            # HTTP boundary so format_evening_digest's `trades_today[:10]` slice
            # gets a list, not a dict (which raises KeyError: slice(...)).
            _raw = trades_res.json() if trades_res.status_code == 200 else []
            trades_data = _raw.get("trades", []) if isinstance(_raw, dict) else _raw

            # phase-62.8: full away sections, flag-gated (OFF path byte-identical).
            away_sections = None
            if settings.away_mode_enabled:
                try:
                    from backend.slack_bot.formatters import format_away_digest_sections
                    away_sections = format_away_digest_sections(await _gather_away_data(client))
                except Exception:
                    logger.exception("away digest sections failed (fail-open)")

        blocks = format_evening_digest(portfolio_data, trades_data, away_sections=away_sections)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Evening Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Evening digest sent")

    except Exception as exc:
        logger.exception("Failed to send evening digest")
        _route_exception_to_p1(exc, endpoint="evening_digest")


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
        # phase-56.2 (55.2 F-C bounded fix): 10s -> 30s. The nightly
        # "unreachable/ReadTimeout" alerts were the 18:00Z trading cycle
        # transiently starving the backend's event loop (it was never down;
        # every FAIL was 1/3). 30s mirrors the digest probes' timeout so a
        # busy-but-alive backend does not read as an outage.
        async with httpx.AsyncClient(timeout=30.0) as client:
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
            # phase-60.4 (AW-2 residual, criterion 3): distinguish busy-vs-down.
            # The away week's single-probe ReadTimeouts (05-27/05-28/06-04) were
            # CYCLES IN PROGRESS hogging the event loop, not a dead backend --
            # the alert text now says which. Context is APPENDED, never used to
            # suppress the alert (a busy backend that drops /api/health is
            # still worth knowing about).
            post = (
                f":rotating_light: *Watchdog Alert* -- Backend unreachable\n"
                f"Detail: {detail} at {datetime.now().strftime('%H:%M:%S')}\n"
                f"{_cycle_state_line()}",
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

    # phase-60.4 (criterion 2): inbound-ingestion silence alarm (dead-man's-
    # switch pattern). tickets.db went dark 2026-04-24 -> 2026-06-10 and the
    # operator learned of it from an audit. Same state-transition spam gate
    # as the heartbeat below; threshold settings.ticket_ingestion_silence_days
    # (default 7). Fully fail-open.
    global _ingestion_silence_last_was_stale
    try:
        from backend.db.tickets_db import TicketsDB

        _silence_days = float(getattr(settings, "ticket_ingestion_silence_days", 7) or 7)
        _age = TicketsDB().get_last_ticket_age_days()
        _silent_now = _age is not None and _age >= _silence_days
        _prior_silent = _ingestion_silence_last_was_stale
        _ingestion_silence_last_was_stale = _silent_now
        if _silent_now and _prior_silent is not True:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": (
                        f":rotating_light: *Ingestion Silence Alarm* -- no inbound ticket "
                        f"ingested for {_age:.1f} days (threshold {_silence_days:.0f}d).\n"
                        f"The operator channel -> tickets.db pipeline may be dead "
                        f"(the 2026-04-24 -> 06-10 outage class). Check the slack bot "
                        f"listener + backend/services/ticket_ingestion.py."
                    )},
                }],
                text="Ingestion Silence Alarm: inbound ticket pipeline may be dead",
            )
            logger.warning("ingestion_silence alarm fired -- age_days=%.1f", _age)
        elif (not _silent_now) and _prior_silent is True:
            logger.info("ingestion_silence recovered -- age_days=%s", _age)
    except Exception as exc:
        logger.warning("ingestion_silence check fail-open: %r", exc)

    # phase-30.1: out-of-band autonomous-cycle heartbeat check. Runs on
    # the same interval cron as the backend health probe so a missing
    # daily cycle is detected even when the backend itself is healthy
    # (the failure mode documented in phase-30.0 Anomaly C: 65h gap
    # 2026-05-17 -> 2026-05-19, backend up, cron skipped). Fully
    # fail-open: any exception is logged and swallowed so it cannot
    # take down the watchdog cron.
    global _cycle_heartbeat_last_was_stale
    try:
        from backend.services.cycle_health import (
            cycle_heartbeat_alarm,
            fire_cycle_heartbeat_alarm,
        )
        verdict = cycle_heartbeat_alarm()
        is_stale_now = bool(verdict.get("should_alarm"))
        prior_stale = _cycle_heartbeat_last_was_stale
        _cycle_heartbeat_last_was_stale = is_stale_now

        if is_stale_now and prior_stale is not True:
            # None -> True or False -> True : fire one P1 alert.
            fire_cycle_heartbeat_alarm(verdict)
            logger.warning(
                "cycle_heartbeat_alarm fired P1 -- age_sec=%s last_completed=%s",
                verdict.get("age_sec"),
                verdict.get("last_completed_at"),
            )
        elif (not is_stale_now) and prior_stale is True:
            # True -> False : recovery. Log only (no P1; P3-grade signal).
            logger.info(
                "cycle_heartbeat recovery -- last_completed=%s",
                verdict.get("last_completed_at"),
            )
        else:
            # Steady-state or first-fresh: silent.
            logger.debug(
                "cycle_heartbeat steady -- stale=%s age_sec=%s weekday_et=%s",
                is_stale_now,
                verdict.get("age_sec"),
                verdict.get("is_weekday_et"),
            )
    except Exception as exc:
        logger.warning("cycle_heartbeat_alarm watchdog fail-open: %r", exc)


def pause_signals(app: "AsyncApp | None" = None) -> bool:
    """Shut down the scheduler, stopping all signal-related jobs.

    Returns True if the scheduler was running and is now stopped,
    False if it was already stopped or never started.
    This is the rollback command for Go-Live checklist item 4.4.6.4.

    phase-25.K: when `app` is provided, fires a P0 Slack escalation
    BEFORE the shutdown via send_trading_escalation. Closes phase-24.5
    F-5(b) audit finding (pause_signals only logged INFO; no Slack post).
    Falls back silently if app is None (preserves callers that use
    pause_signals as a pure rollback without Slack wiring).
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        # phase-25.K: notify Slack BEFORE shutting down so the alert routes
        # through the still-running app instance. asyncio.create_task is
        # fire-and-forget; failures are logged but do not block rollback.
        if app is not None:
            try:
                asyncio.create_task(notify_kill_switch_activated(
                    app=app,
                    trigger="manual_rollback_via_pause_signals",
                    details={
                        "caller": "pause_signals",
                        "scheduler_state": "running",
                        "go_live_checklist_item": "4.4.6.4",
                    },
                ))
            except Exception as exc:
                logger.exception("phase-25.K: failed to schedule kill-switch Slack alert")
                _route_exception_to_p1(exc, endpoint="kill_switch_alert_schedule")
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down -- all signal jobs paused (rollback 4.4.6.4)")
        return True
    logger.info("Scheduler was not running -- no action taken")
    return False


async def notify_kill_switch_activated(
    app: AsyncApp,
    trigger: str,
    details: dict,
) -> None:
    """Notify Slack + iMessage of kill-switch activation (P0 severity).

    phase-25.K: thin wrapper around send_trading_escalation specifically
    for the kill-switch use case. Callers should be anywhere the kill
    switch fires: pause_signals(), paper_trader.check_and_enforce_kill_switch(),
    or operator-triggered flatten endpoints. Closes phase-24.5 F-5(b) audit.
    """
    await send_trading_escalation(
        app=app,
        severity="P0",
        title="Kill Switch Activated",
        details={"trigger": trigger, **details},
        actions=[
            "Inspect handoff/kill_switch_audit.jsonl for full breach details",
            "Run /portfolio to confirm positions are flat",
            "Investigate root cause before resume",
        ],
    )


async def notify_trade_confirmation(
    app: AsyncApp,
    trade: dict,
) -> None:
    """phase-25.J: post a trade confirmation to the configured Slack channel.

    Receives the trade dict shape returned by paper_trader.execute_buy/sell.
    Closes phase-24.5 audit F-5(a). Cross-process delivery (when paper_trader
    runs in the backend process and slack_bot is separate) is the future
    25.J.1 follow-up that polls BQ paper_trades for new rows; this function
    is the in-process building block.
    """
    settings = get_settings()
    if not settings.slack_channel_id:
        return

    from backend.slack_bot.formatters import format_trade_confirmation

    blocks = format_trade_confirmation(trade)
    action = str(trade.get("action") or "TRADE").upper()
    ticker = str(trade.get("ticker") or "?")
    text_fallback = f"{action} {ticker} (paper)"

    try:
        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=text_fallback,
        )
        logger.info("Trade confirmation posted: %s", text_fallback)
    except Exception:
        logger.exception("phase-25.J: trade confirmation post failed for %s", text_fallback)


async def notify_kill_switch_deactivated(
    app: AsyncApp,
    reason: str,
) -> None:
    """Notify Slack of kill-switch resume (P1 severity).

    phase-25.K: lighter-severity counterpart to notify_kill_switch_activated.
    Use after operator-triggered resume to inform the team that
    autonomous trading is back online.
    """
    await send_trading_escalation(
        app=app,
        severity="P1",
        title="Kill Switch Resumed",
        details={"reason": reason},
        actions=["Monitor next autonomous cycle for healthy completion"],
    )


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
    except Exception as exc:
        logger.exception(f"Failed to send alert for {ticker}")
        _route_exception_to_p1(exc, endpoint="ticker_alert", extra={"ticker": ticker})


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
                # phase-47.1: daily_price_refresh no longer uses the close-only
                # fetch/write closures (they wrote the WRONG table
                # pyfinagent_data.price_snapshots). It is registered below as the
                # module-level `run_production` (full-universe OHLCV ->
                # financial_reports.historical_prices via ingest_prices).
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
            # phase-25.M: was logger.warning, swallowed silently in
            # default views. Promote to ERROR + exc_info so a factory
            # wiring failure surfaces visibly (audit bucket 24.5 F-5(d)).
            logger.error(
                "register_phase9_jobs: production-fn wiring failed: %r", exc, exc_info=True
            )
            prod_fns_per_job = {}
    # phase-23.3.3: include APScheduler safety params per researcher's brief.
    # `misfire_grace_time` prevents stale-tick fires on restart;
    # `coalesce=True` collapses missed ticks into one fire (defensive for
    # any future jobstore migration; harmless with default in-memory store).
    # phase-47.1: pin every phase-9 cron to an explicit UTC timezone (the MAIN
    # jobs above already pin America/New_York; these previously inherited the
    # ambiguous system default). daily_price_refresh misfire grace raised to 6h
    # so a Mac-asleep tick is still caught on wake; the catch-up-on-start in
    # start_scheduler is the primary restart-survival path.
    mapping = {
        "daily_price_refresh":     ("backend.slack_bot.jobs.daily_price_refresh", "cron",
                                    {"hour": 1, "misfire_grace_time": 21600, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "weekly_fred_refresh":     ("backend.slack_bot.jobs.weekly_fred_refresh", "cron",
                                    {"day_of_week": "sun", "hour": 2, "misfire_grace_time": 7200, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "nightly_mda_retrain":     ("backend.slack_bot.jobs.nightly_mda_retrain", "cron",
                                    {"hour": 3, "misfire_grace_time": 3600, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "hourly_signal_warmup":    ("backend.slack_bot.jobs.hourly_signal_warmup", "cron",
                                    {"minute": 5, "misfire_grace_time": 600, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "nightly_outcome_rebuild": ("backend.slack_bot.jobs.nightly_outcome_rebuild", "cron",
                                    {"hour": 4, "misfire_grace_time": 3600, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "weekly_data_integrity":   ("backend.slack_bot.jobs.weekly_data_integrity", "cron",
                                    {"day_of_week": "mon", "hour": 5, "misfire_grace_time": 7200, "coalesce": True, "timezone": ZoneInfo("UTC")}),
        "cost_budget_watcher":     ("backend.slack_bot.jobs.cost_budget_watcher", "cron",
                                    {"hour": 6, "misfire_grace_time": 3600, "coalesce": True, "timezone": ZoneInfo("UTC")}),
    }
    for job_id, (module_path, trigger, kwargs) in mapping.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            # phase-47.1: daily_price_refresh runs the module-level production
            # entrypoint (full-universe OHLCV ingest); all others use bare `run`.
            attr = "run_production" if job_id == "daily_price_refresh" else "run"
            run_fn = getattr(mod, attr, None) or getattr(mod, "run")
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
