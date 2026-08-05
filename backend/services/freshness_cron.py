"""phase-82.10 APScheduler wiring for the data-freshness evaluator.

THE DEFECT THIS MODULE EXISTS TO FIX -- and note it is NOT the one the
masterplan step description states. The step says the freshness alarm "cannot
page". Measured 2026-08-05, that is only half true:

  * The EMITTER already exists and already pages. `_fire_freshness_alarm`
    (`backend/services/cycle_health.py`) dispatches a P1 through
    `raise_cron_alert_sync` for every red source, and `compute_freshness`
    already calls it. The phase-66 hotfix note in
    `backend/services/observability/alerting.py` documents a real page storm
    from that exact alarm ("~120 pages/hour the moment a dashboard tab was
    open against a red table") -- direct evidence the channel was live.
  * What was missing is the TRIGGER. Every production caller of
    `compute_freshness` was an HTTP handler that only the frontend calls
    (`backend/api/paper_trading.py`, `backend/api/observability_api.py` x2).
    Nothing polled them. So a correct alarm fired into an empty room for 128
    days and the operator learned about the dead FRED feed from a manual
    investigation. An alarm that only fires when a human happens to open a
    browser tab is not a monitor.

THE TRAP A NAIVE FIX WALKS INTO. Simply calling `compute_freshness` on a timer
reintroduces the phase-66 page storm in slow motion. `AlertDeduper` does NOT
suppress steady state: a P1 severity bypasses the consecutive-occurrence
threshold, but the per-(source, error_type) REPEAT WINDOW still applies, so the
same alert re-fires every `alert_repeat_hours` forever. Measured directly:

    AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=3)
    five back-to-back P1 should_fire calls -> [True, False, False, False, False]
    ...then rewind last_fired_at by 1h1m    -> True

On a table that stays red for 128 days that is roughly 512 pages. So this
module owns a STATE-TRANSITION gate (fire only when a source becomes newly
red) and passes `emit_alarm=False` to suppress the inner level-triggered path.
Pattern borrowed from the existing watchdog gates in
`backend/slack_bot/scheduler.py` (`_cycle_heartbeat_last_was_stale` /
`_ingestion_silence_last_was_stale`), which exist for exactly this reason;
`cycle_health.cycle_heartbeat_alarm`'s docstring already states that the
CALLER, not the alarm, owns transition gating.

Mirrors `backend/backtest/macro_cron.py` (the phase-82.0 sibling),
`backend/meta_evolution/cron.py` and `backend/harness_self_audit_report.py`:
a separate cron-module shim with fail-open registration. ASCII-only logger
messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

JOB_ID = "freshness_evaluator"

# Cadence rationale (dbt source-freshness guidance: check at >= 2x the
# frequency of the tightest SLA). The tightest SLA in
# `cycle_health._TABLE_MAX_AGE_SEC` is 26h (historical_prices,
# paper_portfolio_snapshots), so any cadence <= 13h suffices. 6h leaves
# margin without approaching the alert-fatigue zone. A minutes-scale interval
# would buy nothing -- no source here has an SLA shorter than 26h -- while
# multiplying the blast radius of any gating mistake.
DEFAULT_INTERVAL_HOURS = 6

# Module-level transition state: the set of source names that were red on the
# previous evaluation. `None` means "no baseline yet" (fresh process), which
# deliberately pages for a red source on the first run after a restart -- on
# restart the operator SHOULD be told a table is red. The backend runs under
# launchd 24/7, so restarts are rare and operator-initiated.
_last_red_sources: Optional[set[str]] = None


def reset_transition_state() -> None:
    """Test seam: drop the remembered red set so a fixture starts clean.

    Without this, a test that drives a red fixture and then a green one would
    have its second assertion pass for the wrong reason (the gate, not the
    band, suppressing the alert).
    """
    global _last_red_sources
    _last_red_sources = None


def _red_sources(payload: dict) -> set[str]:
    """Names of sources whose band is red in a compute_freshness payload.

    Reads the `sources` mapping that `compute_freshness` returns rather than
    re-deriving bands: `_TABLE_MAX_AGE_SEC` covers only 4 of the 6 monitored
    tables (paper_trades and signals_log use the caller-supplied cycle
    interval), so any re-derivation here would drift from the dashboard.
    """
    sources = payload.get("sources") or {}
    return {
        str(name)
        for name, info in sources.items()
        if isinstance(info, dict) and info.get("band") == "red"
    }


def run_freshness_check(
    *,
    bq: Any = None,
    settings: Any = None,
    notify: Optional[Callable[..., Any]] = None,
) -> dict:
    """Execute one scheduled freshness evaluation.

    Returns a summary dict: `{"overall_band", "red_sources", "newly_red",
    "alerts_emitted", "ok"}`. Fail-open at the top level -- this runs on a
    scheduler thread and an exception escaping here must never take down
    neighbouring jobs.

    `bq`, `settings` and `notify` are injectable so tests can drive the real
    code path with fakes instead of reaching BigQuery or Slack.
    """
    global _last_red_sources
    try:
        from backend.services.cycle_health import compute_freshness

        if settings is None:
            from backend.config.settings import get_settings

            settings = get_settings()
        if bq is None:
            from backend.db.bigquery_client import get_bq_client

            bq = get_bq_client()
        if notify is None:
            from backend.services.observability.alerting import (
                raise_cron_alert_sync,
            )

            notify = raise_cron_alert_sync

        # Identical expression to the three HTTP call sites so the pager and
        # the dashboard can never disagree about a band. NOTE: there is no
        # `paper_cycle_interval_sec` field in settings -- measured 2026-08-05,
        # all four sites resolve to the 86400.0 default. Not this step's
        # defect to fix; mirroring it is what keeps the two consistent.
        cycle_interval_sec = float(
            getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0)
        )

        # emit_alarm=False: THIS function owns the transition gate. See the
        # module docstring for why the inner level-triggered path must not
        # also fire.
        payload = compute_freshness(bq, cycle_interval_sec, emit_alarm=False)

        red_now = _red_sources(payload)
        baseline = _last_red_sources
        newly_red = red_now if baseline is None else (red_now - baseline)
        _last_red_sources = red_now

        emitted = 0
        sources = payload.get("sources") or {}
        for table_name in sorted(newly_red):
            info = sources.get(table_name) or {}
            try:
                ratio = info.get("ratio")
                ratio_str = (
                    f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "N/A"
                )
                notify(
                    source="cycle_health",
                    error_type=f"freshness_critical_{table_name}",
                    severity="P1",
                    title=f"Data freshness critical: {table_name}",
                    details={
                        "table": table_name,
                        "last_tick_age_sec": str(info.get("last_tick_age_sec")),
                        "interval_sec": str(info.get("interval_sec")),
                        "ratio": ratio_str,
                        "band": info.get("band"),
                        "detected_by": JOB_ID,
                    },
                )
                emitted += 1
            except Exception as exc:
                logger.warning(
                    "freshness_evaluator: dispatch fail-open for %s: %r",
                    table_name,
                    exc,
                )

        still_red = sorted(red_now - newly_red)
        if still_red:
            # Deliberately log-only. Re-paging a source that has been red
            # since the last tick is the alert-fatigue failure mode this
            # module exists to avoid.
            logger.warning(
                "freshness_evaluator: still red (no page, steady state): %s",
                ",".join(still_red),
            )
        recovered = sorted((baseline or set()) - red_now)
        if recovered:
            logger.info(
                "freshness_evaluator: recovered to non-red: %s",
                ",".join(recovered),
            )

        logger.info(
            "freshness_evaluator: overall=%s red=%d newly_red=%d alerts=%d",
            payload.get("overall_band"),
            len(red_now),
            len(newly_red),
            emitted,
        )
        return {
            "ok": True,
            "overall_band": payload.get("overall_band"),
            "red_sources": sorted(red_now),
            "newly_red": sorted(newly_red),
            "alerts_emitted": emitted,
        }
    except Exception as exc:
        # A construction/import failure happens before any receipt is written,
        # so without this the job fails silently forever -- the exact
        # invisible-failure mode this step exists to kill.
        logger.warning("run_freshness_check fail-open: %r", exc)
        return {
            "ok": False,
            "overall_band": None,
            "red_sources": [],
            "newly_red": [],
            "alerts_emitted": 0,
            "error": repr(exc),
        }


def register_freshness_cron(
    scheduler: Any,
    *,
    replace_existing: bool = True,
    hours: int = DEFAULT_INTERVAL_HOURS,
) -> Optional[str]:
    """Register the freshness evaluator on a scheduler-like object.

    `scheduler` must expose `.add_job(func, trigger, id, replace_existing,
    **kwargs)` (AsyncIOScheduler, BackgroundScheduler, StubScheduler).
    Returns the job_id on success or None if add_job raised (fail-open --
    a scheduler problem must never break backend startup).

    `replace_existing=True` is mandatory per the APScheduler userguide;
    without it a restart duplicates the job in a persistent jobstore.
    """
    try:
        scheduler.add_job(
            run_freshness_check,
            trigger="interval",
            id=JOB_ID,
            replace_existing=replace_existing,
            hours=hours,
            name="Data freshness evaluator (pages on newly-red sources)",
        )
    except Exception as exc:
        logger.warning("register_freshness_cron add_job fail-open: %r", exc)
        return None
    logger.info("registered %s: every %dh", JOB_ID, hours)
    return JOB_ID


__all__ = [
    "JOB_ID",
    "DEFAULT_INTERVAL_HOURS",
    "register_freshness_cron",
    "reset_transition_state",
    "run_freshness_check",
]
