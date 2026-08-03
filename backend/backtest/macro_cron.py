"""phase-82.0 APScheduler wiring for FRED macro ingestion.

THE ROOT CAUSE THIS MODULE EXISTS TO FIX: `ingest_macro`
(`backend/backtest/data_ingestion.py`) had exactly one caller in the entire
repository -- `run_full_ingestion` -- whose own call sites were a
cold-start-only guard (`backtest_engine.py`, gated `if prices_count == 0`,
dead in practice because `historical_prices` is not empty), a manually
operator-triggered API task, and a one-shot migration script. There was no
`add_job`, no launchd plist, no crontab entry anywhere. `git log -S
"ingest_macro"` shows no commit ever removed one.

So `historical_macro` did not BREAK in March 2026. It was never running on a
cadence at all; 2026-03-25 is simply the timestamp of the last manual run.

CORRECTED IN CYCLE-2 (after a Q/A FAIL overturned the original claim here):
the table then sat 128 days stale and `preload_macro()` **cached it anyway**.
The phase-25.D7 staleness guard tested `isinstance(date, datetime.date)`, but
`historical_macro.date` is a STRING column, so the check was vacuous and the
refusal branch never executed -- from the day it was written. Backtests were
therefore silently trained on 212-day-old macro features rather than hanging
on the per-cutoff BQ fallback. Do not repeat the earlier "refused to cache /
40-minute hang" story: it does not reproduce.

Mirrors `backend/meta_evolution/cron.py` (separate cron-module shim, fail-open
registration) and `backend/autoresearch/cron.py`. ASCII-only logger messages
per `.claude/rules/security.md`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JOB_ID = "macro_ingest_daily"
TIMEZONE = ZoneInfo("America/New_York")
# 08:10 ET on weekdays. FRED publishes daily series (DGS10, T10Y2Y) on a
# business-day cadence in the early morning ET; monthly/quarterly releases land
# on their own schedule and are picked up by whichever run follows them. Daily
# beats weekly here only because the dedupe makes a no-op run nearly free.
DEFAULT_DAY_OF_WEEK = "mon-fri"
DEFAULT_HOUR = 8
DEFAULT_MINUTE = 10


def run_macro_ingest(*, bq_client: Any = None, settings: Any = None) -> int:
    """Execute one macro ingestion pass. Returns rows inserted.

    Fail-open at the top level: this runs on a scheduler thread, and an
    exception escaping here would be logged by APScheduler but must never be
    allowed to take down neighbouring jobs. The INNER dedupe check is
    deliberately fail-CLOSED (see `_get_existing_macro`) -- a dedupe failure
    aborts the batch rather than duplicating the table.
    """
    try:
        from backend.config.settings import get_settings
        from backend.db.bigquery_client import BigQueryClient
        from backend.backtest.data_ingestion import DataIngestionService

        settings = settings or get_settings()
        bq = bq_client or BigQueryClient(settings)
        svc = DataIngestionService(bq.client, settings)

        fred_key = getattr(settings, "fred_api_key", "") or ""
        if hasattr(fred_key, "get_secret_value"):
            # NEVER str() a SecretStr -- it renders as asterisks.
            fred_key = fred_key.get_secret_value()

        # start_date is the ingestion floor, not the backtest window. end_date
        # is resolved internally to today; passing backtest_end_date here is
        # exactly the defect this step fixes.
        start = getattr(settings, "backtest_start_date", "2018-01-01")
        inserted = svc.ingest_macro(start, None, fred_key)
        logger.info("macro_ingest_daily: inserted %d rows", inserted)
        return inserted
    except Exception as exc:
        # A construction/import failure happens BEFORE ingest_macro can write
        # its own receipt, so without this the job would fail silently forever
        # -- the exact invisible-failure mode phase-82.0 exists to kill. (This
        # is not hypothetical: the first run of this function died here on a
        # BigQueryClient signature error and reported a clean 0.)
        logger.warning("run_macro_ingest fail-open: %r", exc)
        _write_failure_receipt(repr(exc))
        return 0


def _write_failure_receipt(error: str) -> None:
    """Record a run that died before the ingestion service existed."""
    try:
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "handoff" / "logs"
        path.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "job": "macro_ingest",
            "rows_inserted": 0,
            "outcome": f"failed_before_ingest: {error}",
            "observation_end": "",
        }
        with open(path / "macro_ingest_receipts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # pragma: no cover - observability only
        pass


def register_macro_ingest_cron(
    scheduler: Any,
    *,
    replace_existing: bool = True,
    day_of_week: str = DEFAULT_DAY_OF_WEEK,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
) -> Optional[str]:
    """Register the macro ingestion job on a scheduler-like object.

    `scheduler` must expose `.add_job(func, trigger, id, replace_existing,
    **kwargs)` (BackgroundScheduler, AsyncIOScheduler, StubScheduler).
    Returns the job_id on success or None if add_job raised (fail-open).

    `replace_existing=True` is mandatory per the APScheduler userguide --
    without it every restart duplicates the job in a persistent jobstore.
    """
    try:
        scheduler.add_job(
            run_macro_ingest,
            trigger="cron",
            id=JOB_ID,
            replace_existing=replace_existing,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=TIMEZONE,
        )
    except Exception as exc:
        logger.warning("register_macro_ingest_cron add_job fail-open: %r", exc)
        return None
    logger.info(
        "registered macro_ingest_daily cron: %s %02d:%02d %s",
        day_of_week,
        hour,
        minute,
        TIMEZONE.key,
    )
    return JOB_ID
