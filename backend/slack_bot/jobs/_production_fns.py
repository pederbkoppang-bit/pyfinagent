"""phase-23.6.1: production fetch/write/alert factory closures for phase-9 jobs.

Each factory returns a sync closure matching the job's `fetch_fn` / `write_fn`
/ `alert_fn` signature. Closures fail-open: a real-world error (yfinance 429,
FRED missing key, BQ quota, Slack rate limit) logs WARNING and returns an
empty result. The heartbeat then surfaces `written=0` honestly so the
operator sees the problem on the dashboard — NOT a silent stub-fallback.

External dependencies (yfinance, fredapi, google.cloud.bigquery) are imported
INSIDE each closure body so:
- Module import is fast.
- Tests that assert "no live yfinance call" still pass.
- A missing optional dependency only breaks the specific job that needs it.

The `make_alert_fn_for_*` factories bridge the sync APScheduler job context to
the async Slack `chat_postMessage` API via
`asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10)` — the only
working pattern when `start_scheduler` runs inside an already-running event
loop (Cosmic Python ch. 13, Slack Bolt async docs).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover -- type-only import
    from slack_bolt.async_app import AsyncApp

logger = logging.getLogger(__name__)


# ── price + FRED + ledger / outcome BQ helpers ─────────────────


def _bq_client():
    """Lazy BQ client; preserves test isolation when no creds available."""
    from google.cloud import bigquery
    project = os.environ.get("GCP_PROJECT_ID", "sunny-might-477607-p8")
    return bigquery.Client(project=project)


# ── daily_price_refresh ────────────────────────────────────────
# phase-47.1 DEPRECATION: these close-only factories wrote to the WRONG table
# (pyfinagent_data.price_snapshots -- no `ingested_at`, only 5 tickers) and are
# NO LONGER WIRED for daily_price_refresh. The job now runs
# `daily_price_refresh.run_production` (full-universe OHLCV ->
# financial_reports.historical_prices via DataIngestionService.ingest_prices).
# Kept for back-compat / existing tests; do NOT re-wire without fixing the
# destination table + schema (full OHLCV + ingested_at).


def make_price_fetch_fn() -> Callable[[list[str]], dict[str, Any]]:
    """Return a sync closure that fetches latest OHLCV per ticker via yfinance."""

    def _fetch(tickers: list[str]) -> dict[str, Any]:
        try:
            import yfinance as yf  # lazy import
        except ImportError as exc:
            logger.warning("daily_price_refresh: yfinance not installed: %r", exc)
            return {}
        out: dict[str, Any] = {}
        try:
            df = yf.download(tickers, period="2d", progress=False, threads=False)
        except Exception as exc:  # network, rate-limit, schema change
            logger.warning("daily_price_refresh: yfinance download fail-open: %r", exc)
            return {}
        # yfinance returns a multi-index DataFrame for >1 ticker; for 1 ticker
        # it's a single-level frame. Normalise to per-ticker dict.
        try:
            for ticker in tickers:
                col = (slice(None), ticker) if hasattr(df.columns, "levels") and len(df.columns.levels) >= 2 else slice(None)
                try:
                    last = df["Close"][ticker].dropna().iloc[-1] if hasattr(df.columns, "levels") and len(df.columns.levels) >= 2 else df["Close"].dropna().iloc[-1]
                except Exception:
                    continue
                out[ticker] = {"close": float(last)}
        except Exception as exc:
            logger.warning("daily_price_refresh: parse fail-open: %r", exc)
            return out
        return out

    return _fetch


def make_price_write_fn() -> Callable[[dict[str, Any]], int]:
    """Return a sync closure that writes price snapshots to BQ.

    Schema: `pyfinagent_data.price_snapshots(ticker STRING, date STRING,
    close FLOAT, recorded_at TIMESTAMP)`. If the table does not exist, the
    closure logs WARNING and returns 0; operator can create it via a
    migration.
    """

    def _write(rows: dict[str, Any]) -> int:
        if not rows:
            return 0
        today = date.today().isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()
        records = [
            {
                "ticker": ticker,
                "date": today,
                "close": payload.get("close"),
                "recorded_at": now_iso,
            }
            for ticker, payload in rows.items()
            if payload.get("close") is not None
        ]
        if not records:
            return 0
        try:
            client = _bq_client()
            table_id = f"{client.project}.pyfinagent_data.price_snapshots"
            errors = client.insert_rows_json(table_id, records)
            if errors:
                logger.warning("daily_price_refresh: BQ insert errors: %r", errors)
                return 0
            return len(records)
        except Exception as exc:
            logger.warning("daily_price_refresh: BQ write fail-open: %r", exc)
            return 0

    return _write


# ── weekly_fred_refresh ────────────────────────────────────────


def make_fred_fetch_fn() -> Callable[[list[str]], dict[str, Any]]:
    """Return a sync closure that fetches FRED series via fredapi."""

    def _fetch(series: list[str]) -> dict[str, Any]:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.warning("weekly_fred_refresh: FRED_API_KEY not set")
            return {s: [] for s in series}
        try:
            from fredapi import Fred  # lazy import
        except ImportError as exc:
            logger.warning("weekly_fred_refresh: fredapi not installed: %r", exc)
            return {s: [] for s in series}
        out: dict[str, Any] = {}
        try:
            fred = Fred(api_key=api_key)
            cutoff = (date.today() - timedelta(days=14)).isoformat()
            for s in series:
                try:
                    ser = fred.get_series(s, observation_start=cutoff)
                    out[s] = [
                        {"date": str(idx.date()), "value": float(val)}
                        for idx, val in ser.dropna().items()
                    ]
                except Exception as exc:
                    logger.warning("weekly_fred_refresh: %s fail-open: %r", s, exc)
                    out[s] = []
        except Exception as exc:
            logger.warning("weekly_fred_refresh: Fred client fail-open: %r", exc)
            return {s: [] for s in series}
        return out

    return _fetch


def make_fred_write_fn() -> Callable[[dict[str, Any]], int]:
    """Return a sync closure that writes FRED observations to BQ.

    Schema: `pyfinagent_data.fred_observations(series STRING, date STRING,
    value FLOAT, recorded_at TIMESTAMP)`. Same fail-open policy as price-write.
    """

    def _write(rows: dict[str, Any]) -> int:
        if not rows:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        for series_id, observations in rows.items():
            for obs in observations or []:
                records.append({
                    "series": series_id,
                    "date": obs.get("date"),
                    "value": obs.get("value"),
                    "recorded_at": now_iso,
                })
        if not records:
            return 0
        try:
            client = _bq_client()
            table_id = f"{client.project}.pyfinagent_data.fred_observations"
            errors = client.insert_rows_json(table_id, records)
            if errors:
                logger.warning("weekly_fred_refresh: BQ insert errors: %r", errors)
                return 0
            return len(records)
        except Exception as exc:
            logger.warning("weekly_fred_refresh: BQ write fail-open: %r", exc)
            return 0

    return _write


# ── nightly_outcome_rebuild ────────────────────────────────────


#: phase-82.39. The window the production closure uses. Named so the test can
#: drive the SAME builder over a FIXED window instead of the rolling one -- a
#: fixture pinned to "last 30 days" would pass today and silently return zero
#: rows after 2026-08-26 (measured: SELLs with a P&L are 2026-05 x8, 2026-06 x20,
#: 2026-07 x4, and the rolling window already returns only 3).
LEDGER_FETCH_WINDOW_DAYS = 30

LEDGER_TABLE = "sunny-might-477607-p8.financial_reports.paper_trades"

#: The production query, as a FULLY PLAIN string literal. Nothing here is
#: interpolated, and that is load-bearing rather than stylistic:
#: `schema_oracle.extract_sql_literals` reassembles an f-string from its CONSTANT
#: parts only, so anything interpolated is invisible to the unknown-column sweep.
#: Measured while building this step -- interpolating the table name dropped
#: `tables_resolved` from 1 to 0, and interpolating the WHERE predicate erased the
#: `SAFE.TIMESTAMP(created_at)` date semantics, emptying `scope`. Writing the
#: query as an f-string would therefore have made this file INVISIBLE to the very
#: instrument that found its defect. That is the same f-string recall hole step
#: 82.55 exists to close, and this step must not widen it while fixing the bug
#: the sweep caught. `test_the_production_sql_stays_visible_to_the_sweep` pins it.
LEDGER_FETCH_SQL = """
                SELECT trade_id, ticker, action, price, quantity, created_at,
                       SAFE_CAST(realized_pnl_pct AS FLOAT64) AS pnl
                FROM `sunny-might-477607-p8.financial_reports.paper_trades`
                WHERE SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                  AND realized_pnl_pct IS NOT NULL
                LIMIT 1000
            """

#: The substring the fixed-window variant swaps out. Kept separate so the swap
#: has an assertable target rather than being a silent no-op if the SQL drifts.
_ROLLING_PREDICATE = (
    "SAFE.TIMESTAMP(created_at) >= "
    "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)"
)


def build_ledger_fetch_sql(
    *,
    window_days: int = LEDGER_FETCH_WINDOW_DAYS,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """phase-82.39: the nightly_outcome_rebuild fetch SQL, as a real seam.

    THE DEFECT THIS REPLACES. The query used to SELECT `timestamp` and
    `realized_pnl` from `financial_reports.paper_trades` and to filter on
    `TIMESTAMP_TRUNC(timestamp, DAY)` and `realized_pnl IS NOT NULL`. NEITHER
    COLUMN EXISTS -- measured against the live schema 2026-08-06: the table has
    18 columns, `timestamp` present=False, `realized_pnl` present=False, and the
    real columns are `created_at` (STRING, REQUIRED) and `realized_pnl_pct`
    (FLOAT, NULLABLE). BigQuery answered 400 `Unrecognized name: timestamp`, the
    fail-open `except` below returned [], and the job produced an outcome
    rebuild over an empty set that looked like a successful no-op. Dead since
    this file's first commit (2301b977, 2026-05-11).

    WHY THIS IS A FUNCTION AND NOT AN INLINE STRING. Inline, the only way a test
    could reach the SQL was to copy it -- and a copied string proves nothing
    about what production issues. Extracted, criterion 1's dry run validates the
    exact text `_fetch` sends.

    WHY `SAFE.TIMESTAMP` AND NOT `TIMESTAMP_TRUNC`. `created_at` is a STRING
    column holding ISO timestamps, so it must be parsed before it can be
    compared as a timestamp; `backend/services/cycle_health.py` already lists
    `('paper_trades', 'created_at')` for exactly this reason. NOTE the idiom is
    per-column and NOT portable: `SAFE.TIMESTAMP` applied to a column that is
    already a native TIMESTAMP returns 400 "SAFE with function timestamp is not
    supported" (31 such columns exist in this project's oracle).

    `start` / `end` (ISO dates) override the rolling window. Production never
    passes them; the guards do, so criterion 2 can assert a row count that does
    not rot with the calendar.
    """
    if not (start and end):
        return LEDGER_FETCH_SQL
    replacement = (
        f"SAFE.TIMESTAMP(created_at) >= TIMESTAMP('{start}')\n"
        f"                  AND SAFE.TIMESTAMP(created_at) < TIMESTAMP('{end}')"
    )
    # Assert the target exists before replacing: a `str.replace` that matches
    # nothing is indistinguishable from success, and would silently hand back the
    # rolling window while the caller believed it had a fixed one.
    if _ROLLING_PREDICATE not in LEDGER_FETCH_SQL:
        raise AssertionError(
            "the rolling predicate is no longer present verbatim in "
            "LEDGER_FETCH_SQL; the windowed variant would silently be a no-op"
        )
    return LEDGER_FETCH_SQL.replace(_ROLLING_PREDICATE, replacement, 1)


def make_ledger_fetch_fn() -> Callable[[], list[dict]]:
    """Return a sync closure that fetches recent paper trades from BQ.

    Reads `financial_reports.paper_trades` (per CLAUDE.md BigQuery dataset
    map) for the last 30 days. Outcome computation lives in the job module
    (`_compute_outcomes`); we only fetch.

    phase-82.39: the fetch stays FAIL-OPEN -- a nightly job must never crash the
    scheduler -- but it is no longer fail-SILENT. The failure branch dispatches a
    P1 through the canonical operator channel. P1 and not P2: with
    `slack_webhook_url` empty on this machine a P2 is logged and dropped
    (`alerting.py`), which is the same invisibility this step exists to remove.
    """

    def _fetch() -> list[dict]:
        try:
            client = _bq_client()
            sql = build_ledger_fetch_sql()
            rows = list(client.query(sql, location="us-central1").result(timeout=30))
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("nightly_outcome_rebuild: BQ fetch fail-open: %r", exc)
            _alert_fetch_failure(exc)
            return []

    return _fetch


def _alert_fetch_failure(exc: BaseException) -> None:
    """phase-82.39: make the swallowed fetch failure operator-visible.

    Function-local import of the emitter, matching the 82.10/82.11 convention --
    which also means a test MUST patch
    `backend.services.observability.alerting.raise_cron_alert_sync` and not a
    module-scope name here, because there is no module-scope name to patch.

    Fail-open squared: a notification problem must not turn a tolerated fetch
    failure into an exception escaping into the scheduler.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync

        raise_cron_alert_sync(
            source="nightly_outcome_rebuild",
            error_type="ledger_fetch_failed",
            severity="P1",
            title="nightly_outcome_rebuild: BigQuery ledger fetch failed",
            details={
                "table": LEDGER_TABLE,
                "error": f"{type(exc).__name__}: {exc}"[:600],
                "consequence": "outcome rebuild ran over ZERO trades this night",
            },
        )
    except Exception as alert_exc:  # noqa: BLE001 -- never break the job
        logger.warning(
            "nightly_outcome_rebuild: alert dispatch fail-open: %r", alert_exc
        )


def make_outcome_write_fn() -> Callable[[list[dict]], int]:
    """Return a sync closure that writes outcomes to BQ.

    Schema: `financial_reports.outcome_tracking(trade_id STRING, ticker STRING,
    pnl FLOAT, outcome STRING, recorded_at TIMESTAMP)`.
    """

    def _write(outcomes: list[dict]) -> int:
        if not outcomes:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        records = [{**o, "recorded_at": now_iso} for o in outcomes]
        try:
            client = _bq_client()
            table_id = f"{client.project}.financial_reports.outcome_tracking"
            errors = client.insert_rows_json(table_id, records)
            if errors:
                logger.warning("nightly_outcome_rebuild: BQ insert errors: %r", errors)
                return 0
            return len(records)
        except Exception as exc:
            logger.warning("nightly_outcome_rebuild: BQ write fail-open: %r", exc)
            return 0

    return _write


# ── alert_fn factories (sync→async Slack post bridge) ──────────


def _post_slack_sync(
    app: "AsyncApp",
    loop: asyncio.AbstractEventLoop,
    channel: str,
    text: str,
    blocks: list[dict] | None = None,
) -> None:
    """Bridge a sync caller to AsyncApp.client.chat_postMessage.

    Called from APScheduler's executor thread; the slack-bot's main asyncio
    loop is on a different thread. `run_coroutine_threadsafe` is the only
    safe way to dispatch the async call (Slack Bolt async docs; Cosmic
    Python ch. 13).
    """
    try:
        coro = app.client.chat_postMessage(channel=channel, text=text, blocks=blocks)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.result(timeout=10)
    except Exception as exc:
        # phase-25.M: promote to ERROR + exc_info=True so the failure
        # surfaces in the alert pipeline rather than being swallowed at
        # the default WARNING level (audit bucket 24.5 F-5(d)).
        logger.error("alert_fn: Slack post failed: %r", exc, exc_info=True)


def make_alert_fn_for_budget(
    app: "AsyncApp",
    loop: asyncio.AbstractEventLoop,
    channel: str,
) -> Callable[[str, dict], None]:
    """Return a sync alert_fn for cost_budget_watcher.

    Signature matches `alert_fn(reason: str, state: dict) -> None`.

    phase-25.M: factory now fails LOUD at wiring time. Previously a
    misconfig (channel="" or app/loop None) produced a closure that
    silently posted to Slack's "" channel, triggering an API 400 that
    only surfaced at WARNING -- invisible in default log views. Audit
    bucket 24.5 F-5(d).
    """
    if app is None:
        raise ValueError(
            "make_alert_fn_for_budget: app is required (got None); cost-budget alerts would silently drop"
        )
    if loop is None:
        raise ValueError(
            "make_alert_fn_for_budget: loop is required (got None); cost-budget alerts would silently drop"
        )
    if not channel:
        raise ValueError(
            f"make_alert_fn_for_budget: channel must be non-empty (got {channel!r}); cost-budget alerts would post to empty channel"
        )

    def _alert(reason: str, state: dict) -> None:
        text = f":rotating_light: *Cost-budget breach* — {reason}"
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"```{json.dumps(state, default=str, indent=2)[:2500]}```"}},
        ]
        _post_slack_sync(app, loop, channel, text, blocks)

    return _alert


def make_alert_fn_for_integrity(
    app: "AsyncApp",
    loop: asyncio.AbstractEventLoop,
    channel: str,
) -> Callable[[list[dict]], None]:
    """Return a sync alert_fn for weekly_data_integrity.

    Signature matches `alert_fn(drifts: list[dict]) -> None`.
    """

    def _alert(drifts: list[dict]) -> None:
        if not drifts:
            return
        n = len(drifts)
        head = f":warning: *Data-integrity drift* — {n} table(s) outside threshold"
        body = "\n".join(
            f"- `{d.get('table')}`: {d.get('prev')} → {d.get('cur')} ({d.get('pct', 0):.1%})"
            for d in drifts[:10]
        )
        text = f"{head}\n{body}"
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        ]
        _post_slack_sync(app, loop, channel, text, blocks)

    return _alert


__all__ = [
    "make_price_fetch_fn",
    "make_price_write_fn",
    "make_fred_fetch_fn",
    "make_fred_write_fn",
    "make_ledger_fetch_fn",
    "make_outcome_write_fn",
    "make_alert_fn_for_budget",
    "make_alert_fn_for_integrity",
]
