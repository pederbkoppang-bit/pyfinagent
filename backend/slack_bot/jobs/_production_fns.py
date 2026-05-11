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
from pathlib import Path
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


def make_ledger_fetch_fn() -> Callable[[], list[dict]]:
    """Return a sync closure that fetches recent paper trades from BQ.

    Reads `financial_reports.paper_trades` (per CLAUDE.md BigQuery dataset
    map) for the last 30 days. Outcome computation lives in the job module
    (`_compute_outcomes`); we only fetch.
    """

    def _fetch() -> list[dict]:
        try:
            client = _bq_client()
            sql = """
                SELECT trade_id, ticker, action, price, quantity, timestamp,
                       SAFE_CAST(realized_pnl AS FLOAT64) AS pnl
                FROM `sunny-might-477607-p8.financial_reports.paper_trades`
                WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                  AND realized_pnl IS NOT NULL
                LIMIT 1000
            """
            rows = list(client.query(sql, location="us-central1").result(timeout=30))
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("nightly_outcome_rebuild: BQ fetch fail-open: %r", exc)
            return []

    return _fetch


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
        logger.warning("alert_fn: Slack post fail-open: %r", exc)


def make_alert_fn_for_budget(
    app: "AsyncApp",
    loop: asyncio.AbstractEventLoop,
    channel: str,
) -> Callable[[str, dict], None]:
    """Return a sync alert_fn for cost_budget_watcher.

    Signature matches `alert_fn(reason: str, state: dict) -> None`.
    """

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
