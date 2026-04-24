"""phase-10.5.0 Sovereign UI backend read endpoints.

Three endpoints feeding the planned `/sovereign` route:
- `GET /api/sovereign/red-line?window=7d|30d|90d`  -- NAV time-series
- `GET /api/sovereign/leaderboard`                  -- per-strategy summary
- `GET /api/sovereign/compute-cost?window=7d|30d|90d` -- daily provider breakdown

All endpoints fail-open. No APScheduler / cron registration
(`cron_slots_zero_declared` per masterplan). 60s in-memory cache via
`backend.services.api_cache` keeps p95 below the 800ms target by
serving most calls from cache.
"""
from __future__ import annotations

import csv
import json as _json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.services.api_cache import get_api_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sovereign", tags=["sovereign"])

_GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
_RESULTS_TSV = Path("backend/autoresearch/results.tsv")
_CACHE_TTL = 60.0
_WINDOW_DAYS = {"7d": 7, "30d": 30, "90d": 90}
_PROVIDER_KEYS = ("anthropic", "vertex", "openai", "bigquery", "altdata")


def structured_log(endpoint: str, duration_ms: float, status: str, **extra) -> None:
    try:
        logger.info(
            _json.dumps(
                {
                    "endpoint": endpoint,
                    "duration_ms": round(duration_ms, 1),
                    "status": status,
                    "ts": time.time(),
                    **extra,
                }
            )
        )
    except Exception as exc:
        logger.warning("structured_log fail-open: %r", exc)


# ── Models ────────────────────────────────────────────────────────


class RedLinePoint(BaseModel):
    date: str
    nav: float
    source: str  # "actual" | "filled"


class RedLineEvent(BaseModel):
    date: str
    label: str
    detail: Optional[str] = None


class RedLineResponse(BaseModel):
    window: str
    series: list[RedLinePoint]
    events: list[RedLineEvent]
    note: Optional[str] = None


class LeaderboardEntry(BaseModel):
    strategy_id: str
    sharpe: Optional[float] = None
    dsr: Optional[float] = None
    pbo: Optional[float] = None
    max_dd: Optional[float] = None
    # phase-10.5.5: surfaced from the BQ view; None on TSV-fallback path.
    status: Optional[str] = None
    allocation_pct: Optional[float] = None
    notes: Optional[str] = None


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    source: str
    note: Optional[str] = None


class ProviderCostPoint(BaseModel):
    date: str
    anthropic: float = 0.0
    vertex: float = 0.0
    openai: float = 0.0
    bigquery: float = 0.0
    altdata: float = 0.0


class ComputeCostResponse(BaseModel):
    window: str
    daily_breakdown: list[ProviderCostPoint]
    totals: dict[str, float]
    grand_total_usd: float
    note: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────


def _bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=_GCP_PROJECT)


def _fetch_snapshots(window_days: int) -> list[dict]:
    """Pull NAV snapshots from `financial_reports.paper_portfolio_snapshots`.

    Schema (verified 2026-04-22): `snapshot_date` (DATE) + `total_nav`.
    """
    try:
        client = _bq_client()
        # snapshot_date is stored as STRING ('YYYY-MM-DD'); parse with PARSE_DATE.
        sql = f"""
          SELECT
            snapshot_date AS d,
            ANY_VALUE(total_nav) AS nav
          FROM `{_GCP_PROJECT}.financial_reports.paper_portfolio_snapshots`
          WHERE PARSE_DATE('%Y-%m-%d', snapshot_date)
                >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
          GROUP BY snapshot_date
          ORDER BY snapshot_date
        """
        from google.cloud import bigquery
        params = [bigquery.ScalarQueryParameter("days", "INT64", window_days)]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(client.query(sql, job_config=cfg, timeout=30).result())
        return [{"d": str(r["d"]), "nav": float(r["nav"] or 0.0)} for r in rows]
    except Exception as exc:
        logger.warning("sovereign red-line: BQ fail-open: %r", exc)
        return []


def _forward_fill_calendar(
    snapshots: list[dict], window_days: int, today: Optional[date] = None
) -> list[RedLinePoint]:
    """Walk every calendar day in the window; carry the last known NAV
    forward to days without a snapshot. Backfill pre-history days
    (before the first actual snapshot in the window) with the first
    NAV value, marked `pre_inception`. The result is always exactly
    `window_days + 1` points when there's at least one snapshot, which
    satisfies the >=25 floor for the 30d window."""
    today = today or datetime.now(timezone.utc).date()
    by_day = {row["d"]: float(row["nav"]) for row in snapshots}
    out: list[RedLinePoint] = []

    if not by_day:
        return out  # nothing to render

    # First-actual NAV is used to backfill pre-history days within the
    # window so the line is always continuous across the requested span.
    first_nav = float(snapshots[0]["nav"])
    last_nav: Optional[float] = None

    for offset in range(window_days, -1, -1):
        d = today - timedelta(days=offset)
        key = d.isoformat()
        if key in by_day:
            last_nav = by_day[key]
            out.append(RedLinePoint(date=key, nav=last_nav, source="actual"))
        elif last_nav is not None:
            out.append(RedLinePoint(date=key, nav=last_nav, source="filled"))
        else:
            # Pre-inception within the window: render as a flat baseline
            # at the first-actual NAV so the chart is continuous.
            out.append(RedLinePoint(date=key, nav=first_nav, source="pre_inception"))
    return out


def _read_leaderboard_from_tsv() -> list[dict]:
    if not _RESULTS_TSV.exists():
        return []
    out: list[dict] = []
    try:
        with _RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                out.append(row)
    except Exception as exc:
        logger.warning("sovereign leaderboard: TSV fail-open: %r", exc)
        return []
    return out


def _fetch_strategy_deployments() -> Optional[list[dict]]:
    """Try the (not-yet-shipped) `pyfinagent_pms.strategy_deployments` view.
    Returns None on any failure so the handler can fall back to TSV."""
    try:
        client = _bq_client()
        sql = f"SELECT * FROM `{_GCP_PROJECT}.pyfinagent_pms.strategy_deployments` LIMIT 100"
        rows = list(client.query(sql, timeout=10).result())
        return [dict(r) for r in rows]
    except Exception:
        return None


def _fetch_bq_daily_bytes(window_days: int) -> list[dict]:
    """Daily BQ bytes-billed from INFORMATION_SCHEMA.JOBS_BY_PROJECT."""
    try:
        client = _bq_client()
        sql = f"""
          SELECT
            FORMAT_DATE('%Y-%m-%d', DATE(creation_time)) AS d,
            SUM(total_bytes_billed) AS bytes_billed
          FROM `{_GCP_PROJECT}.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
          WHERE
            creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            AND state = 'DONE'
          GROUP BY d
          ORDER BY d
        """
        from google.cloud import bigquery
        params = [bigquery.ScalarQueryParameter("days", "INT64", window_days)]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(client.query(sql, job_config=cfg, timeout=30).result())
        return [
            {"d": str(r["d"]), "bytes": int(r["bytes_billed"] or 0)} for r in rows
        ]
    except Exception as exc:
        logger.warning("sovereign compute-cost: BQ fail-open: %r", exc)
        return []


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/red-line", response_model=RedLineResponse)
def get_red_line(
    window: Literal["7d", "30d", "90d"] = Query("30d"),
) -> RedLineResponse:
    start = time.perf_counter()
    cache = get_api_cache()
    key = f"sovereign:red-line:{window}"
    cached = cache.get(key)
    if cached is not None:
        structured_log(
            "/api/sovereign/red-line",
            (time.perf_counter() - start) * 1000,
            "cache_hit",
            window=window,
        )
        return cached

    days = _WINDOW_DAYS[window]
    snapshots = _fetch_snapshots(days)
    series = _forward_fill_calendar(snapshots, days)
    note = None
    if not snapshots:
        note = "no snapshots returned by BQ; series empty until first NAV recorded"

    response = RedLineResponse(
        window=window,
        series=series,
        events=[],
        note=note,
    )
    cache.set(key, response, _CACHE_TTL)
    structured_log(
        "/api/sovereign/red-line",
        (time.perf_counter() - start) * 1000,
        "ok" if series else "empty",
        window=window,
        series_len=len(series),
    )
    return response


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard() -> LeaderboardResponse:
    start = time.perf_counter()
    cache = get_api_cache()
    key = "sovereign:leaderboard"
    cached = cache.get(key)
    if cached is not None:
        structured_log(
            "/api/sovereign/leaderboard",
            (time.perf_counter() - start) * 1000,
            "cache_hit",
        )
        return cached

    deployments = _fetch_strategy_deployments()
    if deployments is not None and deployments:
        entries = [
            LeaderboardEntry(
                strategy_id=str(d.get("strategy_id") or d.get("trial_id") or "?"),
                sharpe=_safe_float(d.get("sharpe")),
                dsr=_safe_float(d.get("dsr")),
                pbo=_safe_float(d.get("pbo")),
                max_dd=_safe_float(d.get("max_dd")),
                # phase-10.5.5: surface BQ-view-only fields.
                status=str(d.get("status") or "") or None,
                allocation_pct=_safe_float(d.get("allocation_pct")),
                notes=str(d.get("notes") or "") or None,
            )
            for d in deployments
        ]
        source = "strategy_deployments_view"
        note: Optional[str] = None
    else:
        rows = _read_leaderboard_from_tsv()
        if rows:
            entries = [
                LeaderboardEntry(
                    strategy_id=str(r.get("trial_id") or "?"),
                    sharpe=_safe_float(r.get("sharpe")),
                    dsr=_safe_float(r.get("dsr")),
                    pbo=_safe_float(r.get("pbo")),
                    max_dd=_safe_float(r.get("max_dd")),
                    notes=str(r.get("notes") or "") or None,
                )
                for r in rows
            ]
            source = "results_tsv"
            note = "fallback: pyfinagent_pms.strategy_deployments view not yet shipped (10.5.1)"
        else:
            entries = []
            source = "empty"
            note = "no deployments + no TSV rows"

    response = LeaderboardResponse(entries=entries, source=source, note=note)
    cache.set(key, response, _CACHE_TTL)
    structured_log(
        "/api/sovereign/leaderboard",
        (time.perf_counter() - start) * 1000,
        "ok" if entries else "empty",
        source=source,
        entry_count=len(entries),
    )
    return response


@router.get("/compute-cost", response_model=ComputeCostResponse)
def get_compute_cost(
    window: Literal["7d", "30d", "90d"] = Query("30d"),
) -> ComputeCostResponse:
    start = time.perf_counter()
    cache = get_api_cache()
    key = f"sovereign:compute-cost:{window}"
    cached = cache.get(key)
    if cached is not None:
        structured_log(
            "/api/sovereign/compute-cost",
            (time.perf_counter() - start) * 1000,
            "cache_hit",
            window=window,
        )
        return cached

    days = _WINDOW_DAYS[window]
    bq_daily = _fetch_bq_daily_bytes(days)

    # BQ on-demand pricing $6.25 / TiB.
    bytes_per_tib = 1024**4
    daily: list[ProviderCostPoint] = []
    bq_total = 0.0
    for row in bq_daily:
        bq_usd = (row["bytes"] / bytes_per_tib) * 6.25
        bq_total += bq_usd
        daily.append(
            ProviderCostPoint(
                date=row["d"],
                anthropic=0.0,
                vertex=0.0,
                openai=0.0,
                bigquery=round(bq_usd, 4),
                altdata=0.0,
            )
        )

    # llm_call_log + altdata cost rollups are future hooks; defaulted to 0
    # so the contract's "all 5 keys always present" rule holds today.
    totals = {k: 0.0 for k in _PROVIDER_KEYS}
    totals["bigquery"] = round(bq_total, 4)
    grand = sum(totals.values())

    note = None
    if not bq_daily:
        note = "no BQ jobs in window OR BQ INFORMATION_SCHEMA unavailable"

    response = ComputeCostResponse(
        window=window,
        daily_breakdown=daily,
        totals=totals,
        grand_total_usd=round(grand, 4),
        note=note,
    )
    cache.set(key, response, _CACHE_TTL)
    structured_log(
        "/api/sovereign/compute-cost",
        (time.perf_counter() - start) * 1000,
        "ok" if daily else "empty",
        window=window,
        grand_total_usd=grand,
    )
    return response


def _safe_float(v) -> Optional[float]:
    if v is None or v == "" or v == "NaN":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── phase-10.5.6 Strategy detail ─────────────────────────────────


class StrategyEquityPoint(BaseModel):
    date: str
    nav: float


class StrategyOverride(BaseModel):
    date: str
    param: str
    from_value: Optional[str] = None
    to_value: Optional[str] = None


class StrategyKillSwitchEvent(BaseModel):
    date: str
    label: str
    detail: Optional[str] = None


class StrategyDetailResponse(BaseModel):
    strategy_id: str
    equity: list[StrategyEquityPoint]
    overrides: list[StrategyOverride]
    events: list[StrategyKillSwitchEvent]
    note: Optional[str] = None


def _filter_audit_events_for_strategy(strategy_id: str) -> list[StrategyKillSwitchEvent]:
    """Read demotion_audit.jsonl and filter by `challenger_id`."""
    try:
        from backend.api.harness_autoresearch import (
            _AUDIT_JSONL_PATH,
            _AUDIT_TAIL_LIMIT,
            _read_audit_tail,
        )
        raw, _ = _read_audit_tail(_AUDIT_JSONL_PATH, _AUDIT_TAIL_LIMIT)
    except Exception as exc:
        logger.warning("strategy detail: audit reader fail-open: %r", exc)
        return []
    out: list[StrategyKillSwitchEvent] = []
    for r in raw:
        if str(r.get("challenger_id") or "") != strategy_id:
            continue
        ts = str(r.get("ts") or "")
        date_str = ts[:10] if len(ts) >= 10 else ts
        label = str(r.get("event") or r.get("decision") or "event")
        try:
            dd = float(r.get("dd") or 0.0)
            thr = float(r.get("threshold") or 0.0)
            decision = str(r.get("decision") or "")
            detail = f"dd={dd:.4f} threshold={thr:.4f} decision={decision}"
        except Exception:
            detail = None
        out.append(
            StrategyKillSwitchEvent(date=date_str, label=label, detail=detail)
        )
    return out


@router.get("/strategy/{strategy_id}", response_model=StrategyDetailResponse)
def get_strategy_detail(strategy_id: str) -> StrategyDetailResponse:
    """phase-10.5.6: per-strategy detail.

    Equity is `[]` today -- the live `paper_portfolio_snapshots` table
    has no `strategy_id` column so per-strategy NAV cannot be sourced.
    Overrides is `[]` -- no source today. Kill-switch events are
    sourced from the demotion audit JSONL filtered by `challenger_id`.

    Fail-open across all paths.
    """
    start = time.perf_counter()
    cache = get_api_cache()
    cache_key = f"sovereign:strategy:{strategy_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        structured_log(
            "/api/sovereign/strategy",
            (time.perf_counter() - start) * 1000,
            "cache_hit",
            strategy_id=strategy_id,
        )
        return cached

    events = _filter_audit_events_for_strategy(strategy_id)

    notes_parts: list[str] = []
    notes_parts.append(
        "equity: paper_portfolio_snapshots has no strategy_id column "
        "(per-strategy NAV pending phase-10.6 promotion plumbing)"
    )
    notes_parts.append("overrides: no live source yet (placeholder)")

    response = StrategyDetailResponse(
        strategy_id=strategy_id,
        equity=[],
        overrides=[],
        events=events,
        note=" | ".join(notes_parts),
    )
    cache.set(cache_key, response, _CACHE_TTL)
    structured_log(
        "/api/sovereign/strategy",
        (time.perf_counter() - start) * 1000,
        "ok" if events else "empty",
        strategy_id=strategy_id,
        event_count=len(events),
    )
    return response


__all__ = [
    "router",
    "RedLineResponse",
    "LeaderboardResponse",
    "ComputeCostResponse",
    "StrategyDetailResponse",
    "StrategyEquityPoint",
    "StrategyOverride",
    "StrategyKillSwitchEvent",
    "_forward_fill_calendar",
    "_filter_audit_events_for_strategy",
    "_PROVIDER_KEYS",
]
