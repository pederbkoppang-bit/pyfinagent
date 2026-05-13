"""phase-15.10 Observability endpoint.

Thin alias over `backend.services.perf_tracker.PerfTracker.summarize()`
exposing p50/p95/p99 latency + per-endpoint breakdown. The underlying
tracker is populated by the app-wide latency middleware in
`backend.main`; this handler is read-only.

Verification contract: the response must contain the bare keys
`p50`, `p95`, `p99` (no `_ms` suffix) per `phase-15.10` verification.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from backend.services.perf_tracker import get_perf_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/observability", tags=["observability"])


@router.get("/freshness")
async def get_observability_freshness() -> dict[str, Any]:
    """phase-16.22 alias: surface signal-freshness under the
    /api/observability/* prefix expected by the masterplan verification.
    Delegates to the canonical implementation in `paper_trading.py`'s
    /freshness route, which computes per-source last_tick_age + ingest
    lag from BigQuery via `cycle_health.compute_freshness`.
    """
    import asyncio as _asyncio
    from backend.config.settings import get_settings as _get_settings
    from backend.db.bigquery_client import BigQueryClient as _BQ
    from backend.services.cycle_health import compute_freshness as _cf

    settings = _get_settings()
    bq = _BQ(settings)
    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
    return await _asyncio.to_thread(_cf, bq, cycle_interval_sec)


@router.get("/data-freshness")
async def get_observability_data_freshness() -> dict[str, Any]:
    """phase-25.C7: unified per-table freshness across the 6 data sources
    covered by `compute_freshness` (paper signals + historical price/macro/
    fundamentals tables added in 25.A7). Identical payload to /freshness;
    the rename clarifies that the scope is the full data warehouse, not
    just paper-trading signals.
    """
    import asyncio as _asyncio
    from backend.config.settings import get_settings as _get_settings
    from backend.db.bigquery_client import BigQueryClient as _BQ
    from backend.services.cycle_health import compute_freshness as _cf

    settings = _get_settings()
    bq = _BQ(settings)
    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
    return await _asyncio.to_thread(_cf, bq, cycle_interval_sec)


@router.get("/latency")
def get_observability_latency(
    window: int = Query(300, ge=10, le=3600),
) -> dict[str, Any]:
    """Return p50/p95/p99 latency + per-endpoint breakdown.

    Strips the `_ms` suffix from the core percentile keys so the
    verification assertion `all(k in d for k in ('p50','p95','p99'))`
    passes. Keeps `total_requests`, `window_seconds`, `per_endpoint`
    as-is for tile rendering.
    """
    try:
        s = get_perf_tracker().summarize(window_seconds=window)
    except Exception as exc:
        logger.warning("observability: summarize fail-open: %r", exc)
        s = {
            "window_seconds": window,
            "total_requests": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "cache_hit_rate_pct": 0,
            "per_endpoint": {},
        }
    return {
        "p50": s.get("p50_ms", 0),
        "p95": s.get("p95_ms", 0),
        "p99": s.get("p99_ms", 0),
        "total_requests": s.get("total_requests", 0),
        "window_seconds": s.get("window_seconds", window),
        "cache_hit_rate_pct": s.get("cache_hit_rate_pct", 0),
        "per_endpoint": s.get("per_endpoint", {}),
    }


__all__ = ["router"]
