"""
Performance API — cache stats, latency metrics, and autoresearch optimizer control.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.services.perf_tracker import get_perf_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/perf", tags=["performance"])

# ── Optimizer state (module-level, same pattern as backtest.py) ───

_optimizer_state = {
    "status": "idle",  # idle | running | stopped
    "iterations": 0,
    "best_p95_ms": None,
    "kept": 0,
    "discarded": 0,
}
_optimizer_task: Optional[asyncio.Task] = None


# ── Metrics endpoints ────────────────────────────────────────────

@router.get("/summary")
async def get_perf_summary(window: int = Query(300, ge=10, le=3600)):
    """Latency summary: p50/p95/p99, cache hit rate, per-endpoint breakdown."""
    return get_perf_tracker().summarize(window_seconds=window)


@router.get("/slow")
async def get_slow_endpoints(threshold: float = Query(1000, ge=100)):
    """Endpoints with p95 latency above threshold in ms."""
    return get_perf_tracker().get_slow_endpoints(threshold_ms=threshold)


# phase-4.14.23: LLM-call p95 latency endpoint fed by the BQ
# llm_call_log table (see scripts/migrations/add_llm_call_log.py).
# The harness tab consumes this to render the p95 trend tile.
# Falls back to {"unavailable": true} when the table has not yet
# been materialised in this environment so the UI can hide the tile
# without error.
@router.get("/llm/p95")
async def get_llm_p95_latency(
    window_hours: int = Query(24, ge=1, le=168),
    provider: Optional[str] = Query(None, description="anthropic|gemini|openai|github-models"),
):
    """Return p50/p95/p99 latency + call count for LLM traffic in the window."""
    try:
        from backend.db.bigquery_client import BigQueryClient
        from backend.config.settings import get_settings
        s = get_settings()
        bq = BigQueryClient(s)
        dataset = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        where = ["ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @h HOUR)"]
        params: list = [
            {"name": "h", "parameterType": {"type": "INT64"}, "parameterValue": {"value": window_hours}},
        ]
        if provider:
            where.append("provider = @p")
            params.append({
                "name": "p",
                "parameterType": {"type": "STRING"},
                "parameterValue": {"value": provider},
            })
        sql = f"""
            SELECT
              COUNT(*) AS n,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)] AS p50_ms,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_ms,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS p99_ms
            FROM `{s.gcp_project_id}.{dataset}.llm_call_log`
            WHERE {' AND '.join(where)}
        """
        # BigQueryClient exposes .client for raw queries.
        rows = list(bq.client.query(sql).result())
        if not rows:
            return {"unavailable": True, "reason": "no rows"}
        r = rows[0]
        return {
            "window_hours": window_hours,
            "provider": provider,
            "n": int(r.get("n") or 0),
            "p50_ms": float(r.get("p50_ms") or 0.0),
            "p95_ms": float(r.get("p95_ms") or 0.0),
            "p99_ms": float(r.get("p99_ms") or 0.0),
        }
    except Exception as e:
        logger.debug("llm/p95 query failed: %s", e)
        return {"unavailable": True, "reason": str(e)[:200]}


@router.get("/cache")
async def get_cache_stats():
    """Cache statistics: entry count, hit rate, total gets/hits."""
    cache = get_api_cache()
    return {
        **cache.stats(),
        "ttl_config": dict(ENDPOINT_TTLS),
    }


@router.post("/cache/clear")
async def clear_cache():
    """Flush all cache entries."""
    count = get_api_cache().clear()
    return {"cleared": count}


# ── Optimizer endpoints ──────────────────────────────────────────

@router.post("/optimize")
async def start_optimizer():
    """Start the autoresearch performance optimizer loop."""
    global _optimizer_task

    if _optimizer_task and not _optimizer_task.done():
        raise HTTPException(409, "Optimizer is already running")

    from backend.services.perf_optimizer import PerfOptimizer
    optimizer = PerfOptimizer()
    _optimizer_task = asyncio.create_task(_run_optimizer(optimizer))
    _optimizer_state["status"] = "running"
    return {"status": "started"}


@router.post("/optimize/stop")
async def stop_optimizer():
    """Stop the optimizer loop gracefully."""
    if _optimizer_task and not _optimizer_task.done():
        from backend.services.perf_optimizer import PerfOptimizer
        # Signal stop via module-level flag
        PerfOptimizer._running = False
        _optimizer_state["status"] = "stopped"
        return {"status": "stopping"}
    _optimizer_state["status"] = "idle"
    return {"status": "not_running"}


@router.get("/optimize/status")
async def get_optimizer_status():
    """Current optimizer state."""
    return _optimizer_state


@router.get("/optimize/experiments")
def get_optimizer_experiments():
    """Full experiment history from perf_results.tsv."""
    import os
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "services", "experiments", "perf_results.tsv"
    )
    if not os.path.exists(tsv_path):
        return {"experiments": []}

    experiments = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) >= len(header):
                experiments.append(dict(zip(header, values)))

    return {"experiments": experiments}


# ── Internal ─────────────────────────────────────────────────────

async def _run_optimizer(optimizer):
    try:
        await optimizer.run_loop(_optimizer_state)
    except Exception as e:
        logger.error("Perf optimizer error: %s", e, exc_info=True)
        _optimizer_state["status"] = "error"
        _optimizer_state["error"] = str(e)
