"""
Paper Trading API — endpoints for autonomous virtual fund management.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.services.autonomous_loop import run_daily_cycle, get_loop_status
from backend.services.paper_trader import PaperTrader
from backend.services.perf_metrics import compute_sharpe_from_snapshots, compute_alpha

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])

# APScheduler instance (set by lifespan)
_scheduler = None
_scheduler_job_id = "paper_trading_daily"


# ── Models ───────────────────────────────────────────────────────

class StartRequest(BaseModel):
    starting_capital: Optional[float] = None


class StartResponse(BaseModel):
    status: str
    portfolio_id: str
    starting_capital: float
    scheduler_active: bool


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/start", response_model=StartResponse)
async def start_paper_trading(req: StartRequest = StartRequest()):
    """Initialize paper portfolio and start the autonomous scheduler."""
    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    # Initialize portfolio (idempotent)
    portfolio = trader.get_or_create_portfolio()

    # Start scheduler if not already running
    scheduler_active = False
    if _scheduler and not _scheduler.get_job(_scheduler_job_id):
        _add_scheduler_job(settings)
        scheduler_active = True
    elif _scheduler and _scheduler.get_job(_scheduler_job_id):
        scheduler_active = True

    return StartResponse(
        status="active",
        portfolio_id=portfolio["portfolio_id"],
        starting_capital=portfolio["starting_capital"],
        scheduler_active=scheduler_active,
    )


@router.post("/stop")
async def stop_paper_trading():
    """Pause the autonomous trading scheduler."""
    get_api_cache().invalidate("paper:*")
    if _scheduler:
        job = _scheduler.get_job(_scheduler_job_id)
        if job:
            _scheduler.remove_job(_scheduler_job_id)
            return {"status": "stopped", "message": "Scheduler paused"}
    return {"status": "not_running"}


@router.get("/status")
async def get_status():
    """Current paper trading status: NAV, P&L, scheduler info."""
    cache = get_api_cache()
    cache_key = "paper:status"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = bq.get_paper_portfolio("default")
    if not portfolio:
        return {"status": "not_initialized", "message": "Call POST /start first"}

    positions = trader.get_positions()
    loop_status = get_loop_status()

    scheduler_active = False
    if _scheduler and _scheduler.get_job(_scheduler_job_id):
        scheduler_active = True

    result = {
        "status": "active" if scheduler_active else "paused",
        "portfolio": {
            "nav": portfolio.get("total_nav"),
            "cash": portfolio.get("current_cash"),
            "starting_capital": portfolio.get("starting_capital"),
            "pnl_pct": portfolio.get("total_pnl_pct"),
            "benchmark_return_pct": portfolio.get("benchmark_return_pct"),
            "inception_date": portfolio.get("inception_date"),
            "updated_at": portfolio.get("updated_at"),
        },
        "position_count": len(positions),
        "scheduler_active": scheduler_active,
        "loop": loop_status,
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:status"])
    return result


@router.get("/portfolio")
async def get_portfolio():
    """All open positions with current prices."""
    cache = get_api_cache()
    cache_key = "paper:portfolio"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = bq.get_paper_portfolio("default")
    if not portfolio:
        raise HTTPException(404, "Paper portfolio not initialized")

    positions = trader.get_positions()
    result = {
        "portfolio": portfolio,
        "positions": positions,
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:portfolio"])
    return result


@router.get("/trades")
async def get_trades(limit: int = Query(100, ge=1, le=1000)):
    """Trade history."""
    cache = get_api_cache()
    cache_key = f"paper:trades:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    trades = bq.get_paper_trades(limit=limit)
    result = {"trades": trades, "count": len(trades)}
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:trades"])
    return result


@router.get("/snapshots")
async def get_snapshots(limit: int = Query(365, ge=1, le=3650)):
    """Daily NAV history for charting."""
    cache = get_api_cache()
    cache_key = f"paper:snapshots:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    snapshots = bq.get_paper_snapshots(limit=limit)
    result = {"snapshots": snapshots, "count": len(snapshots)}
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:snapshots"])
    return result


@router.get("/performance")
async def get_performance():
    """Performance metrics: win rate, avg return, alpha, Sharpe."""
    cache = get_api_cache()
    cache_key = "paper:performance"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)

    portfolio = bq.get_paper_portfolio("default")
    if not portfolio:
        raise HTTPException(404, "Paper portfolio not initialized")

    snapshots = bq.get_paper_snapshots(limit=365)
    trades = bq.get_paper_trades(limit=1000)

    # Compute metrics from trades
    sell_trades = [t for t in trades if t.get("action") == "SELL"]
    buy_trades = [t for t in trades if t.get("action") == "BUY"]

    # Rough P&L per closed trade (compare sell value to buy cost)
    total_trades = len(sell_trades)
    total_cost = sum(t.get("analysis_cost_today", 0) for t in snapshots)

    # Sharpe from NAV snapshots — delegates to canonical risk-free-adjusted formula
    sharpe = compute_sharpe_from_snapshots(snapshots)

    pnl_pct = portfolio.get("total_pnl_pct", 0) or 0
    bench_pct = portfolio.get("benchmark_return_pct", 0) or 0

    result = {
        "nav": portfolio.get("total_nav"),
        "starting_capital": portfolio.get("starting_capital"),
        "pnl_pct": pnl_pct,
        "benchmark_return_pct": bench_pct,
        "alpha_pct": compute_alpha(pnl_pct, bench_pct),
        "sharpe_ratio": sharpe,
        "total_sell_trades": total_trades,
        "total_buy_trades": len(buy_trades),
        "total_analysis_cost": round(total_cost, 2),
        "days_active": len(snapshots),
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:performance"])
    return result


@router.post("/run-now")
async def run_now():
    """Manually trigger one daily cycle (for testing)."""
    status = get_loop_status()
    if status["running"]:
        raise HTTPException(409, "A trading cycle is already in progress")

    get_api_cache().invalidate("paper:*")
    settings = get_settings()
    # Run in background to avoid HTTP timeout
    asyncio.create_task(_run_cycle_background(settings))
    return {"status": "started", "message": "Daily cycle triggered"}


async def _run_cycle_background(settings):
    try:
        result = await run_daily_cycle(settings)
        logger.info(f"Manual paper trading cycle result: {result.get('status')}")
    except Exception as e:
        logger.error(f"Manual cycle failed: {e}", exc_info=True)


# ── Scheduler Integration ────────────────────────────────────────

def init_scheduler(scheduler):
    """Called from app lifespan to wire up APScheduler."""
    global _scheduler
    _scheduler = scheduler
    settings = get_settings()
    if settings.paper_trading_enabled:
        _add_scheduler_job(settings)
        logger.info(f"Paper trading scheduler active: daily at {settings.paper_trading_hour}:00 ET")


def _add_scheduler_job(settings):
    if not _scheduler:
        return
    _scheduler.add_job(
        _scheduled_run,
        "cron",
        hour=settings.paper_trading_hour,
        minute=0,
        day_of_week="mon-fri",
        id=_scheduler_job_id,
        replace_existing=True,
    )


async def _scheduled_run():
    """Wrapper for the scheduler (APScheduler calls this)."""
    settings = get_settings()
    try:
        result = await run_daily_cycle(settings)
        logger.info(f"Scheduled paper trading result: {result.get('status')}")
    except Exception as e:
        logger.error(f"Scheduled paper trading failed: {e}", exc_info=True)
