"""
Backtest API — walk-forward backtesting + quant optimizer endpoints.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# Module-level state for async backtest runs
_backtest_state = {
    "status": "idle",  # idle | running | completed | error
    "run_id": None,
    "progress": "",
    "result": None,
    "error": None,
}

# Module-level state for optimizer
_optimizer_state = {
    "status": "idle",  # idle | running | stopped
    "iterations": 0,
    "best_sharpe": None,
    "best_dsr": None,
    "kept": 0,
    "discarded": 0,
}
_optimizer_task: Optional[asyncio.Task] = None


# ── Request / Response Models ────────────────────────────────────

class BacktestRunRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    train_window_months: Optional[int] = None
    test_window_months: Optional[int] = None
    embargo_days: Optional[int] = None
    holding_days: Optional[int] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    starting_capital: Optional[float] = None
    max_positions: Optional[int] = None
    top_n_candidates: Optional[int] = None


class IngestRequest(BaseModel):
    start_date: Optional[str] = Field(None, description="Price history start date")
    end_date: Optional[str] = Field(None, description="Price history end date")


class OptimizerStartRequest(BaseModel):
    max_iterations: int = Field(100, ge=1, le=1000)
    use_llm: bool = Field(False, description="Use LLM proposals (costs ~$0.01/proposal) vs random perturbation ($0)")


# ── Backtest Endpoints ───────────────────────────────────────────

@router.post("/run")
async def run_backtest(req: BacktestRunRequest = BacktestRunRequest()):
    """Start a walk-forward backtest (async background task)."""
    global _backtest_state

    if _backtest_state["status"] == "running":
        raise HTTPException(400, "Backtest already running")

    run_id = str(uuid.uuid4())[:8]
    _backtest_state = {
        "status": "running",
        "run_id": run_id,
        "progress": "Initializing...",
        "result": None,
        "error": None,
    }

    asyncio.create_task(_run_backtest_async(run_id, req))
    return {"status": "started", "run_id": run_id}


@router.get("/status")
async def get_backtest_status():
    """Poll backtest progress."""
    return {
        "status": _backtest_state["status"],
        "run_id": _backtest_state["run_id"],
        "progress": _backtest_state["progress"],
        "has_result": _backtest_state["result"] is not None,
    }


@router.get("/results")
async def get_backtest_results():
    """Full backtest results with analytics."""
    if _backtest_state["result"] is None:
        if _backtest_state["status"] == "running":
            raise HTTPException(202, "Backtest still running")
        raise HTTPException(404, "No backtest results available")
    return _backtest_state["result"]


@router.get("/results/{window_id}")
async def get_window_result(window_id: int):
    """Per-window detail."""
    if _backtest_state["result"] is None:
        raise HTTPException(404, "No backtest results available")
    
    windows = _backtest_state["result"].get("per_window", [])
    for w in windows:
        if w.get("window_id") == window_id:
            return w
    raise HTTPException(404, f"Window {window_id} not found")


# ── Data Ingestion Endpoints ─────────────────────────────────────

@router.post("/ingest")
async def run_data_ingestion(req: IngestRequest = IngestRequest(start_date=None, end_date=None)):
    """Ingest historical price, fundamental, and macro data into BigQuery."""
    settings = get_settings()
    bq = BigQueryClient(settings)

    from backend.backtest.data_ingestion import DataIngestionService
    from backend.tools.screener import screen_universe
    service = DataIngestionService(bq.client, settings)

    try:
        # Get S&P 500 tickers for ingestion
        screen = screen_universe()
        tickers = [s["ticker"] for s in screen] if screen else []
        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]

        result = await asyncio.to_thread(
            service.run_full_ingestion,
            tickers=tickers,
            start_date=req.start_date or settings.backtest_start_date,
            end_date=req.end_date or settings.backtest_end_date,
            fred_api_key=settings.fred_api_key,
        )
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@router.get("/ingest/status")
async def get_ingestion_status():
    """Get row counts for historical data tables."""
    settings = get_settings()
    bq = BigQueryClient(settings)

    from backend.backtest.data_ingestion import DataIngestionService
    service = DataIngestionService(bq.client, settings)

    try:
        status = service.get_ingestion_status()
        return status
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Optimizer Endpoints ──────────────────────────────────────────

@router.post("/optimize")
async def start_optimizer(req: OptimizerStartRequest = OptimizerStartRequest(max_iterations=100, use_llm=False)):
    """Start quant strategy optimization loop (background)."""
    global _optimizer_task, _optimizer_state

    if _optimizer_state["status"] == "running":
        raise HTTPException(400, "Optimizer already running")

    _optimizer_state = {
        "status": "running",
        "iterations": 0,
        "best_sharpe": None,
        "best_dsr": None,
        "kept": 0,
        "discarded": 0,
    }

    _optimizer_task = asyncio.create_task(
        _run_optimizer_async(req.max_iterations, req.use_llm)
    )
    return {"status": "started", "max_iterations": req.max_iterations}


@router.post("/optimize/stop")
async def stop_optimizer():
    """Stop the optimization loop gracefully."""
    global _optimizer_state
    if _optimizer_state["status"] != "running":
        return {"status": "not_running"}

    _optimizer_state["status"] = "stopped"
    return {"status": "stopping"}


@router.get("/optimize/status")
async def get_optimizer_status():
    """Current optimizer state."""
    return _optimizer_state


@router.get("/optimize/experiments")
async def get_optimizer_experiments():
    """Full experiment history from quant_results.tsv."""
    cache = get_api_cache()
    cache_key = "backtest:experiments"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    import os
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv"
    )
    if not os.path.exists(tsv_path):
        return {"experiments": []}

    experiments = []
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) == len(header):
                experiments.append(dict(zip(header, values)))

    result = {"experiments": experiments}
    cache.set(cache_key, result, ENDPOINT_TTLS["backtest:experiments"])
    return result


@router.get("/optimize/best")
async def get_optimizer_best():
    """Best strategy params + feature importance from latest optimization."""
    cache = get_api_cache()
    cache_key = "backtest:best"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    import os
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv"
    )
    if not os.path.exists(tsv_path):
        raise HTTPException(404, "No optimization results. Run optimizer first.")

    # Find best kept experiment
    best_sharpe = -999
    best_row = None
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) == len(header):
                row = dict(zip(header, values))
                if row.get("status") in ("keep", "BASELINE"):
                    try:
                        sharpe = float(row.get("metric_after", 0))
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_row = row
                    except (ValueError, TypeError):
                        pass

    if not best_row:
        raise HTTPException(404, "No successful experiments found")

    result = {
        "best_sharpe": best_sharpe,
        "best_dsr": float(best_row.get("dsr", 0)),
        "best_experiment": best_row,
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["backtest:best"])
    return result


# ── Async Helpers ────────────────────────────────────────────────

async def _run_backtest_async(run_id: str, req: BacktestRunRequest):
    """Run backtest in background thread."""
    global _backtest_state
    try:
        settings = get_settings()
        bq = BigQueryClient(settings)

        from backend.backtest.backtest_engine import BacktestEngine
        from backend.backtest.analytics import generate_report, compute_baseline_strategies
        from backend.backtest import cache as bt_cache

        def progress_cb(msg: str):
            _backtest_state["progress"] = msg

        engine = BacktestEngine(
            bq_client=bq,
            project=settings.gcp_project_id,
            dataset=settings.bq_dataset_reports,
            start_date=req.start_date or settings.backtest_start_date,
            end_date=req.end_date or settings.backtest_end_date,
            train_window_months=req.train_window_months or settings.backtest_train_window_months,
            test_window_months=req.test_window_months or settings.backtest_test_window_months,
            embargo_days=req.embargo_days or settings.backtest_embargo_days,
            holding_days=req.holding_days or settings.backtest_holding_days,
            tp_pct=req.tp_pct or settings.backtest_tp_pct,
            sl_pct=req.sl_pct or settings.backtest_sl_pct,
            starting_capital=req.starting_capital or settings.backtest_starting_capital,
            max_positions=req.max_positions or settings.backtest_max_positions,
            top_n_candidates=req.top_n_candidates or settings.backtest_top_n_candidates,
            progress_callback=progress_cb,
        )

        result = await asyncio.to_thread(engine.run_backtest)

        # Compute baselines over the full test range
        baselines = None
        if result.windows:
            try:
                test_start = result.windows[0].test_start
                test_end = result.windows[-1].test_end
                candidate_tickers = list({
                    p.get("ticker", "") for w in result.windows for p in w.predictions if p.get("ticker")
                })
                baselines = compute_baseline_strategies(
                    bt_cache.cached_prices, test_start, test_end, candidate_tickers,
                )
            except Exception as e:
                logger.warning(f"Baseline computation failed: {e}")

        report = generate_report(result, baselines=baselines)
        report["run_id"] = run_id
        report["config"] = {
            "start_date": req.start_date or settings.backtest_start_date,
            "end_date": req.end_date or settings.backtest_end_date,
            "strategy": getattr(engine, "strategy", "triple_barrier"),
        }

        _backtest_state["status"] = "completed"
        _backtest_state["result"] = report

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        _backtest_state["status"] = "error"
        _backtest_state["error"] = str(e)


async def _run_optimizer_async(max_iterations: int, use_llm: bool):
    """Run quant optimizer in background thread."""
    global _optimizer_state
    try:
        settings = get_settings()
        bq = BigQueryClient(settings)

        from backend.backtest.backtest_engine import BacktestEngine
        from backend.backtest.quant_optimizer import QuantStrategyOptimizer

        engine = BacktestEngine(
            bq_client=bq,
            project=settings.gcp_project_id,
            dataset=settings.bq_dataset_reports,
            start_date=settings.backtest_start_date,
            end_date=settings.backtest_end_date,
        )

        def status_cb(iterations, best_sharpe, best_dsr, kept, discarded):
            _optimizer_state.update({
                "iterations": iterations,
                "best_sharpe": best_sharpe,
                "best_dsr": best_dsr,
                "kept": kept,
                "discarded": discarded,
            })

        optimizer = QuantStrategyOptimizer(engine, status_callback=status_cb)

        # Wire MDA updates to MetaCoordinator for MDA→Agent bridge
        from backend.services.autonomous_loop import get_coordinator
        coordinator = get_coordinator()

        await asyncio.to_thread(
            optimizer.run_loop,
            max_iterations=max_iterations,
            use_llm=use_llm,
            stop_check=lambda: _optimizer_state["status"] != "running",
            on_mda_update=coordinator.update_mda_features,
        )

        _optimizer_state["status"] = "completed"

    except Exception as e:
        logger.error(f"Optimizer failed: {e}", exc_info=True)
        _optimizer_state["status"] = "error"
