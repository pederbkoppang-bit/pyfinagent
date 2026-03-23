"""
Backtest API — walk-forward backtesting + quant optimizer endpoints.
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.backtest import result_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# Module-level state for async backtest runs
_backtest_state = {
    "status": "idle",  # idle | running | completed | error
    "run_id": None,
    "progress": {},
    "result": None,
    "error": None,
    "traceback": None,
    "engine_source": None,  # "backtest" | "optimizer" | None
}

# Auto-load latest backtest result on module init so results survive restarts
_prev = result_store.load_latest()
if _prev:
    _backtest_state["status"] = "completed"
    _backtest_state["run_id"] = _prev.get("run_id")
    _backtest_state["result"] = _prev
    logger.info("Auto-loaded previous backtest result (run_id=%s)", _prev.get("run_id"))

# Module-level state for optimizer
_optimizer_state = {
    "status": "idle",  # idle | running | stopped
    "iterations": 0,
    "best_sharpe": None,
    "best_dsr": None,
    "kept": 0,
    "discarded": 0,
    "traceback": None,
}
_optimizer_task: Optional[asyncio.Task] = None


def _is_engine_busy() -> bool:
    """True if either backtest or optimizer is running."""
    return (
        _backtest_state["status"] == "running"
        or _optimizer_state["status"] == "running"
    )


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
    if _optimizer_state["status"] == "running":
        raise HTTPException(409, "Optimizer is running - backtest unavailable")

    run_id = str(uuid.uuid4())[:8]
    _backtest_state = {
        "status": "running",
        "run_id": run_id,
        "progress": {"step": "initializing", "step_detail": "Starting backtest..."},
        "result": None,
        "error": None,
        "traceback": None,
        "engine_source": "backtest",
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
        "error": _backtest_state["error"],
        "traceback": _backtest_state["traceback"],
        "engine_source": _backtest_state.get("engine_source"),
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
    if _backtest_state["status"] == "running":
        raise HTTPException(409, "Backtest is running - optimizer unavailable")

    _optimizer_state = {
        "status": "running",
        "iterations": 0,
        "best_sharpe": None,
        "best_dsr": None,
        "kept": 0,
        "discarded": 0,
        "current_step": "starting",
        "current_detail": "",
        "run_id": "",
        "error": None,
        "traceback": None,
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
async def get_optimizer_experiments(run_id: str | None = None):
    """Full experiment history from quant_results.tsv.

    Args:
        run_id: If provided, filter to experiments from this optimizer run only.
    """
    cache = get_api_cache()
    cache_key = f"backtest:experiments:{run_id or 'all'}"
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
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) == len(header):
                row = dict(zip(header, values))
                if run_id and row.get("run_id") not in (run_id, "BASELINE"):
                    # Include BASELINEs from this run (same run_id prefix won't match,
                    # but we want the last BASELINE before this run's experiments).
                    # Actually filter: only keep rows whose run_id matches, or BASELINE
                    # rows that appear just before the first matching experiment.
                    pass
                experiments.append(row)

    # If run_id filter requested, find experiments belonging to this run
    if run_id:
        # Find the BASELINE that immediately precedes the first experiment with this run_id
        run_experiments = []
        last_baseline_idx = -1
        for idx, exp in enumerate(experiments):
            if exp.get("run_id") == "BASELINE":
                last_baseline_idx = idx
            elif exp.get("run_id") == run_id:
                if not run_experiments and last_baseline_idx >= 0:
                    run_experiments.append(experiments[last_baseline_idx])
                run_experiments.append(exp)
        # If only baseline found (run just started), show the latest baseline
        if not run_experiments and last_baseline_idx >= 0:
            run_experiments = [experiments[last_baseline_idx]]
        experiments = run_experiments

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
    with open(tsv_path, "r", encoding="utf-8") as f:
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


# ── Backtest Runs (Persistence) ──────────────────────────────────

@router.get("/runs")
async def list_backtest_runs():
    """List all persisted backtest results (newest first)."""
    return {"runs": result_store.list_runs()}


@router.get("/runs/{run_id}")
async def get_backtest_run(run_id: str):
    """Load a specific historical backtest result."""
    data = result_store.load_result(run_id)
    if data is None:
        raise HTTPException(404, f"No saved result for run_id={run_id}")
    return data


@router.delete("/runs/{run_id}")
async def delete_backtest_run(run_id: str):
    """Delete a specific backtest result from disk."""
    deleted = result_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(404, f"No saved result for run_id={run_id}")
    return {"status": "deleted", "run_id": run_id}


# ── Optimizer Insights ───────────────────────────────────────────

@router.get("/optimize/insights")
async def get_optimizer_insights():
    """Full optimizer context for the Insights tab: param bounds, experiments with full params, data scope."""
    import os
    from backend.backtest.quant_optimizer import _PARAM_BOUNDS, _INT_PARAMS, _CATEGORICAL_PARAMS

    # Read experiments with full params from TSV
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv"
    )
    experiments: list[dict] = []
    if os.path.exists(tsv_path):
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for idx, line in enumerate(f):
                values = line.strip().split("\t")
                if len(values) >= len(header):
                    row = dict(zip(header, values))
                    exp: dict = {
                        "index": idx,
                        "run_id": row.get("run_id", ""),
                        "param_changed": row.get("param_changed", ""),
                        "metric_before": _safe_float(row.get("metric_before", "0")),
                        "metric_after": _safe_float(row.get("metric_after", "0")),
                        "delta": _safe_float(row.get("delta", "0")),
                        "status": row.get("status", ""),
                        "dsr": _safe_float(row.get("dsr", "0")),
                        "top5_mda": row.get("top5_mda", "").split(",") if row.get("top5_mda") else [],
                        "timestamp": row.get("timestamp", ""),
                    }
                    # Parse params_json column if present
                    if "params_json" in row and row["params_json"]:
                        try:
                            exp["params_full"] = json.loads(row["params_json"])
                        except (json.JSONDecodeError, TypeError):
                            exp["params_full"] = None
                    else:
                        exp["params_full"] = None
                    experiments.append(exp)

    # Data scope from latest backtest result
    data_scope: dict = {}
    if _backtest_state["result"]:
        r = _backtest_state["result"]
        config = r.get("config", {})
        per_window = r.get("per_window", [])
        windows = []
        for w in per_window:
            windows.append({
                "id": w.get("window_id"),
                "train_start": w.get("train_start"),
                "train_end": w.get("train_end"),
                "test_start": w.get("test_start"),
                "test_end": w.get("test_end"),
                "n_candidates": w.get("n_candidates", 0),
                "n_train_samples": w.get("n_train_samples", 0),
            })
        data_scope = {
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "strategy": config.get("strategy", "triple_barrier"),
            "n_windows": len(per_window),
            "n_features": per_window[0].get("n_features", 0) if per_window else 0,
            "windows": windows,
        }

    return {
        "param_bounds": {k: list(v) for k, v in _PARAM_BOUNDS.items()},
        "int_params": list(_INT_PARAMS),
        "categorical_params": {k: list(v) for k, v in _CATEGORICAL_PARAMS.items()},
        "experiments": experiments,
        "data_scope": data_scope,
    }


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


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

        def progress_cb(data: dict):
            _backtest_state["progress"] = data

        engine = BacktestEngine(
            bq_client=bq.client,
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
            commission_model=settings.backtest_commission_model,
            commission_per_share=settings.backtest_commission_per_share,
        )

        result = await asyncio.to_thread(engine.run_backtest)

        # Compute baselines — emit progress so UI shows activity during this phase
        baselines = None
        if result.windows:
            try:
                elapsed = round(time.time() - engine._backtest_start_time, 1)
                progress_cb({
                    "window": engine._total_windows,
                    "total_windows": engine._total_windows,
                    "step": "finalizing",
                    "step_detail": "Comparing vs SPY, equal-weight & momentum benchmarks...",
                    "candidates_found": 0,
                    "samples_built": 0,
                    "samples_total": 0,
                    "elapsed_seconds": elapsed,
                    "cache_hits": 0,
                    "cache_misses": 0,
                })
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

        # Emit progress for report generation phase
        elapsed = round(time.time() - engine._backtest_start_time, 1)
        progress_cb({
            "window": engine._total_windows,
            "total_windows": engine._total_windows,
            "step": "finalizing",
            "step_detail": "Generating portfolio analytics report...",
            "candidates_found": 0,
            "samples_built": 0,
            "samples_total": 0,
            "elapsed_seconds": elapsed,
            "cache_hits": 0,
            "cache_misses": 0,
        })
        report = generate_report(result, baselines=baselines)
        report["run_id"] = run_id
        report["config"] = {
            "start_date": req.start_date or settings.backtest_start_date,
            "end_date": req.end_date or settings.backtest_end_date,
            "strategy": getattr(engine, "strategy", "triple_barrier"),
        }

        _backtest_state["status"] = "completed"
        _backtest_state["result"] = report
        _backtest_state["engine_source"] = None

        # Persist to disk so results survive restarts
        try:
            result_store.save_result(run_id, report)
        except Exception as save_err:
            logger.warning("Failed to persist backtest result: %s", save_err)

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        _backtest_state["status"] = "error"
        _backtest_state["error"] = str(e)
        _backtest_state["traceback"] = traceback.format_exc()


async def _run_optimizer_async(max_iterations: int, use_llm: bool):
    """Run quant optimizer in background thread."""
    global _optimizer_state
    try:
        settings = get_settings()
        bq = BigQueryClient(settings)

        from backend.backtest.backtest_engine import BacktestEngine
        from backend.backtest.quant_optimizer import QuantStrategyOptimizer

        def engine_progress_cb(data: dict):
            """Forward engine sub-step progress into optimizer + backtest state."""
            step = data.get("step", "")
            detail = data.get("step_detail", "")
            window = data.get("window", 0)
            total = data.get("total_windows", 0)
            _optimizer_state["current_detail"] = (
                f"W{window}/{total} {step}: {detail}" if window else detail
            )
            # Mirror into backtest progress so the unified progress panel renders
            _backtest_state["progress"] = data

        engine = BacktestEngine(
            bq_client=bq.client,
            project=settings.gcp_project_id,
            dataset=settings.bq_dataset_reports,
            start_date=settings.backtest_start_date,
            end_date=settings.backtest_end_date,
            progress_callback=engine_progress_cb,
            commission_model=settings.backtest_commission_model,
            commission_per_share=settings.backtest_commission_per_share,
        )

        def status_cb(iterations, best_sharpe, best_dsr, kept, discarded,
                      current_step="", current_detail="", run_id=""):
            _optimizer_state.update({
                "iterations": iterations,
                "best_sharpe": best_sharpe,
                "best_dsr": best_dsr,
                "kept": kept,
                "discarded": discarded,
                "current_step": current_step,
                "current_detail": current_detail,
                "run_id": run_id,
            })
            # Invalidate experiment cache so next poll gets fresh data
            get_api_cache().invalidate("backtest:experiments*")
            get_api_cache().invalidate("backtest:best")

        optimizer = QuantStrategyOptimizer(engine, status_callback=status_cb)

        # Wire MDA updates to MetaCoordinator for MDA->Agent bridge
        from backend.services.autonomous_loop import get_coordinator
        coordinator = get_coordinator()

        def on_result(report: dict):
            """Push optimizer baseline/kept results to backtest tabs."""
            _backtest_state["status"] = "completed"
            _backtest_state["result"] = report
            _backtest_state["error"] = None
            _backtest_state["engine_source"] = None

        # Set backtest state so the unified progress panel activates
        _backtest_state["status"] = "running"
        _backtest_state["run_id"] = _optimizer_state.get("run_id") or "opt"
        _backtest_state["progress"] = {"step": "initializing", "step_detail": "Optimizer starting engine..."}
        _backtest_state["result"] = None
        _backtest_state["error"] = None
        _backtest_state["traceback"] = None
        _backtest_state["engine_source"] = "optimizer"

        await asyncio.to_thread(
            optimizer.run_loop,
            max_iterations=max_iterations,
            use_llm=use_llm,
            stop_check=lambda: _optimizer_state["status"] != "running",
            on_mda_update=coordinator.update_mda_features,
            on_result=on_result,
        )

        _optimizer_state["status"] = "completed"
        # If no on_result fired (all discarded), reset backtest state
        if _backtest_state.get("engine_source") == "optimizer" and _backtest_state["status"] == "running":
            _backtest_state["status"] = "idle"
            _backtest_state["progress"] = {}
            _backtest_state["engine_source"] = None

    except Exception as e:
        try:
            logger.error(f"Optimizer failed: {e}", exc_info=True)
        except Exception:
            logger.error("Optimizer failed (encoding-safe): %s", ascii(str(e)))
        _optimizer_state["status"] = "error"
        _optimizer_state["error"] = str(e)
        _optimizer_state["traceback"] = traceback.format_exc()
        # Reset backtest state on optimizer error
        if _backtest_state.get("engine_source") == "optimizer":
            _backtest_state["status"] = "idle"
            _backtest_state["progress"] = {}
            _backtest_state["error"] = None
            _backtest_state["engine_source"] = None
