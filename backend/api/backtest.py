"""
Backtest API — walk-forward backtesting + quant optimizer endpoints.
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
# Stash the last completed result so metric cards stay populated during a new run
_previous_result: dict | None = None

# Lazy-load flag — avoids blocking module import with glob+sort on results dir
_prev_loaded: bool = False


def _ensure_prev_loaded() -> None:
    """Lazy-load the latest persisted backtest result on first access."""
    global _prev_loaded, _previous_result
    if _prev_loaded:
        return
    _prev_loaded = True
    _prev = result_store.load_latest()
    if _prev:
        _backtest_state["status"] = "completed"
        _backtest_state["run_id"] = _prev.get("run_id")
        _backtest_state["result"] = _prev
        logger.info("Auto-loaded previous backtest result (run_id=%s)", _prev.get("run_id"))


# Dedicated executor for heavy backtest/optimizer tasks so they don't starve
# the default asyncio threadpool used by lightweight endpoints.
_heavy_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bt-heavy")

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
    global _backtest_state, _previous_result

    if _backtest_state["status"] == "running":
        raise HTTPException(400, "Backtest already running")
    if _optimizer_state["status"] == "running":
        raise HTTPException(409, "Optimizer is running - backtest unavailable")

    run_id = str(uuid.uuid4())[:8]
    if _backtest_state["result"] is not None:
        _previous_result = _backtest_state["result"]
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
    _ensure_prev_loaded()
    return {
        "status": _backtest_state["status"],
        "run_id": _backtest_state["run_id"],
        "progress": _backtest_state["progress"],
        "has_result": _backtest_state["result"] is not None or _previous_result is not None,
        "error": _backtest_state["error"],
        "traceback": _backtest_state["traceback"],
        "engine_source": _backtest_state.get("engine_source"),
    }


@router.get("/results")
async def get_backtest_results():
    """Full backtest results with analytics."""
    _ensure_prev_loaded()
    if _backtest_state["result"] is not None:
        return _backtest_state["result"]
    # Serve stale result while a new run is in progress
    if _previous_result is not None:
        return _previous_result
    if _backtest_state["status"] == "running":
        raise HTTPException(202, "Backtest still running")
    raise HTTPException(404, "No backtest results available")


@router.get("/results/{window_id}")
async def get_window_result(window_id: int):
    """Per-window detail."""
    active = _backtest_state["result"] or _previous_result
    if active is None:
        raise HTTPException(404, "No backtest results available")
    
    windows = active.get("per_window", [])
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
        status = await asyncio.to_thread(service.get_ingestion_status)
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
    """Current optimizer state. Also checks for CLI-launched optimizer."""
    import subprocess
    state = dict(_optimizer_state)
    
    # FIRST: Always check for running processes (highest priority)
    try:
        # Check for multiple optimizer-related processes
        optimizer_patterns = ["run_optimizer.py", "run_quick_test.py", "QuantStrategyOptimizer"]
        running_process = None
        for pattern in optimizer_patterns:
            result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
            if result.stdout.strip():
                running_process = pattern
                break
        
        if running_process:
            state["status"] = "running"
            state["current_step"] = f"Running {running_process.replace('.py', '')}"
            # Get current iteration count
            import os
            tsv_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv")
            if os.path.exists(tsv_path):
                with open(tsv_path) as f:
                    lines = f.readlines()
                non_baseline = [l for l in lines[1:] if "BASELINE" not in l and l.strip()]
                state["iterations"] = len(non_baseline)
            return state
    except Exception:
        pass
    
    # SECOND: Check if optimizer is idle but we have completed results  
    if state.get("status") == "idle":
        # Check if we have completed results but status wasn't properly updated
        import os
        best_results_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "experiments", "optimizer_best.json")
        if os.path.exists(best_results_path):
            try:
                import json
                with open(best_results_path) as f:
                    best_results = json.load(f)
                # If we have recent results (within last 24 hours), show completed status
                from datetime import datetime, timedelta
                saved_at = datetime.fromisoformat(best_results.get("saved_at", "").replace("Z", "+00:00"))
                if saved_at > datetime.now().astimezone() - timedelta(hours=24):
                    state.update({
                        "status": "completed",
                        "best_sharpe": best_results.get("sharpe"),
                        "best_dsr": best_results.get("dsr"),
                        "kept": best_results.get("kept", 0),
                        "discarded": best_results.get("discarded", 0),
                        "run_id": best_results.get("run_id"),
                    })
                    # Also get iteration count from TSV
                    tsv_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv")
                    if os.path.exists(tsv_path):
                        with open(tsv_path) as f:
                            lines = f.readlines()
                        non_baseline = [l for l in lines[1:] if "BASELINE" not in l and l.strip()]
                        state["iterations"] = len(non_baseline)
                    return state
            except Exception as e:
                pass
    return state


@router.get("/optimize/runs")
def get_optimizer_runs():
    """List optimizer runs (each BASELINE marks a new run). Returns summary per run, newest first."""
    import os
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv"
    )
    if not os.path.exists(tsv_path):
        return {"runs": []}

    # Parse TSV into groups: each BASELINE starts a new group
    groups: list[dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        current_group: dict | None = None
        for line in f:
            values = line.strip().split("\t")
            if len(values) < len(header):
                continue
            row = dict(zip(header, values + [""] * (len(header) - len(values))))
            if row.get("status") == "BASELINE":
                current_group = {
                    "baseline_ts": row.get("timestamp", ""),
                    "baseline_sharpe": row.get("metric_after", "0"),
                    "experiment_count": 0,
                    "kept": 0,
                    "discarded": 0,
                }
                groups.append(current_group)
            elif current_group is not None:
                current_group["experiment_count"] += 1
                st = row.get("status", "")
                if st == "keep":
                    current_group["kept"] += 1
                elif st in ("discard", "dsr_reject", "crash"):
                    current_group["discarded"] += 1

    # Assign index (0 = latest) and reverse so newest first
    for i, g in enumerate(reversed(groups)):
        g["index"] = i
    groups.reverse()
    return {"runs": groups}


@router.get("/optimize/experiments")
def get_optimizer_experiments(run_id: str | None = None, run_index: int | None = None):
    """Full experiment history from quant_results.tsv.

    Args:
        run_id: If provided, filter to experiments from this optimizer run only.
        run_index: If provided (0=latest), return experiments from that BASELINE group.
    """
    cache = get_api_cache()
    cache_key = f"backtest:experiments:{run_id or 'all'}:{run_index if run_index is not None else 'none'}"
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
            if len(values) >= len(header) - 1:  # tolerate missing trailing columns
                experiments.append(dict(zip(header, values + [""] * (len(header) - len(values)))))

    # Group by parent_run_id (preferred) or positional BASELINE boundaries (fallback)
    baselines = [e for e in experiments if e.get("status") == "BASELINE"]

    if run_index is not None:
        # run_index 0 = latest baseline
        sorted_baselines = list(reversed(baselines))
        if 0 <= run_index < len(sorted_baselines):
            target = sorted_baselines[run_index]
            target_id = target.get("run_id")
            children = [e for e in experiments if e.get("parent_run_id") == target_id]
            experiments = [target] + children
        else:
            experiments = []
    elif run_id:
        # Return the baseline + its children
        target_baseline = next((e for e in baselines if e.get("run_id") == run_id), None)
        if target_baseline:
            children = [e for e in experiments if e.get("parent_run_id") == run_id]
            experiments = [target_baseline] + children
        else:
            # Maybe run_id is an experiment — return its parent group
            target_exp = next((e for e in experiments if e.get("run_id") == run_id), None)
            parent_id = target_exp.get("parent_run_id") if target_exp else None
            if parent_id:
                parent = next((e for e in baselines if e.get("run_id") == parent_id), None)
                children = [e for e in experiments if e.get("parent_run_id") == parent_id]
                experiments = ([parent] if parent else []) + children
            else:
                experiments = [target_exp] if target_exp else []
    else:
        # Default: return latest baseline + its experiments
        last_baseline_idx = -1
        for idx, exp in enumerate(experiments):
            if exp.get("status") == "BASELINE":
                last_baseline_idx = idx
        if last_baseline_idx >= 0:
            experiments = experiments[last_baseline_idx:]

    result = {"experiments": experiments}
    cache.set(cache_key, result, ENDPOINT_TTLS["backtest:experiments"])
    return result


@router.get("/optimize/best")
def get_optimizer_best():
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
            if len(values) >= len(header) - 1:  # tolerate missing trailing columns
                row = dict(zip(header, values + [""] * (len(header) - len(values))))
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
def list_backtest_runs():
    """List all runs from TSV master index (newest first).

    Each row includes has_detail=True if a JSON result file exists on disk.
    The TSV is the single source of truth for the run list; JSON files are
    detail-only storage loaded via /runs/{run_id}.
    """
    import json as _json
    import os
    tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments", "quant_results.tsv"
    )
    if not os.path.exists(tsv_path):
        return {"runs": []}

    # Build set of run_ids that have JSON detail files
    detail_ids: set[str] = set()
    for entry in result_store.list_runs():
        rid = entry.get("run_id")
        if rid:
            detail_ids.add(rid)

    rows: list[dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            row = dict(zip(header, values + [""] * (len(header) - len(values))))
            rid = row.get("run_id", "")
            is_baseline = row.get("status") == "BASELINE"
            parent = row.get("parent_run_id", "")

            # Extract strategy from params_json, fall back to detail JSON
            strategy = "unknown"
            pj = row.get("params_json", "")
            if pj:
                try:
                    strategy = _json.loads(pj).get("strategy", "unknown")
                except (ValueError, TypeError):
                    pass
            if strategy == "unknown" and rid in detail_ids:
                # Try to get strategy from the JSON result file
                try:
                    detail = result_store.load_result(rid)
                    if detail:
                        strategy = (detail.get("params", {}).get("strategy") or
                                   detail.get("config", {}).get("strategy") or
                                   detail.get("strategy_params", {}).get("strategy") or
                                   "unknown")
                except Exception:
                    pass

            sharpe_str = row.get("metric_after", "")
            try:
                sharpe = float(sharpe_str)
            except (ValueError, TypeError):
                sharpe = None

            # Count experiments for baselines (read from same TSV data)
            experiment_count = 0
            if is_baseline:
                # Count children in the same TSV that have this run as parent
                with open(tsv_path, "r", encoding="utf-8") as exp_file:
                    exp_header = exp_file.readline().strip().split("\t")
                    child_count = 0
                    for exp_line in exp_file:
                        exp_values = exp_line.strip().split("\t") + [""] * max(0, len(exp_header) - len(exp_line.strip().split("\t")))
                        exp_row = dict(zip(exp_header, exp_values))
                        if exp_row.get("parent_run_id") == rid:
                            child_count += 1
                    experiment_count = child_count
            
            rows.append({
                "run_id": rid,
                "timestamp": row.get("timestamp", ""),
                "strategy": strategy,
                "sharpe": sharpe,
                "status": row.get("status", ""),
                "param_changed": row.get("param_changed", ""),
                "delta": row.get("delta", ""),
                "dsr": row.get("dsr", ""),
                "is_baseline": is_baseline,
                "parent_run_id": parent if parent else None,
                "has_detail": rid in detail_ids,
                "experiment_count": experiment_count,
            })

    # Append JSON-only runs not in TSV (legacy standalone backtests)
    tsv_ids = {r["run_id"] for r in rows}
    for entry in result_store.list_runs():
        rid = entry.get("run_id", "")
        if rid and rid not in tsv_ids:
            rows.append({
                "run_id": rid,
                "timestamp": entry.get("timestamp", ""),
                "strategy": entry.get("strategy", "unknown"),
                "sharpe": entry.get("sharpe"),
                "status": "BASELINE",
                "param_changed": "",
                "delta": "",
                "dsr": "",
                "is_baseline": True,
                "parent_run_id": None,
                "has_detail": True,
                "experiment_count": 0,  # JSON-only runs have no TSV experiments
            })

    rows.reverse()  # newest first
    return {"runs": rows}


@router.get("/runs/{run_id}")
def get_backtest_run(run_id: str):
    """Load a specific historical backtest result."""
    data = result_store.load_result(run_id)
    if data is None:
        raise HTTPException(404, f"No saved result for run_id={run_id}")
    return data


@router.delete("/runs/{run_id}")
def delete_backtest_run(run_id: str):
    """Delete a specific backtest result from disk."""
    global _previous_result
    deleted = result_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(404, f"No saved result for run_id={run_id}")
    # Clear in-memory state if the deleted run is the one currently displayed
    if _backtest_state.get("run_id") == run_id:
        _backtest_state["result"] = None
        _backtest_state["status"] = "idle"
        _backtest_state["run_id"] = None
        _backtest_state["engine_source"] = None
    if _previous_result and _previous_result.get("run_id") == run_id:
        _previous_result = None
    return {"status": "deleted", "run_id": run_id}


@router.delete("/optimize/history")
def delete_optimizer_history():
    """Delete optimizer experiment history (quant_results.tsv + optimizer_best.json).

    Use this to clear stale baselines after code changes that affect Sharpe/DSR calculation.
    """
    import os

    experiments_dir = os.path.join(
        os.path.dirname(__file__), "..", "backtest", "experiments"
    )
    deleted_files: list[str] = []

    tsv_path = os.path.join(experiments_dir, "quant_results.tsv")
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
        deleted_files.append("quant_results.tsv")

    best_path = os.path.join(experiments_dir, "optimizer_best.json")
    if os.path.exists(best_path):
        os.remove(best_path)
        deleted_files.append("optimizer_best.json")

    # Also delete saved backtest results so optimizer doesn't warm-start from stale data
    results_dir = os.path.join(experiments_dir, "results")
    if os.path.isdir(results_dir):
        import glob
        for f in glob.glob(os.path.join(results_dir, "*.json")):
            os.remove(f)
            deleted_files.append(f"results/{os.path.basename(f)}")

    # Clear cached API responses so next fetch sees empty state
    cache = get_api_cache()
    cache.invalidate("backtest:experiments*")
    cache.invalidate("backtest:best")
    cache.invalidate("backtest:insights")
    cache.invalidate("backtest:runs*")

    # Clear in-memory state so stale results don't survive file deletion
    global _previous_result
    _backtest_state["result"] = None
    _backtest_state["status"] = "idle"
    _backtest_state["run_id"] = None
    _backtest_state["engine_source"] = None
    _previous_result = None

    logger.info("Deleted optimizer history: %s", deleted_files)
    return {"status": "deleted", "files": deleted_files}


# ── Optimizer Insights ───────────────────────────────────────────

@router.get("/optimize/insights")
def get_optimizer_insights():
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
                if len(values) >= len(header) - 1:  # tolerate missing trailing columns
                    row = dict(zip(header, values + [""] * (len(header) - len(values))))
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

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_heavy_executor, engine.run_backtest)

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
        _previous_result = None  # fresh result available; discard stale copy

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
            global _previous_result
            _backtest_state["status"] = "completed"
            _backtest_state["result"] = report
            _backtest_state["error"] = None
            _backtest_state["engine_source"] = None
            _previous_result = None  # fresh result available

        # Stash current result before clearing
        global _previous_result
        if _backtest_state["result"] is not None:
            _previous_result = _backtest_state["result"]
        # Set backtest state so the unified progress panel activates
        _backtest_state["status"] = "running"
        _backtest_state["run_id"] = _optimizer_state.get("run_id") or "opt"
        _backtest_state["progress"] = {"step": "initializing", "step_detail": "Optimizer starting engine..."}
        _backtest_state["result"] = None
        _backtest_state["error"] = None
        _backtest_state["traceback"] = None
        _backtest_state["engine_source"] = "optimizer"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _heavy_executor,
            partial(
                optimizer.run_loop,
                max_iterations=max_iterations,
                use_llm=use_llm,
                stop_check=lambda: _optimizer_state["status"] != "running",
                on_mda_update=coordinator.update_mda_features,
                on_result=on_result,
            ),
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


# ── Sharpe History Endpoint ────────────────────────────────────

@router.get("/sharpe-history")
def get_sharpe_history():
    """Return a timeline of Sharpe ratio progression across all backtest runs.

    Cross-references result JSON files with quant_results.tsv for kept/discarded status.
    """
    from pathlib import Path
    import csv

    results_dir = Path(__file__).parent.parent / "backtest" / "experiments" / "results"
    tsv_path = Path(__file__).parent.parent / "backtest" / "experiments" / "quant_results.tsv"

    # -- Load TSV status map: run_id -> {status, parent_run_id, dsr} --------
    tsv_status: dict[str, dict] = {}
    if tsv_path.exists():
        try:
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    rid = row.get("run_id", "").strip()
                    if rid:
                        raw_status = row.get("status", "").strip()
                        tsv_status[rid] = {
                            "status": raw_status,
                            "parent_run_id": row.get("parent_run_id", "").strip() or None,
                            "dsr": row.get("dsr", "").strip(),
                        }
        except Exception as e:
            logger.warning("Failed to read quant_results.tsv: %s", e)

    # -- Scan all result JSON files -----------------------------------------
    timeline: list[dict] = []
    if results_dir.exists():
        for p in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping corrupt result %s: %s", p.name, e)
                continue

            analytics = data.get("analytics", {})
            sharpe = analytics.get("sharpe")
            if sharpe is None:
                continue

            run_id = data.get("run_id", p.stem.split("_", 1)[-1])
            is_baseline = data.get("is_baseline", False)
            parent_run_id = data.get("parent_run_id")

            # Determine status from TSV
            tsv_entry = tsv_status.get(run_id, {})
            raw_status = tsv_entry.get("status", "")
            if raw_status.upper() == "BASELINE":
                status = "baseline"
            elif raw_status.lower() in ("keep", "kept"):
                status = "kept"
            elif raw_status.lower() in ("discard", "discarded", "dsr_reject"):
                status = "discarded"
            elif raw_status:
                status = raw_status.lower()
            else:
                status = "unknown"

            # DSR: prefer TSV, fall back to JSON analytics
            dsr_val = None
            tsv_dsr = tsv_entry.get("dsr", "")
            if tsv_dsr:
                try:
                    dsr_val = float(tsv_dsr)
                except (ValueError, TypeError):
                    pass
            if dsr_val is None:
                dsr_val = analytics.get("deflated_sharpe")

            # Extract timestamp from filename (ISO-sortable)
            ts_str = result_store._extract_timestamp(p.stem)
            # Convert to ISO 8601 with dashes
            if len(ts_str) >= 15 and "T" in ts_str:
                # 20260325T225146Z -> 2026-03-25T22:51:46Z
                raw = ts_str.replace("Z", "")
                parts = raw.split("T")
                if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    d, t = parts
                    ts_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"

            timeline.append({
                "timestamp": ts_str,
                "run_id": run_id,
                "sharpe": round(sharpe, 4),
                "dsr": round(dsr_val, 4) if dsr_val is not None else None,
                "total_return_pct": round(analytics.get("total_return_pct", 0), 2),
                "max_drawdown_pct": round(analytics.get("max_drawdown", 0), 2),
                "n_trades": analytics.get("n_trades", 0),
                "is_baseline": is_baseline,
                "parent_run_id": parent_run_id,
                "status": status,
            })

    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])

    # -- Compute best_sharpe_so_far and envelope ----------------------------
    best_so_far = float("-inf")
    envelope: list[dict] = []
    for entry in timeline:
        s = entry["sharpe"]
        if s > best_so_far:
            best_so_far = s
            envelope.append({"timestamp": entry["timestamp"], "sharpe": s})
        entry["best_sharpe_so_far"] = round(best_so_far, 4)

    # -- Summary ------------------------------------------------------------
    kept_count = sum(1 for e in timeline if e["status"] == "kept")
    discarded_count = sum(1 for e in timeline if e["status"] == "discarded")
    initial_sharpe = timeline[0]["sharpe"] if timeline else 0
    current_best = best_so_far if best_so_far > float("-inf") else 0
    improvement = round((current_best - initial_sharpe) / initial_sharpe * 100, 1) if initial_sharpe else 0

    return {
        "timeline": timeline,
        "best_sharpe_envelope": envelope,
        "summary": {
            "initial_sharpe": initial_sharpe,
            "current_best_sharpe": round(current_best, 4),
            "improvement_pct": improvement,
            "total_experiments": len(timeline),
            "kept_count": kept_count,
            "discarded_count": discarded_count,
        },
    }


# ── Harness Dashboard Endpoints ─────────────────────────────────

import os as _os
_HANDOFF_DIR = _os.path.join(_os.path.dirname(__file__), "..", "..", "handoff")


def _read_handoff_file(filename: str) -> str | None:
    """Read a handoff file, return contents or None."""
    path = _os.path.join(_HANDOFF_DIR, filename)
    if not _os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning("Failed to read handoff/%s: %s", filename, e)
        return None


@router.get("/harness/log")
def get_harness_log():
    """Parse harness_log.md into structured cycle data."""
    import re
    content = _read_handoff_file("harness_log.md")
    if not content:
        return {"cycles": []}

    cycles: list[dict] = []
    # Split on cycle headers: ## Cycle N -- timestamp
    parts = re.split(r"^## (Cycle \d+)\s*--\s*(.+)$", content, flags=re.MULTILINE)
    i = 1
    while i + 2 < len(parts):
        cycle_name = parts[i].strip()
        timestamp = parts[i + 1].strip()
        body = parts[i + 2].strip()

        cycle: dict = {"cycle": cycle_name, "timestamp": timestamp}

        # Extract fields
        for field, key in [
            (r"\*\*Planner hypothesis:\*\*\s*(.+)", "hypothesis"),
            (r"\*\*Generator:\*\*\s*(.+)", "generator"),
            (r"\*\*Evaluator verdict:\*\*\s*(\w+)", "verdict"),
            (r"\*\*Decision:\*\*\s*(.+)", "decision"),
            (r"\*\*Total cycle time:\*\*\s*(.+)", "duration"),
        ]:
            m = re.search(field, body)
            if m:
                cycle[key] = m.group(1).strip()

        # Extract scores
        scores: dict = {}
        for score_match in re.finditer(r"- (Statistical|Robustness|Simplicity|Reality Gap):\s*(\d+)/10", body):
            scores[score_match.group(1).lower().replace(" ", "_")] = int(score_match.group(2))
        cycle["scores"] = scores

        # Extract sub-periods and 2x costs
        sp = re.search(r"- Sub-periods:\s*(.+)", body)
        if sp:
            cycle["sub_periods"] = sp.group(1).strip()
        costs = re.search(r"- 2x costs:\s*Sharpe=([0-9.]+)", body)
        if costs:
            cycle["costs_2x_sharpe"] = float(costs.group(1))

        cycles.append(cycle)
        i += 3

    return {"cycles": cycles}


@router.get("/harness/critique")
def get_harness_critique():
    """Return latest evaluator_critique.md as structured data."""
    content = _read_handoff_file("evaluator_critique.md")
    if not content:
        return {"content": None, "raw": None}
    return {"content": content, "raw": content}


@router.get("/harness/contract")
def get_harness_contract():
    """Return current contract.md."""
    content = _read_handoff_file("contract.md")
    if not content:
        return {"content": None}
    return {"content": content}


@router.get("/harness/validation")
def get_harness_validation():
    """Return validation results from JSON files."""
    validation: dict = {}
    subperiod: dict = {}

    val_path = _os.path.join(_HANDOFF_DIR, "validation_results.json")
    if _os.path.exists(val_path):
        try:
            with open(val_path, "r") as f:
                validation = json.load(f)
        except Exception:
            pass

    sub_path = _os.path.join(_HANDOFF_DIR, "subperiod_validation_results.json")
    if _os.path.exists(sub_path):
        try:
            with open(sub_path, "r") as f:
                subperiod = json.load(f)
        except Exception:
            pass

    return {"validation": validation, "subperiod": subperiod}


@router.get("/harness/criteria")
def get_harness_criteria():
    """Return evaluator criteria."""
    content = _read_handoff_file("evaluator_criteria.md")
    if not content:
        return {"content": None}
    return {"content": content}


# ── Budget Dashboard Endpoints ───────────────────────────────────

@router.get("/budget/summary")
def get_budget_summary():
    """Return budget summary with known costs and projections."""
    # Known fixed monthly costs
    fixed_costs = [
        {"category": "Claude Max Subscription", "monthly_usd": 200.00, "type": "fixed", "note": "Anthropic API access"},
        {"category": "Google Cloud (BQ + Storage)", "monthly_usd": 15.00, "type": "estimated", "note": "~$0.50/day average"},
        {"category": "Mac Mini (amortized)", "monthly_usd": 25.00, "type": "fixed", "note": "$600 / 24 months"},
        {"category": "Domain & Infrastructure", "monthly_usd": 10.00, "type": "estimated", "note": "DNS, misc"},
    ]

    # Estimated variable costs (when paper trading activates)
    variable_costs = [
        {"category": "Gemini API (Paper Trading)", "monthly_usd": 75.00, "type": "projected", "note": "~$2-5/day when active"},
        {"category": "Data APIs", "monthly_usd": 0.00, "type": "actual", "note": "Yahoo Finance + FRED (free)"},
    ]

    total_fixed = sum(c["monthly_usd"] for c in fixed_costs)
    total_variable = sum(c["monthly_usd"] for c in variable_costs)
    total_monthly = total_fixed + total_variable

    # Budget constraints from Peder
    monthly_budget = 350.00  # Peder's max comfort zone
    cash_available = 2000.00  # Estimated operating cash (conservative)
    runway_months = round(cash_available / total_monthly, 1) if total_monthly > 0 else 99

    return {
        "fixed_costs": fixed_costs,
        "variable_costs": variable_costs,
        "summary": {
            "total_fixed_monthly": total_fixed,
            "total_variable_monthly": total_variable,
            "total_monthly": total_monthly,
            "monthly_budget": monthly_budget,
            "budget_utilization_pct": round(total_monthly / monthly_budget * 100, 1),
            "cash_available": cash_available,
            "runway_months": runway_months,
        },
        "status": "under_budget" if total_monthly <= monthly_budget else "over_budget",
        "note": "Costs are estimates. Actual BQ billing data not yet integrated.",
    }
