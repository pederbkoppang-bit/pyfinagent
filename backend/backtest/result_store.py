"""
Backtest result persistence — save/load/list JSON results on disk.
Eliminates the need to re-run backtests after every app restart.

Storage: backend/backtest/experiments/results/{ISO_timestamp}_{run_id}.json
Natural filesystem sort → newest file last.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent / "experiments" / "results"


def _ensure_dir() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(run_id: str, report: dict) -> Path:
    """Persist a backtest report to JSON on disk.

    Filename: ``{ISO_timestamp}_{run_id}.json`` where the timestamp comes from
    the report's ``completed_at`` field or is derived from the current time.
    """
    _ensure_dir()

    # Use current UTC time for the filename prefix (sortable)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Sanitize run_id to be filesystem-safe
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
    filename = f"{ts}_{safe_id}.json"
    path = _RESULTS_DIR / filename

    path.write_text(json.dumps(report, default=str), encoding="utf-8")
    logger.info("Saved backtest result to %s", path.name)
    return path


def load_result(run_id: str) -> dict | None:
    """Load a specific backtest result by run_id."""
    _ensure_dir()
    # First try filename-based lookup (fast path)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
    for p in _RESULTS_DIR.glob(f"*_{safe_id}.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", p.name, e)
    # Fallback: scan all files for matching run_id field
    for p in _RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("run_id") == run_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def load_latest() -> dict | None:
    """Load the most recent backtest result (by filename sort order)."""
    _ensure_dir()
    files = sorted(_RESULTS_DIR.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read latest result: %s", e)
        return None


def list_runs() -> list[dict]:
    """Return summary metadata for all saved backtest runs.

    Returns list of ``{run_id, timestamp, strategy, sharpe, total_return_pct, filename}``,
    sorted newest-first.
    """
    _ensure_dir()
    runs: list[dict] = []
    for p in sorted(_RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            analytics = data.get("analytics", {})
            config = data.get("config", {})
            runs.append({
                "run_id": data.get("run_id", p.stem.split("_", 1)[-1]),
                "timestamp": p.stem.split("_", 1)[0],
                "strategy": config.get("strategy", data.get("strategy_params", {}).get("strategy", "unknown")),
                "sharpe": analytics.get("sharpe"),
                "total_return_pct": analytics.get("total_return_pct"),
                "filename": p.name,
                "is_baseline": data.get("is_baseline", False),
                "parent_run_id": data.get("parent_run_id"),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping corrupt result file %s: %s", p.name, e)
    return runs


def delete_run(run_id: str) -> bool:
    """Delete a specific result file by run_id. Returns True if deleted."""
    _ensure_dir()
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
    for p in _RESULTS_DIR.glob(f"*_{safe_id}.json"):
        p.unlink()
        logger.info("Deleted backtest result %s", p.name)
        return True
    # Fallback: scan all files for matching run_id field
    for p in _RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("run_id") == run_id:
                p.unlink()
                logger.info("Deleted backtest result %s", p.name)
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False
