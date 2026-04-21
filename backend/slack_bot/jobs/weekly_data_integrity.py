"""phase-9.7 Weekly data-integrity scan.

Row-count-drift check: compares current row counts against the prior week's
snapshot. If any table drifts by > threshold (default 20%), alerts Slack.

phase-9.9.1: APScheduler fires run() with zero args. `current_counts` and
`prior_counts` are therefore optional; if None at call time, `fetch_fn or
_default_fetch_counts` supplies current, and a JSON snapshot at
`snapshot_path` (default handoff/logs/row_count_snapshot.json) supplies prior.
After drift computation, current counts are saved back as next week's prior.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "weekly_data_integrity"
DRIFT_THRESHOLD = 0.20
_DEFAULT_SNAPSHOT_PATH = "handoff/logs/row_count_snapshot.json"


def run(
    *,
    current_counts: dict[str, int] | None = None,
    prior_counts: dict[str, int] | None = None,
    fetch_fn: Callable[[], dict[str, int]] | None = None,
    snapshot_path: str | None = None,
    alert_fn: Callable[[list[dict]], None] | None = None,
    store: IdempotencyStore | None = None,
    iso_year_week: str | None = None,
    drift_threshold: float = DRIFT_THRESHOLD,
) -> dict[str, Any]:
    key = IdempotencyKey.weekly(JOB_NAME, iso_year_week=iso_year_week)
    snapshot = snapshot_path or _DEFAULT_SNAPSHOT_PATH
    if current_counts is None:
        current_counts = (fetch_fn or _default_fetch_counts)()
    if prior_counts is None:
        prior_counts = _load_snapshot(snapshot)
    cur = current_counts or {}
    prev = prior_counts or {}
    result: dict[str, Any] = {"drifts": [], "key": key, "skipped": False}
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        drifts = _compute_drifts(cur, prev, threshold=drift_threshold)
        result["drifts"] = drifts
        if drifts and alert_fn is not None:
            try:
                alert_fn(drifts)
            except Exception as exc:
                logger.warning("data_integrity: alert_fn fail-open: %r", exc)
        try:
            _save_snapshot(cur, snapshot)
        except Exception as exc:
            logger.warning("data_integrity: snapshot save fail-open: %r", exc)
    return result


def _compute_drifts(cur: dict[str, int], prev: dict[str, int], *, threshold: float) -> list[dict]:
    drifts = []
    for table, cur_n in cur.items():
        prev_n = prev.get(table)
        if prev_n is None or prev_n == 0:
            continue
        delta = abs(cur_n - prev_n) / prev_n
        if delta > threshold:
            drifts.append({"table": table, "prev": prev_n, "cur": cur_n, "delta_pct": delta})
    return drifts


def _default_fetch_counts() -> dict[str, int]:
    """Query pyfinagent_data.__TABLES__ for current row counts. Fail-open to {}."""
    try:
        from backend.db.bigquery_client import BigQueryClient
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        dataset = os.getenv("PYFINAGENT_DATASET", "pyfinagent_data")
        client = BigQueryClient()
        sql = f"SELECT table_id, row_count FROM `{project}.{dataset}.__TABLES__`"
        rows = client.query(sql)
        return {r["table_id"]: int(r["row_count"]) for r in rows}
    except Exception as exc:
        logger.warning("data_integrity: __TABLES__ fetch fail-open: %r", exc)
        return {}


def _load_snapshot(path: str) -> dict[str, int]:
    """Load prior-week snapshot. Return empty dict if missing/unreadable."""
    try:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("data_integrity: snapshot load fail-open: %r", exc)
        return {}


def _save_snapshot(counts: dict[str, int], path: str) -> None:
    """Write current counts as next week's prior."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(counts, f, sort_keys=True)


__all__ = ["run", "JOB_NAME", "DRIFT_THRESHOLD"]
