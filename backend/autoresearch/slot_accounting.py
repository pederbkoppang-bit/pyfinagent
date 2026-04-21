"""phase-10.8 Slot accounting to BQ `pyfinagent_data.harness_learning_log`.

One canonical BQ sink for every phase-10 routine (10.3 Thursday batch, 10.4
Friday promotion, 10.6 monthly champion/challenger, 10.7 rollback). Lets the
Harness-tab (phase-10.9) query a single table for all phase-10 activity.

Writes via `insert_rows_json` (streaming), matches `bigquery_client.py:251`
idiom. Idempotency is handled upstream per-routine via `already_fired`
guards -- no insertId dedup needed here.

**Pitfall note (from research):** `backend/services/learning_logger.py:70`
uses the wrong dataset `project.trading.harness_learning_log`. This module
hard-defaults to `pyfinagent_data.harness_learning_log`.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_TABLE = "pyfinagent_data.harness_learning_log"
_VALID_SLOT_IDS = frozenset(["thu_batch", "fri_promotion", "monthly_gate", "rollback"])


def log_slot_usage(
    *,
    week_iso: str,
    slot_id: str,
    routine: str,
    result: dict[str, Any],
    phase: str = "phase-10",
    bq_insert_fn: Callable[[str, list[dict[str, Any]]], bool] | None = None,
    table: str = _DEFAULT_TABLE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Log a single phase-10 slot usage to BQ.

    Returns `{inserted, row_id, table, row}`.

    Raises `ValueError` if `slot_id` is not in the canonical set.
    """
    if slot_id not in _VALID_SLOT_IDS:
        raise ValueError(
            f"slot_id={slot_id!r} not in {sorted(_VALID_SLOT_IDS)}"
        )

    now_dt = now or datetime.now(timezone.utc)
    row_id = str(uuid.uuid4())
    try:
        result_json = json.dumps(result, sort_keys=True, default=str)
    except Exception as exc:
        logger.warning("slot_accounting: result serialize fail-open: %r", exc)
        result_json = "{}"

    row: dict[str, Any] = {
        "logged_at": now_dt.isoformat(),
        "row_id": row_id,
        "week_iso": week_iso,
        "slot_id": slot_id,
        "phase": phase,
        "routine": routine,
        "result_json": result_json,
        "status": str(result.get("status", "ok") if isinstance(result, dict) else "ok"),
        "error_msg": str(result.get("error", "")) if isinstance(result, dict) and result.get("error") else "",
    }

    inserted = False
    fn = bq_insert_fn or _default_bq_insert
    try:
        inserted = bool(fn(table, [row]))
    except Exception as exc:
        logger.warning("slot_accounting: BQ insert fail-open: %r", exc)
        inserted = False

    return {
        "inserted": inserted,
        "row_id": row_id,
        "table": table,
        "row": row,
    }


def verify_weekly_invariant(
    week_iso: str,
    *,
    bq_query_fn: Callable[[str, dict[str, Any]], int] | None = None,
    table: str = _DEFAULT_TABLE,
) -> dict[str, Any]:
    """Count `thu_batch` + `fri_promotion` rows for a week. Expect sum == 2.

    Returns `{week_iso, sum, satisfied}`.
    """
    sql = (
        f"SELECT COUNT(*) FROM `{table}` "
        "WHERE week_iso = @week_iso AND phase = 'phase-10' "
        "AND slot_id IN ('thu_batch', 'fri_promotion')"
    )
    params = {"week_iso": week_iso}
    fn = bq_query_fn or _default_bq_query_count
    try:
        total = int(fn(sql, params))
    except Exception as exc:
        logger.warning("slot_accounting: invariant query fail-open: %r", exc)
        total = 0
    return {"week_iso": week_iso, "sum": total, "satisfied": total == 2}


def _default_bq_insert(table: str, rows: list[dict[str, Any]]) -> bool:
    """Stream rows via `google.cloud.bigquery.insert_rows_json`. Fail-open."""
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        errors = client.insert_rows_json(f"{project}.{table}", rows)
        if errors:
            logger.warning("slot_accounting: insert errors: %r", errors)
            return False
        return True
    except Exception as exc:
        logger.warning("slot_accounting: BQ client fail-open: %r", exc)
        return False


def _default_bq_query_count(sql: str, params: dict[str, Any]) -> int:
    """Run a parameterized scalar COUNT query. Returns 0 on failure."""
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        query_params = [
            bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=query_params)
        rows = list(client.query(sql, job_config=cfg).result())
        return int(rows[0][0]) if rows else 0
    except Exception as exc:
        logger.warning("slot_accounting: query count fail-open: %r", exc)
        return 0


__all__ = [
    "log_slot_usage",
    "verify_weekly_invariant",
]
