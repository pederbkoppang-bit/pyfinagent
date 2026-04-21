"""phase-10.11 Harness autoresearch sprint-state endpoint.

`GET /api/harness/sprint-state?week_iso=YYYY-Www` returns a
`HarnessSprintWeekState` snapshot (or `null` for weeks with no data).

Reads `pyfinagent_data.harness_learning_log` (written by phase-10.8
`log_slot_usage`). Takes latest row per slot_id for the requested week.
Fail-open: BQ errors return `null`, never 5xx.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Callable, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/harness", tags=["harness"])

_BQ_TABLE = "pyfinagent_data.harness_learning_log"


class HarnessSprintThu(BaseModel):
    batchId: str
    candidatesKicked: int


class HarnessSprintFri(BaseModel):
    promotedIds: list[str]
    rejectedIds: list[str]


class HarnessSprintMonthly(BaseModel):
    sortinoDelta: float
    approvalPending: bool
    approved: bool


class HarnessSprintWeekState(BaseModel):
    weekIso: str
    thu: Optional[HarnessSprintThu] = None
    fri: Optional[HarnessSprintFri] = None
    monthly: Optional[HarnessSprintMonthly] = None


def _current_week_iso() -> str:
    iso = date.today().isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _project_rows_to_state(week_iso: str, rows_by_slot: dict[str, dict]) -> HarnessSprintWeekState | None:
    """Project latest rows (keyed by slot_id) into the camelCase wire shape.

    Returns None if no rows at all (distinguishes "no sprint activity" from
    "partial sprint activity").
    """
    if not rows_by_slot:
        return None

    state = HarnessSprintWeekState(weekIso=week_iso)

    thu_row = rows_by_slot.get("thu_batch")
    if thu_row:
        payload = _parse_result(thu_row.get("result_json"))
        batch_id = str(payload.get("batch_id", ""))
        kicked = int(payload.get("candidates_kicked", 0) or 0)
        if batch_id:
            state.thu = HarnessSprintThu(batchId=batch_id, candidatesKicked=kicked)

    fri_row = rows_by_slot.get("fri_promotion")
    if fri_row:
        payload = _parse_result(fri_row.get("result_json"))
        state.fri = HarnessSprintFri(
            promotedIds=[str(x) for x in payload.get("promoted_ids", [])],
            rejectedIds=[str(x) for x in payload.get("rejected_ids", [])],
        )

    monthly_row = rows_by_slot.get("monthly_gate")
    if monthly_row:
        payload = _parse_result(monthly_row.get("result_json"))
        sortino = payload.get("sortino_delta")
        if sortino is not None:
            state.monthly = HarnessSprintMonthly(
                sortinoDelta=float(sortino),
                approvalPending=bool(payload.get("approval_pending", False)),
                approved=bool(payload.get("approved", False)),
            )

    return state


def _parse_result(result_json: Any) -> dict[str, Any]:
    if result_json is None:
        return {}
    if isinstance(result_json, dict):
        return result_json
    try:
        return json.loads(result_json) if result_json else {}
    except Exception as exc:
        logger.warning("harness_sprint_state: result_json parse fail-open: %r", exc)
        return {}


def _build_sql(week_iso: str, table: str = _BQ_TABLE) -> str:
    """SQL kept as a module-level helper so tests can inspect the string."""
    return (
        f"SELECT slot_id, result_json, logged_at "
        f"FROM `{table}` "
        f"WHERE week_iso = @week_iso AND phase = 'phase-10' "
        f"ORDER BY logged_at DESC"
    )


def _default_bq_query(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the query against BQ; return list of row dicts. Fail-open to []."""
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        query_params = [
            bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=query_params)
        rows = list(client.query(sql, job_config=cfg).result())
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("harness_sprint_state: BQ query fail-open: %r", exc)
        return []


def fetch_sprint_state(
    *,
    week_iso: str | None = None,
    bq_query_fn: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None,
    table: str = _BQ_TABLE,
) -> HarnessSprintWeekState | None:
    """Pure function so tests can stub BQ with a list of rows.

    Returns the snapshot or None (no data for this week).
    """
    wk = week_iso or _current_week_iso()
    sql = _build_sql(wk, table=table)
    fn = bq_query_fn or _default_bq_query
    rows = fn(sql, {"week_iso": wk})

    # Take LATEST row per slot_id (SQL orders DESC, so first wins).
    rows_by_slot: dict[str, dict] = {}
    for row in rows:
        slot = str(row.get("slot_id", ""))
        if slot and slot not in rows_by_slot:
            rows_by_slot[slot] = row
    return _project_rows_to_state(wk, rows_by_slot)


@router.get("/sprint-state", response_model=Optional[HarnessSprintWeekState])
def get_sprint_state(week_iso: str | None = None):
    """Return the current (or requested) week's sprint state, or null."""
    return fetch_sprint_state(week_iso=week_iso)


__all__ = [
    "router",
    "fetch_sprint_state",
    "HarnessSprintWeekState",
    "HarnessSprintThu",
    "HarnessSprintFri",
    "HarnessSprintMonthly",
    "_build_sql",
    "_current_week_iso",
]
