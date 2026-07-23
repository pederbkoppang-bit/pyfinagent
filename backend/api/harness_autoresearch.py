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
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/harness", tags=["harness"])

_BQ_TABLE = "pyfinagent_data.harness_learning_log"


# phase-23.6.4: restore observability symbols that tests expect (surfaced
# during 23.6.3 cleanup -- `structured_log` and `_read_audit_tail` were
# referenced by tests/api/test_observability.py but missing from this module).
# `structured_log` mirrors the sibling cost_budget_api.py helper so both
# endpoints emit the same stable JSON envelope; `_read_audit_tail` reads
# the last N JSONL events from a path with fail-open semantics.

def structured_log(endpoint: str, duration_ms: float, status: str, **extra) -> None:
    """phase-15.10/23.6.4: emit one structured JSON log line per endpoint call.

    Stable envelope: `{endpoint, duration_ms, status, ts, **extra}`. Extras
    flow through as-is so callers can attach context like `week_iso`,
    `truncated`, etc. Fail-open -- a JSON-serialization failure logs a
    WARNING and is swallowed.
    """
    try:
        logger.info(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "duration_ms": round(duration_ms, 1),
                    "status": status,
                    "ts": time.time(),
                    **extra,
                }
            )
        )
    except Exception as exc:
        logger.warning("structured_log fail-open: %r", exc)


def _read_audit_tail(path: Any, limit: int) -> tuple[list[dict], bool]:
    """phase-23.6.4: read the last `limit` JSONL events from `path`.

    Returns `(events, truncated)` where:
      - `events` is the list of parsed JSON objects (empty on any error).
      - `truncated` is True iff the file had more rows than `limit`.

    Fail-open: missing path / parse errors / encoding issues all return
    `([], False)` so the caller can render an empty audit tail without
    a 5xx.
    """
    try:
        p = Path(path) if not isinstance(path, Path) else path
        if not p.exists():
            return ([], False)
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        truncated = len(lines) > limit
        tail = lines[-limit:] if truncated else lines
        events: list[dict] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        return (events, truncated)
    except Exception as exc:
        logger.warning("_read_audit_tail fail-open (path=%r): %r", path, exc)
        return ([], False)


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
        rows = list(client.query(sql, job_config=cfg).result(timeout=30))
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
