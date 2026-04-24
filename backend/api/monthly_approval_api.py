"""phase-15.3 Monthly HITL approval endpoints.

Thin read/mutate surface over `handoff/logs/monthly_approval_state.json`.
Delegates state persistence + expiry transitions to
`backend.autoresearch.monthly_champion_challenger.record_approval()`;
this module only handles HTTP routing + response shaping.

Endpoints:
- `GET  /api/harness/monthly-approval/status[?month_key=YYYY-MM]` -- read.
- `POST /api/harness/monthly-approval/{month_key}`                -- body `{"action":"approved"|"rejected"}`.

Fail-open: missing file / corrupt state return `{"status":"no_pending"}`
rather than 5xx. POST against a month with no pending row returns
`{"status":"pending", "reason":"no_row_to_resolve"}` so callers can
distinguish "nothing happened" from the real terminal states.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.autoresearch.monthly_champion_challenger import (
    _DEFAULT_STATE_PATH,
    record_approval,
)
from backend.config.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/harness/monthly-approval",
    tags=["monthly-approval"],
)

# Status values the frontend + verification command may see.
_ALLOWED_ACTIONS = frozenset({"approved", "rejected"})


class MonthlyApprovalState(BaseModel):
    status: str
    month: Optional[str] = None
    sortino_delta: Optional[float] = None
    dd_ratio: Optional[float] = None
    pbo: Optional[float] = None
    expires_at_iso: Optional[str] = None
    created_at_iso: Optional[str] = None
    resolved_at_iso: Optional[str] = None
    challenger_id: Optional[str] = None
    reason: Optional[str] = None


class ApprovalActionBody(BaseModel):
    action: str


def _current_month_key() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _load_state_file(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        txt = path.read_text(encoding="utf-8")
        if not txt.strip():
            return {}
        data = json.loads(txt)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("monthly_approval_api: state load fail-open: %r", exc)
        return {}


def _row_to_model(row: dict[str, Any], *, effective_status: Optional[str] = None) -> MonthlyApprovalState:
    return MonthlyApprovalState(
        status=effective_status or str(row.get("status", "no_pending")),
        month=row.get("month"),
        sortino_delta=row.get("sortino_delta"),
        dd_ratio=row.get("dd_ratio"),
        pbo=row.get("pbo"),
        expires_at_iso=row.get("expires_at_iso"),
        created_at_iso=row.get("created_at_iso"),
        resolved_at_iso=row.get("resolved_at_iso"),
        challenger_id=row.get("challenger_id"),
        reason=row.get("reason"),
    )


_STRATEGY_DEPLOYMENTS_LOG_TYPES: dict[str, str] = {
    "strategy_id": "STRING",
    "status": "STRING",
    "sharpe": "FLOAT64",
    "dsr": "FLOAT64",
    "pbo": "FLOAT64",
    "max_dd": "FLOAT64",
    "deployed_at": "TIMESTAMP",
    "allocation_pct": "FLOAT64",
    "notes": "STRING",
}


def _default_bq_logger(log_row: dict[str, Any]) -> None:
    """Production BQ audit writer for HITL terminal transitions.

    Inserts a row into `<project>.pyfinagent_pms.strategy_deployments_log`.
    Fail-open: BQ-client construction or insert errors are logged and
    swallowed so the in-memory / JSON transition still completes.

    Binds NULL values using the column's real BQ type (not STRING) --
    BQ rejects a STRING-typed NULL into a FLOAT64 column.
    """
    try:
        from google.cloud import bigquery  # local import -- avoid eager GCP auth cost

        settings = Settings()
        project = settings.gcp_project_id
        client = bigquery.Client(project=project)
        table_id = f"{project}.pyfinagent_pms.strategy_deployments_log"
        cols = ", ".join(log_row.keys())
        vals = ", ".join(f"@v_{k}" for k in log_row.keys())
        query = f"INSERT INTO `{table_id}` ({cols}) VALUES ({vals})"
        params = []
        for k, v in log_row.items():
            col_type = _STRATEGY_DEPLOYMENTS_LOG_TYPES.get(k, "STRING")
            if v is None:
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", col_type, None))
            elif col_type == "FLOAT64":
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "FLOAT64", float(v)))
            elif col_type == "TIMESTAMP":
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "TIMESTAMP", v))
            else:
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v)))
        job = client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=params))
        job.result()
    except Exception as exc:
        logger.warning("_default_bq_logger fail-open: %r", exc)


def _virtual_status(row: dict[str, Any], now: datetime) -> str:
    """Surface `expired` when pending + now >= expires_at, without writing.

    Actual file transition happens lazily on the next POST or on
    `run_monthly_sortino_gate`. This keeps GET a pure read.
    """
    status = str(row.get("status", "no_pending"))
    if status != "pending":
        return status
    expires_iso = row.get("expires_at_iso")
    if not expires_iso:
        return status
    try:
        expires_at = datetime.fromisoformat(expires_iso)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except Exception:
        return status
    return "expired" if now >= expires_at else "pending"


@router.get("/status", response_model=MonthlyApprovalState)
def get_monthly_approval_status(
    month_key: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$"),
) -> MonthlyApprovalState:
    key = month_key or _current_month_key()
    state = _load_state_file(_DEFAULT_STATE_PATH)
    row = state.get(key)
    if not isinstance(row, dict):
        return MonthlyApprovalState(
            status="no_pending",
            month=key,
            reason="no_state_row_for_month",
        )
    effective = _virtual_status(row, datetime.now(timezone.utc))
    return _row_to_model(row, effective_status=effective)


@router.post("/{month_key}", response_model=MonthlyApprovalState)
def post_monthly_approval(month_key: str, body: ApprovalActionBody) -> MonthlyApprovalState:
    action = (body.action or "").strip().lower()
    if action not in _ALLOWED_ACTIONS:
        # Preserve allow-list invariant -- degrade to rejected on bad action.
        return MonthlyApprovalState(
            status="rejected",
            month=month_key,
            reason=f"invalid_action:{body.action!r}",
        )

    try:
        updated = record_approval(month_key, status=action, bq_fn=_default_bq_logger)
    except Exception as exc:
        logger.warning("monthly_approval_api: record_approval fail-open: %r", exc)
        updated = {}

    if not updated:
        # No pending row existed. Conservative policy: leave the file untouched
        # and return "pending" so callers know nothing has resolved yet.
        return MonthlyApprovalState(
            status="pending",
            month=month_key,
            reason="no_row_to_resolve",
        )

    return _row_to_model(updated)


__all__ = [
    "router",
    "MonthlyApprovalState",
    "ApprovalActionBody",
]
