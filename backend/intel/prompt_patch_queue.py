"""phase-6.5.7 prompt-patch queue over `intel_prompt_patches`.

Append-only. Status transitions (`pending` -> `approved`/`rejected`/`applied`/`expired`)
are recorded by inserting a NEW row, not by updating. Reads use
`latest-status-per-patch_id` SQL.

Dedup key: `patch_id = sha256(patch_type + ":" + patch_text + ":" + (chunk_id or ""))[:16]`.

Fail-open on every BQ call. ASCII-only logger messages.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_TABLE = "intel_prompt_patches"


def _patch_id(patch_type: str, patch_text: str, chunk_id: str | None) -> str:
    h = hashlib.sha256(
        f"{patch_type}:{patch_text}:{chunk_id or ''}".encode("utf-8")
    ).hexdigest()
    return h[:16]


def dedup(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In-memory dedup on `patch_id` (first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in patches:
        pid = p.get("patch_id") or _patch_id(
            p.get("patch_type", ""),
            p.get("patch_text", ""),
            p.get("chunk_id"),
        )
        if pid in seen:
            continue
        seen.add(pid)
        q = dict(p)
        q["patch_id"] = pid
        out.append(q)
    return out


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("prompt_patch_queue: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("prompt_patch_queue: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning(
            "prompt_patch_queue: bigquery.Client() init failed (%r)", exc
        )
        return None


def _insert(rows: list[dict[str, Any]], *, project: str | None, dataset: str | None) -> int:
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_client(proj)
    if client is None:
        return 0
    try:
        table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("prompt_patch_queue insert errors: %s", errors)
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("prompt_patch_queue insert fail-open: %r", exc)
        return 0


def enqueue_patch(
    patch_type: str,
    patch_text: str,
    *,
    chunk_id: str | None = None,
    rationale: str | None = None,
    metadata: dict[str, Any] | None = None,
    project: str | None = None,
    dataset: str | None = None,
) -> str:
    """Compute deterministic patch_id and attempt to insert a `pending` row.

    Always returns the patch_id (deterministic from inputs). BQ failures and
    dedup-skips are logged and swallowed; callers that need insert-confirmation
    should call `get_pending` afterwards. Returning the pid unconditionally lets
    callers trust that a repeated `enqueue_patch(...)` is idempotent.
    """
    pid = _patch_id(patch_type, patch_text, chunk_id)
    row = {
        "patch_id": pid,
        "chunk_id": chunk_id,
        "patch_type": patch_type,
        "patch_text": patch_text,
        "rationale": rationale,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": None,
        "reviewed_by": None,
        "applied_at": None,
        "metadata": metadata or {},
    }
    _insert([row], project=project, dataset=dataset)
    return pid


def get_pending(
    limit: int = 50,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Return up to `limit` patches whose latest status is `pending`. Fail-open."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_client(proj)
    if client is None:
        return []
    try:
        table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
        sql = f"""
WITH ranked AS (
  SELECT
    patch_id, chunk_id, patch_type, patch_text, rationale, status,
    created_at, reviewed_at, reviewed_by, applied_at, metadata,
    ROW_NUMBER() OVER (PARTITION BY patch_id ORDER BY created_at DESC) AS rn
  FROM `{table_ref}`
)
SELECT patch_id, chunk_id, patch_type, patch_text, rationale, status,
       created_at, reviewed_at, reviewed_by, applied_at, metadata
FROM ranked
WHERE rn = 1 AND status = 'pending'
ORDER BY created_at ASC
LIMIT {int(limit)}
"""
        job = client.query(sql)
        return [dict(r) for r in job.result(timeout=30)]
    except Exception as exc:
        logger.warning("prompt_patch_queue get_pending fail-open: %r", exc)
        return []


def mark_approved(
    patch_id: str,
    reviewed_by: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> bool:
    row = {
        "patch_id": patch_id,
        "chunk_id": None,
        "patch_type": "approval",
        "patch_text": "",
        "rationale": None,
        "status": "approved",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": reviewed_by,
        "applied_at": None,
        "metadata": {},
    }
    return _insert([row], project=project, dataset=dataset) > 0


def mark_rejected(
    patch_id: str,
    reason: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> bool:
    row = {
        "patch_id": patch_id,
        "chunk_id": None,
        "patch_type": "rejection",
        "patch_text": "",
        "rationale": reason,
        "status": "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": None,
        "applied_at": None,
        "metadata": {},
    }
    return _insert([row], project=project, dataset=dataset) > 0


__all__ = [
    "dedup",
    "enqueue_patch",
    "get_pending",
    "mark_approved",
    "mark_rejected",
    "_patch_id",
]
