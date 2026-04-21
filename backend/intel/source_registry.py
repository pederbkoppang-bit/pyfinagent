"""phase-6.5.2 intel source registry.

Thin BQ-backed store over `intel_sources` (created by phase-6.5.1 migration).
YAML is a seed / test-fixture format; BQ is the source of truth in production.

Functions:

- `load_from_yaml(path)` -> list[SourceRow]  -- pure, no BQ import
- `upsert_sources(rows, *, project=None, dataset=None)` -> int  -- fail-open
- `load_active_sources(*, project=None, dataset=None)` -> list[SourceRow]
    -- fail-open; filters `kill_switch = false`

House pattern mirrored:
- Fail-open BQ client:  `backend/news/bq_writer.py:61-72`
- Never-raise insert:   `backend/news/bq_writer.py:75-97`
- Settings resolution:  `backend/news/bq_writer.py:41-58`
- YAML + dataclass:     `backend/governance/limits_schema.py`

ASCII-only logger messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_INTEL_SOURCES_TABLE = "intel_sources"


@dataclass
class SourceRow:
    source_id: str
    source_name: str
    source_type: str
    kill_switch: bool
    rate_limit_per_day: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    """Resolve (project, dataset) from args + settings. Never raises."""
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
            logger.warning("intel.registry: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_client(project: str) -> Any:
    """Return a BQ Client or None (fail-open)."""
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("intel.registry: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("intel.registry: bigquery.Client() init failed (%r)", exc)
        return None


def load_from_yaml(path: str | Path) -> list[SourceRow]:
    """Parse a YAML file and return the declared sources.

    Returns an empty list if the file does not exist (fail-open).
    Raises ValueError only if the YAML root is structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("intel.registry: yaml missing at %s", p)
        return []
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("intel.registry: yaml parse error %r", exc)
        return []
    if not isinstance(raw, dict):
        raise ValueError("intel yaml root must be a mapping")
    items = raw.get("sources") or []
    if not isinstance(items, list):
        raise ValueError("intel yaml 'sources' must be a list")
    rows: list[SourceRow] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                SourceRow(
                    source_id=str(item["source_id"]),
                    source_name=str(item["source_name"]),
                    source_type=str(item["source_type"]),
                    kill_switch=bool(item["kill_switch"]),
                    rate_limit_per_day=item.get("rate_limit_per_day"),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        except KeyError as exc:
            logger.warning("intel.registry: yaml row missing key %r", exc)
            continue
    return rows


def _row_to_insert(row: SourceRow, *, now_iso: str) -> dict[str, Any]:
    return {
        "source_id": row.source_id,
        "source_name": row.source_name,
        "source_type": row.source_type,
        "kill_switch": row.kill_switch,
        "rate_limit_per_day": row.rate_limit_per_day,
        "last_scanned_at": None,
        "created_at": now_iso,
        "updated_at": now_iso,
        "metadata": row.metadata or {},
    }


def upsert_sources(
    rows: list[SourceRow],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Insert rows into `intel_sources`. Returns count inserted.

    Fail-open: returns 0 on any BQ error, never raises. Callers are responsible
    for not double-loading identical source_ids (BQ streaming has no UPSERT).
    """
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_client(proj)
    if client is None:
        return 0
    try:
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        payload = [_row_to_insert(r, now_iso=now_iso) for r in rows]
        table_ref = f"{proj}.{ds}.{_INTEL_SOURCES_TABLE}" if proj else f"{ds}.{_INTEL_SOURCES_TABLE}"
        errors = client.insert_rows_json(table_ref, payload)
        if errors:
            logger.warning("intel.registry insert errors: %s", errors)
            return 0
        return len(payload)
    except Exception as exc:
        logger.warning("intel.registry upsert fail-open: %r", exc)
        return 0


def load_active_sources(
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> list[SourceRow]:
    """Query `intel_sources WHERE kill_switch = false`.

    Fail-open: returns [] on any BQ error, never raises.
    """
    proj, ds = _resolve_target(project, dataset)
    client = _get_client(proj)
    if client is None:
        return []
    try:
        table_ref = f"{proj}.{ds}.{_INTEL_SOURCES_TABLE}" if proj else f"{ds}.{_INTEL_SOURCES_TABLE}"
        sql = (
            "SELECT source_id, source_name, source_type, kill_switch, "
            "rate_limit_per_day, metadata "
            f"FROM `{table_ref}` WHERE kill_switch = FALSE"
        )
        job = client.query(sql)
        result = job.result(timeout=30)
        out: list[SourceRow] = []
        for r in result:
            out.append(
                SourceRow(
                    source_id=r.source_id,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    kill_switch=bool(r.kill_switch),
                    rate_limit_per_day=r.rate_limit_per_day,
                    metadata=dict(r.metadata or {}),
                )
            )
        return out
    except Exception as exc:
        logger.warning("intel.registry load_active fail-open: %r", exc)
        return []
