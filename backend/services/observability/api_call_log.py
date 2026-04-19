"""phase-6.7 buffered writer for the `api_call_log` BQ table.

Research (brief 2026-04-19):
- AI-telemetry volume 10-50x general API volume -> keep `llm_call_log`
  and `api_call_log` as separate tables (OneUptime 2026 guidance)
- Buffered writer batches rows to minimise BQ insert cost; flush every 60s
  or 100 rows, whichever first

Schema (DDL in `scripts/migrations/add_api_call_log.py`):
    ts TIMESTAMP NOT NULL
    source STRING NOT NULL           (finnhub | benzinga | alpaca | fred | ...)
    endpoint STRING                  (path or full URL; caller decides)
    http_status INT64
    latency_ms FLOAT64
    response_bytes INT64
    cost_usd_est FLOAT64             (0.0 for free tier)
    ok BOOL                          (http_status < 400)
    error_kind STRING                (Timeout | HTTPError | RateLimited | ConnError | None)
    request_id STRING                (if server returns one)

Fail-open:
- BQ import absent -> warn once, discard rows silently
- BQ insert exception -> log WARNING, rows already buffered are dropped
- Never raises out of `log_api_call()`
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


_FLUSH_ROWS = 100
_FLUSH_SECONDS = 60
_TABLE_NAME = "api_call_log"
_WARNED_BQ_ABSENT = False


@dataclass
class _ApiCallRow:
    ts: str
    source: str
    endpoint: str
    http_status: int | None
    latency_ms: float
    response_bytes: int
    cost_usd_est: float
    ok: bool
    error_kind: str | None
    request_id: str | None


_buffer: list[_ApiCallRow] = []
_last_flush_ts: datetime = datetime.now(timezone.utc)
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_api_call(
    source: str,
    endpoint: str = "",
    http_status: int | None = None,
    latency_ms: float = 0.0,
    response_bytes: int = 0,
    cost_usd_est: float = 0.0,
    ok: bool = True,
    error_kind: str | None = None,
    request_id: str | None = None,
) -> None:
    """Buffer a row; flushes if size or time threshold crossed. Never raises."""
    try:
        row = _ApiCallRow(
            ts=_now_iso(),
            source=source,
            endpoint=endpoint,
            http_status=http_status,
            latency_ms=float(latency_ms),
            response_bytes=int(response_bytes),
            cost_usd_est=float(cost_usd_est),
            ok=bool(ok),
            error_kind=error_kind,
            request_id=request_id,
        )
        with _lock:
            _buffer.append(row)
            should_flush = _should_flush_locked()
        if should_flush:
            flush()
    except Exception as exc:  # pragma: no cover
        logger.debug("log_api_call fail-open err=%r", exc)


def _should_flush_locked() -> bool:
    if len(_buffer) >= _FLUSH_ROWS:
        return True
    if (datetime.now(timezone.utc) - _last_flush_ts).total_seconds() >= _FLUSH_SECONDS:
        return True
    return False


def flush() -> int:
    """Flush buffered rows to BQ. Returns rows flushed (0 on any failure). Never raises."""
    global _last_flush_ts, _WARNED_BQ_ABSENT
    with _lock:
        rows = list(_buffer)
        _buffer.clear()
    if not rows:
        with _lock:
            _last_flush_ts = datetime.now(timezone.utc)
        return 0

    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        if not _WARNED_BQ_ABSENT:
            logger.warning(
                "api_call_log: google-cloud-bigquery absent (%r); "
                "dropping %d buffered rows",
                exc,
                len(rows),
            )
            _WARNED_BQ_ABSENT = True
        return 0

    try:
        from backend.config.settings import get_settings

        s = get_settings()
        project = s.gcp_project_id
        dataset = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        client = bigquery.Client(project=project)
        table_ref = f"{project}.{dataset}.{_TABLE_NAME}"
        dicts = [asdict(r) for r in rows]
        errors = client.insert_rows_json(table_ref, dicts)
        if errors:
            logger.warning("api_call_log BQ insert errors: %s", errors)
            return 0
    except Exception as exc:
        logger.warning("api_call_log flush fail-open err=%r", exc)
        return 0

    with _lock:
        _last_flush_ts = datetime.now(timezone.utc)
    logger.debug("api_call_log flushed %d rows", len(rows))
    return len(rows)


def buffer_size() -> int:
    """Test helper: observe buffer depth without forcing a flush."""
    with _lock:
        return len(_buffer)


def reset_buffer_for_test() -> None:
    """Test helper: drop buffered rows + reset warn-once latch."""
    global _WARNED_BQ_ABSENT
    with _lock:
        _buffer.clear()
    _WARNED_BQ_ABSENT = False


__all__ = [
    "log_api_call",
    "flush",
    "buffer_size",
    "reset_buffer_for_test",
    "log_llm_call",
    "flush_llm",
    "llm_buffer_size",
]


# ---------------------------------------------------------------------------
# llm_call_log writer -- retrofit for phase-4.14.23 gap.
#
# Schema (scripts/migrations/add_llm_call_log.py):
#   ts TIMESTAMP NOT NULL
#   provider STRING NOT NULL     (anthropic|gemini|openai|github_models)
#   model STRING NOT NULL
#   agent STRING                 (best-effort; optional)
#   latency_ms FLOAT64
#   ttft_ms FLOAT64
#   input_tok INT64
#   output_tok INT64
#   cache_creation_tok INT64     (optional, Claude only)
#   cache_read_tok INT64         (optional, Claude only)
#   request_id STRING
#   ok BOOL

_LLM_TABLE_NAME = "llm_call_log"
_llm_buffer: list[dict[str, Any]] = []
_llm_last_flush_ts: datetime = datetime.now(timezone.utc)
_llm_lock = threading.Lock()


def log_llm_call(
    provider: str,
    model: str,
    agent: str | None = None,
    latency_ms: float = 0.0,
    ttft_ms: float = 0.0,
    input_tok: int = 0,
    output_tok: int = 0,
    cache_creation_tok: int = 0,
    cache_read_tok: int = 0,
    request_id: str | None = None,
    ok: bool = True,
) -> None:
    """Buffer a llm_call_log row. Never raises."""
    try:
        row = {
            "ts": _now_iso(),
            "provider": provider,
            "model": model,
            "agent": agent,
            "latency_ms": float(latency_ms),
            "ttft_ms": float(ttft_ms),
            "input_tok": int(input_tok),
            "output_tok": int(output_tok),
            "cache_creation_tok": int(cache_creation_tok),
            "cache_read_tok": int(cache_read_tok),
            "request_id": request_id,
            "ok": bool(ok),
        }
        with _llm_lock:
            _llm_buffer.append(row)
            should_flush = (
                len(_llm_buffer) >= _FLUSH_ROWS
                or (datetime.now(timezone.utc) - _llm_last_flush_ts).total_seconds()
                >= _FLUSH_SECONDS
            )
        if should_flush:
            flush_llm()
    except Exception as exc:  # pragma: no cover
        logger.debug("log_llm_call fail-open err=%r", exc)


def flush_llm() -> int:
    """Flush llm_call_log buffer to BQ. Returns rows flushed (0 on failure). Never raises."""
    global _llm_last_flush_ts, _WARNED_BQ_ABSENT
    with _llm_lock:
        rows = list(_llm_buffer)
        _llm_buffer.clear()
    if not rows:
        with _llm_lock:
            _llm_last_flush_ts = datetime.now(timezone.utc)
        return 0

    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        if not _WARNED_BQ_ABSENT:
            logger.warning(
                "llm_call_log: google-cloud-bigquery absent (%r); "
                "dropping %d buffered rows",
                exc,
                len(rows),
            )
            _WARNED_BQ_ABSENT = True
        return 0

    try:
        from backend.config.settings import get_settings

        s = get_settings()
        project = s.gcp_project_id
        dataset = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        client = bigquery.Client(project=project)
        table_ref = f"{project}.{dataset}.{_LLM_TABLE_NAME}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("llm_call_log BQ insert errors: %s", errors)
            return 0
    except Exception as exc:
        logger.warning("llm_call_log flush fail-open err=%r", exc)
        return 0

    with _llm_lock:
        _llm_last_flush_ts = datetime.now(timezone.utc)
    logger.debug("llm_call_log flushed %d rows", len(rows))
    return len(rows)


def llm_buffer_size() -> int:
    with _llm_lock:
        return len(_llm_buffer)
