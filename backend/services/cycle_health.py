"""
cycle_health -- Per-cycle timing log + process-level heartbeat for the
signal-freshness / cycle-health strip (4.5.8).

Design: two artifacts, two planes.

  Data plane (BQ): `MAX(event_time)` queries on paper_trades / paper_snapshots
                   -> per-source last_tick_age_sec.
  Control plane (process): handoff/.cycle_heartbeat.json updated by
                   autonomous_loop at cycle start + end. The freshness endpoint
                   surfaces BOTH -- so a dead emitter is visible even when BQ
                   has not had time to age out (Prometheus-style dead-man's-
                   switch guard; see RESEARCH.md 4.5.8 anti-pattern section).

History: append-only JSONL at handoff/cycle_history.jsonl, one row per cycle
with Oracle Analytics Cloud-style fields (cycle_id, started_at, completed_at,
duration_ms, status, error_count, n_trades, data_source_ages, bq_ingest_lag).
"""

from __future__ import annotations

import json

from backend.utils import json_io
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HANDOFF = Path(__file__).resolve().parents[2] / "handoff"
_HANDOFF.mkdir(parents=True, exist_ok=True)
_HISTORY_PATH = _HANDOFF / "cycle_history.jsonl"
_HEARTBEAT_PATH = _HANDOFF / ".cycle_heartbeat.json"

# Two-tier watchdog thresholds per Memfault + OneUptime + Industrial Monitor
# Direct: warn@1.5x, critical@2x expected cycle interval.
WARN_RATIO = 1.5
CRITICAL_RATIO = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _band(age_sec: Optional[float], interval_sec: float) -> str:
    if age_sec is None or interval_sec <= 0:
        return "unknown"
    ratio = age_sec / interval_sec
    if ratio >= CRITICAL_RATIO:
        return "red"
    if ratio >= WARN_RATIO:
        return "amber"
    return "green"


class CycleHealthLog:
    """Thread-safe JSONL writer + tail reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def record_cycle_start(self, cycle_id: str) -> str:
        started_at = _now_iso()
        self._write_heartbeat(cycle_id, "start")
        return started_at

    def record_cycle_end(
        self,
        cycle_id: str,
        started_at: str,
        status: str,
        n_trades: int = 0,
        error_count: int = 0,
        data_source_ages: Optional[dict] = None,
        bq_ingest_lag_sec: Optional[int] = None,
    ) -> None:
        completed_at = _now_iso()
        dur_ms: Optional[int] = None
        st = _parse_iso(started_at)
        ed = _parse_iso(completed_at)
        if st and ed:
            dur_ms = int((ed - st).total_seconds() * 1000)
        row = {
            "cycle_id": cycle_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": dur_ms,
            "status": status,
            "n_trades": int(n_trades),
            "error_count": int(error_count),
            "data_source_ages": data_source_ages or {},
            "bq_ingest_lag_sec": bq_ingest_lag_sec,
        }
        with self._lock:
            try:
                with _HISTORY_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                logger.warning(f"cycle_history write failed: {e}")
        self._write_heartbeat(cycle_id, "end")

    def last_cycles(self, n: int = 10) -> list[dict]:
        if not _HISTORY_PATH.exists():
            return []
        try:
            # Tail-read: keep memory bounded on large files.
            with _HISTORY_PATH.open(encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"cycle_history read failed: {e}")
            return []
        out: list[dict] = []
        for line in lines[-max(n, 1):][::-1]:  # newest first
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json_io.parse_json_line(line))
            except Exception:
                continue
        return out

    def _write_heartbeat(self, cycle_id: str, event: str) -> None:
        payload = {"cycle_id": cycle_id, "event": event, "updated_at": _now_iso()}
        try:
            _HEARTBEAT_PATH.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as e:
            logger.warning(f"heartbeat write failed: {e}")

    def read_heartbeat(self) -> Optional[dict]:
        if not _HEARTBEAT_PATH.exists():
            return None
        try:
            return json_io.load_json_file(_HEARTBEAT_PATH)
        except Exception:
            return None


_log = CycleHealthLog()


def get_log() -> CycleHealthLog:
    return _log


# ── Freshness computation ──────────────────────────────────────────


def _bq_max_event_age(bq: Any, table_logical: str, time_col: str) -> Optional[float]:
    """
    Method 1 per Metaplane: SELECT MAX(time_col) FROM table. Returns age in
    seconds, or None if the query fails / table empty. Uses the BQ client's
    existing _pt_table() helper for table name resolution.
    """
    try:
        table = bq._pt_table(table_logical)
        sql = f"SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX({time_col}), SECOND) AS age FROM `{table}`"
        rows = list(bq.client.query(sql).result())
        if not rows:
            return None
        age = rows[0].get("age") if hasattr(rows[0], "get") else rows[0][0]
        return float(age) if age is not None else None
    except Exception as e:
        logger.debug(f"bq_max_event_age({table_logical}.{time_col}) failed: {e}")
        return None


def compute_freshness(bq: Any, cycle_interval_sec: float) -> dict:
    """
    Build the freshness strip payload. Reads:
      - data plane: per-source MAX(time_col) via BQ
      - control plane: .cycle_heartbeat.json age
      - BQ ingest lag: age of the most recent trade row (proxy)
    """
    heartbeat = _log.read_heartbeat() or {}
    hb_updated = _parse_iso(heartbeat.get("updated_at"))
    hb_age_sec: Optional[float] = None
    if hb_updated:
        hb_age_sec = (datetime.now(timezone.utc) - hb_updated).total_seconds()

    trade_age = _bq_max_event_age(bq, "paper_trades", "created_at")
    snap_age = _bq_max_event_age(bq, "paper_portfolio_snapshots", "snapshot_date")
    bq_ingest_lag = trade_age  # trades are the write-hottest table

    sources = {
        "paper_trades": {
            "last_tick_age_sec": trade_age,
            "ratio": (trade_age / cycle_interval_sec) if (trade_age and cycle_interval_sec) else None,
            "band": _band(trade_age, cycle_interval_sec),
        },
        "paper_snapshots": {
            "last_tick_age_sec": snap_age,
            "ratio": (snap_age / cycle_interval_sec) if (snap_age and cycle_interval_sec) else None,
            "band": _band(snap_age, cycle_interval_sec),
        },
    }

    return {
        "sources": sources,
        "heartbeat": {
            "updated_at": heartbeat.get("updated_at"),
            "event": heartbeat.get("event"),
            "cycle_id": heartbeat.get("cycle_id"),
            "age_sec": round(hb_age_sec, 1) if hb_age_sec is not None else None,
            "ratio": (hb_age_sec / cycle_interval_sec) if (hb_age_sec and cycle_interval_sec) else None,
            "band": _band(hb_age_sec, cycle_interval_sec),
        },
        "bq_ingest_lag_sec": bq_ingest_lag,
        "thresholds": {
            "warn_ratio": WARN_RATIO,
            "critical_ratio": CRITICAL_RATIO,
            "cycle_interval_sec": cycle_interval_sec,
        },
        "computed_at": _now_iso(),
    }
