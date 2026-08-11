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
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_HANDOFF = Path(__file__).resolve().parents[2] / "handoff"
_HANDOFF.mkdir(parents=True, exist_ok=True)
_HISTORY_PATH = _HANDOFF / "cycle_history.jsonl"
_HEARTBEAT_PATH = _HANDOFF / ".cycle_heartbeat.json"

# Two-tier watchdog thresholds per Memfault + OneUptime + Industrial Monitor
# Direct: warn@1.5x, critical@2x expected cycle interval.
WARN_RATIO = 1.5
CRITICAL_RATIO = 2.0

# phase-25.A7: per-table expected maximum age (seconds). Historical tables
# have wildly different cadences than the per-cycle paper tables, so a
# single shared cycle_interval is the wrong yardstick. signals_log and
# paper_* still use the caller-provided cycle_interval_sec.
_TABLE_MAX_AGE_SEC: dict[str, float] = {
    "historical_prices":       93_600.0,     # 26h -- nightly ingest, T+1 US market
    "historical_fundamentals": 8_208_000.0,  # 95 days -- quarterly + filing lag
    "historical_macro":        3_024_000.0,  # 35 days -- monthly FRED + release lag
    "paper_portfolio_snapshots": 93_600.0,    # 26h -- daily snapshot
}

# phase-30.1: out-of-band autonomous-cycle heartbeat threshold. The paper
# trading cron fires Mon-Fri once per day at settings.paper_trading_hour ET
# (default 10 ET, 14-15 UTC depending on DST). Consecutive weekday cycles
# are 24h apart; 26h gives a 2h buffer for cron jitter or backend restart.
# Detected silent-failure incident: 65h 34m gap 2026-05-17 -> 2026-05-19
# documented in handoff/archive/phase-30.0/experiment_results.md Anomaly C.
_CYCLE_HEARTBEAT_STALE_SEC: float = 93_600.0  # 26h
_NYSE_TZ = ZoneInfo("America/New_York")

# phase-85.4 (criterion 4): the SECOND staleness clock -- age since the last
# cycle that actually reached status "completed".
#
# Why a second clock is needed. `_CYCLE_HEARTBEAT_STALE_SEC` above is measured
# against the last TERMINAL row of any status. `record_cycle_end` stamps
# `completed_at` on every terminal row, including `timeout` and `error`, so a
# cycle that times out every single day resets the heartbeat age to ~0 daily
# and `cycle_heartbeat_alarm` never goes stale. That is exactly what happened
# between 2026-07-31 and 2026-08-08: the last COMPLETED cycle was 7 days old,
# the loop was writing fresh `timeout` rows the whole time, and the only way
# to discover it was to hand-read handoff/cycle_history.jsonl.
#
# Threshold rationale. The cron fires Mon-Fri, so the largest legitimate gap
# between consecutive COMPLETED cycles is Fri -> Mon = 72h. 96h (4 days) sits
# one full weekday beyond that: a single failed Monday does not page, a second
# consecutive failed weekday does. Weekday-gated like the heartbeat alarm.
_CYCLE_COMPLETED_STALE_SEC: float = 345_600.0  # 96h = 4 days

# Statuses that mean "the cycle ran the whole way through". Everything else --
# timeout, error, budget_breach, halted_kill_switch, running -- is a
# non-completion for the purposes of the completed-age clock.
_COMPLETED_STATUSES: frozenset[str] = frozenset({"completed"})


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


def _worst_band(bands: list[str]) -> str:
    """phase-25.A7: aggregate a list of band strings to the worst one.

    Priority: red > amber > green > unknown. Empty list returns "unknown".
    """
    order = {"red": 3, "amber": 2, "green": 1, "unknown": 0}
    if not bands:
        return "unknown"
    return max(bands, key=lambda b: order.get(b, 0))


def _fire_freshness_alarm(sources: dict) -> None:
    """phase-25.A7: dispatch a P1 Slack alert for every table in red band.

    Dedup is handled by `AlertDeduper` inside `raise_cron_alert_sync` so a
    polling-loop caller doesn't spam Slack with the same alert. Per-call
    try/except keeps the alarm fail-open -- a Slack failure must never
    break the freshness query.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync
    except Exception as exc:
        logger.warning("freshness alarm: import fail-open: %r", exc)
        return
    for table_name, info in sources.items():
        if info.get("band") != "red":
            continue
        try:
            ratio = info.get("ratio")
            ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "N/A"
            raise_cron_alert_sync(
                source="cycle_health",
                error_type=f"freshness_critical_{table_name}",
                severity="P1",
                title=f"Data freshness critical: {table_name}",
                details={
                    "table": table_name,
                    "last_tick_age_sec": str(info.get("last_tick_age_sec")),
                    "interval_sec": str(info.get("interval_sec")),
                    "ratio": ratio_str,
                    "band": info.get("band"),
                },
            )
        except Exception as exc:
            logger.warning(
                "freshness alarm: dispatch fail-open for %s: %r", table_name, exc,
            )


def _now_utc() -> datetime:
    """phase-30.1: monkeypatch seam for tests. Production returns the live
    UTC clock; test code can override via monkeypatch on this symbol."""
    return datetime.now(timezone.utc)


def cycle_heartbeat_alarm(
    threshold_sec: float = _CYCLE_HEARTBEAT_STALE_SEC,
    completed_threshold_sec: float = _CYCLE_COMPLETED_STALE_SEC,
) -> dict:
    """phase-30.1: out-of-band autonomous-cycle staleness check.

    Reads the most recent `cycle_history.jsonl` row and returns
    `{"stale": bool, "age_sec": float|None, "should_alarm": bool,
      "is_weekday_et": bool, "last_completed_at": str|None}`.

    phase-85.4 (criterion 4) adds a SECOND, independent clock to the same
    verdict dict -- `last_success_at`, `success_age_sec`, `success_stale`,
    `should_alarm_success`, `last_terminal_status` -- measuring the age of
    the last row whose status is a real completion.

    The two clocks answer different questions and the difference is the whole
    point of the phase-85.4 defect:

      age_sec         -- "when did a cycle last END?"    (any terminal status)
      success_age_sec -- "when did a cycle last WORK?"   (status == completed)

    A cycle that times out daily keeps `age_sec` near zero forever while
    `success_age_sec` grows without bound. Between 2026-07-31 and 2026-08-08
    the first clock read ~24h every day and the second read 7 days, and only
    the second was true. Existing keys are unchanged and existing callers are
    unaffected; the success leg is purely additive.

    The function is pure: it returns a verdict dict and does NOT post
    to Slack. The caller (watchdog cron in `slack_bot/scheduler.py`)
    does the state-transition gating so we don't spam the channel
    during a multi-day silent-failure (the documented `_watchdog_last_was_healthy`
    pattern at `slack_bot/scheduler.py:97-101`).

    Decision logic:
    - `stale` is True iff age_sec > threshold_sec.
    - `should_alarm` is True iff `stale AND is_weekday_et`. Weekends
      have no scheduled cron fire so a 26h stale on Sat/Sun is normal.
    - When `cycle_history.jsonl` is missing or empty, returns
      `stale=False, should_alarm=False` (no historical signal to flag
      against; first-boot is not a silent-failure).

    Fail-open: any exception is caught and returns a sentinel dict
    rather than raising, so the watchdog cron can't be brought down
    by an unexpected error here.

    Audit basis: handoff/archive/phase-30.0/experiment_results.md
    Anomaly C (65h 34m gap 2026-05-17 -> 2026-05-19 with no
    out-of-band alert path).
    """
    sentinel = {
        "stale": False,
        "age_sec": None,
        "should_alarm": False,
        "is_weekday_et": False,
        "last_completed_at": None,
        # phase-85.4 (criterion 4): the completed-age leg. Always present so
        # callers never have to .get() defensively.
        "last_success_at": None,
        "success_age_sec": None,
        "success_stale": False,
        "should_alarm_success": False,
        "last_terminal_status": None,
    }
    try:
        if not _HISTORY_PATH.exists():
            return sentinel
        with _HISTORY_PATH.open(encoding="utf-8") as f:
            lines = f.readlines()
        # phase-38.2: skip "started" rows -- they have completed_at=null and
        # would otherwise short-circuit to sentinel, suppressing the alarm
        # on a halted cycle (the exact lost-cycle-3a failure mode).
        #
        # phase-85.4: the SAME reverse scan now also finds the most recent row
        # whose status is a real completion. Two independent clocks, one pass.
        last_row: Optional[dict] = None
        last_success_row: Optional[dict] = None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json_io.parse_json_line(line)
            except Exception:
                continue
            if parsed.get("status") == "started":
                continue
            if last_row is None:
                last_row = parsed
            if last_success_row is None and parsed.get("status") in _COMPLETED_STATUSES:
                last_success_row = parsed
            if last_row is not None and last_success_row is not None:
                break
        if not last_row:
            return sentinel
        now = _now_utc()
        is_weekday_et = now.astimezone(_NYSE_TZ).weekday() < 5

        # ── Leg 2 (phase-85.4): age since the last COMPLETED cycle ──────────
        # Computed first so it survives the early return below when the latest
        # terminal row has an unparseable completed_at.
        success_at = (last_success_row or {}).get("completed_at")
        success_dt = _parse_iso(success_at)
        success_age_sec: Optional[float] = None
        success_stale = False
        if success_dt is not None:
            success_age_sec = (now - success_dt).total_seconds()
            success_stale = success_age_sec > completed_threshold_sec
        elif last_success_row is not None:
            # phase-85.4 cycle-2 (Q/A finding, EVALUATE pass 1): a completed row
            # EXISTS but its completed_at will not parse. The first cut left
            # success_stale False here, which is a fail-OPEN hole in an alarm
            # whose entire job is loudness -- one corrupt timestamp and the
            # "book is not trading" P1 goes silent indefinitely.
            #
            # An unevaluable clock is not a healthy clock. Age stays None
            # (honest: it is unknown), but staleness is TRUE so the operator is
            # told, and the P1 renders the age as "never" rather than a number
            # it cannot support.
            success_stale = True
        elif last_success_row is None and lines:
            # There ARE terminal rows but NONE of them completed. That is the
            # worst case, not the benign first-boot case -- treat it as stale
            # so a ledger of nothing-but-timeouts pages instead of going quiet.
            success_stale = last_row is not None
        success_leg = {
            "last_success_at": success_at,
            "success_age_sec": success_age_sec,
            "success_stale": success_stale,
            "should_alarm_success": success_stale and is_weekday_et,
            "last_terminal_status": last_row.get("status"),
        }

        completed_at = last_row.get("completed_at")
        completed_dt = _parse_iso(completed_at)
        if completed_dt is None:
            return {**sentinel, **success_leg, "is_weekday_et": is_weekday_et}
        age_sec = (now - completed_dt).total_seconds()
        stale = age_sec > threshold_sec
        return {
            "stale": stale,
            "age_sec": age_sec,
            "should_alarm": stale and is_weekday_et,
            "is_weekday_et": is_weekday_et,
            "last_completed_at": completed_at,
            **success_leg,
        }
    except Exception as exc:
        logger.warning("cycle_heartbeat_alarm fail-open: %r", exc)
        return sentinel


def fire_cycle_heartbeat_alarm(verdict: dict) -> None:
    """phase-30.1: dispatch the P1 Slack alert when the heartbeat alarm
    has decided to fire. Called by the watchdog cron AFTER its own
    state-transition gating so duplicates are not posted.

    Identical fail-open pattern to `_fire_freshness_alarm:90-125`:
    a Slack failure must NEVER break the calling cron.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync
    except Exception as exc:
        logger.warning("cycle_heartbeat_alarm: import fail-open: %r", exc)
        return
    try:
        age_sec = verdict.get("age_sec")
        age_h_str = f"{(age_sec/3600):.1f}h" if isinstance(age_sec, (int, float)) else "N/A"
        raise_cron_alert_sync(
            source="cycle_health",
            error_type="cycle_heartbeat_stale_weekday",
            severity="P1",
            title="Autonomous cycle silent: heartbeat stale on weekday",
            details={
                "last_completed_at": str(verdict.get("last_completed_at")),
                "age_sec": str(age_sec),
                "age_hours_approx": age_h_str,
                "threshold_sec": str(_CYCLE_HEARTBEAT_STALE_SEC),
                "is_weekday_et": str(verdict.get("is_weekday_et")),
            },
        )
    except Exception as exc:
        logger.warning("cycle_heartbeat_alarm: dispatch fail-open: %r", exc)


def fire_cycle_completed_stale_alarm(verdict: dict) -> None:
    """phase-85.4 (criterion 4): dispatch the P1 for "no cycle has COMPLETED
    in N days", as distinct from "no cycle has ENDED in N hours".

    Called by the watchdog cron AFTER its own state-transition gating, exactly
    like `fire_cycle_heartbeat_alarm`. Distinct `error_type` so the two alarms
    never collide in the alert dedup layer -- they can and should be able to
    fire independently (daily timeouts = heartbeat quiet, completed-age loud).

    Identical fail-open pattern: a Slack failure must NEVER break the cron.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync
    except Exception as exc:
        logger.warning("cycle_completed_stale_alarm: import fail-open: %r", exc)
        return
    try:
        age_sec = verdict.get("success_age_sec")
        if isinstance(age_sec, (int, float)):
            age_str = f"{(age_sec / 86400):.1f}d"
        else:
            age_str = "never (no completed cycle in the ledger)"
        raise_cron_alert_sync(
            source="cycle_health",
            error_type="cycle_completed_stale_weekday",
            severity="P1",
            title="Autonomous cycle has not COMPLETED -- book is not trading",
            details={
                "last_completed_cycle_at": str(verdict.get("last_success_at")),
                "age_since_last_completion": age_str,
                "age_sec": str(age_sec),
                "threshold_sec": str(_CYCLE_COMPLETED_STALE_SEC),
                "last_terminal_status": str(verdict.get("last_terminal_status")),
                "note": (
                    "Cycles may still be ENDING on schedule -- this alarm measures "
                    "the age of the last cycle that ended in status 'completed'. "
                    "A cycle that times out daily keeps the heartbeat alarm quiet."
                ),
            },
        )
    except Exception as exc:
        logger.warning("cycle_completed_stale_alarm: dispatch fail-open: %r", exc)


class CycleHealthLog:
    """Thread-safe JSONL writer + tail reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def record_cycle_start(self, cycle_id: str) -> str:
        # phase-38.2 (OPEN-11): write a "started" row to cycle_history.jsonl
        # immediately so a halted/SIGKILLd cycle leaves an audit trace.
        # Design (a) per research_brief_phase_38_2.md: append-then-append --
        # one started row at start, one terminal row at end, joined by
        # cycle_id. POSIX O_APPEND atomicity (man write(2)) + the existing
        # threading.Lock guarantee row-boundary integrity.
        started_at = _now_iso()
        row = {
            "cycle_id": cycle_id,
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "status": "started",
            "n_trades": 0,
            "error_count": 0,
            "data_source_ages": {},
            "bq_ingest_lag_sec": None,
        }
        with self._lock:
            try:
                with _HISTORY_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                logger.warning(f"cycle_history start-row write failed: {e}")
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
        meta_scorer_degraded: bool = False,
        rail_skipped: bool = False,
        breaker_tripped: bool = False,
        funnel: Optional[dict] = None,
        degradation: Optional[dict] = None,
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
            # phase-60.4 (criterion 4): persisted so the morning digest can
            # surface "conviction values were no-LLM fallbacks" instead of
            # letting conviction 10.00 masquerade as an LLM judgment.
            "meta_scorer_degraded": bool(meta_scorer_degraded),
            # phase-66.1: rail-guard outcome -- lets the 66.2 funnel separate
            # "Claude rail skipped/tripped" from "gates rejected candidates".
            "rail_skipped": bool(rail_skipped),
            "breaker_tripped": bool(breaker_tripped),
            # phase-66.2: per-stage funnel counts (universe/screened/candidates/
            # new_to_analyze/reeval) -- previously summary-only (log-parse to
            # recover); persisting them makes criterion-b diagnosis durable.
            "funnel": funnel or {},
            # phase-86.38: per-cycle DEGRADATION record, kept separate from the
            # 66.2 funnel because they answer different questions -- the funnel
            # is "how many candidates survived each stage", this is "was the
            # pipeline that judged them the real one". Populated on EVERY cycle,
            # including cycles whose fallback rate is below the paging
            # threshold: a degraded cycle that does not page is exactly the case
            # that previously left no durable trace (measured 2026-08-10:
            # 3 of the 6 TICKERS analysed landed on the lite fallback and no
            # page fired; the alarm's own denominator was NOT measured, so the
            # exact ratio it saw is unestablished).
            # Additive free-form dict -- cycle_history is append-only JSONL, so
            # readers that do not know the key simply ignore it.
            "degradation": degradation or {},
        }
        with self._lock:
            try:
                with _HISTORY_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                logger.warning(f"cycle_history write failed: {e}")
        self._write_heartbeat(cycle_id, "end")

    def last_cycles(self, n: int = 10, include_started: bool = False) -> list[dict]:
        # phase-38.2: by default skip "started" rows so existing callers
        # (UI tabs, alarm) only see terminal rows. Set include_started=True
        # to surface them for orphan audit (see orphan_rows()).
        if not _HISTORY_PATH.exists():
            return []
        try:
            with _HISTORY_PATH.open(encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"cycle_history read failed: {e}")
            return []
        out: list[dict] = []
        for line in reversed(lines):  # newest first
            line = line.strip()
            if not line:
                continue
            try:
                row = json_io.parse_json_line(line)
            except Exception:
                continue
            if not include_started and row.get("status") == "started":
                continue
            out.append(row)
            if len(out) >= max(n, 1):
                break
        return out

    def orphan_rows(self) -> list[dict]:
        # phase-38.2 (OPEN-11 criterion 3): return started rows whose cycle_id
        # has NO matching terminal row. Detects halted / SIGKILLd cycles.
        if not _HISTORY_PATH.exists():
            return []
        try:
            with _HISTORY_PATH.open(encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"cycle_history orphan_rows read failed: {e}")
            return []
        starts: dict[str, dict] = {}
        terminated: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json_io.parse_json_line(line)
            except Exception:
                continue
            cid = row.get("cycle_id")
            if not cid:
                continue
            status = row.get("status")
            if status == "started":
                # Keep only the FIRST started row per cycle_id (deterministic).
                starts.setdefault(cid, row)
            else:
                # Any non-started status (completed/failed/etc.) terminates.
                terminated.add(cid)
        return [row for cid, row in starts.items() if cid not in terminated]

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


# phase-43.0 cycle-14: STRING/DATE-typed timestamp columns that require
# SAFE.TIMESTAMP(MAX(col)) coercion to TIMESTAMP before TIMESTAMP_DIFF.
# All other call sites use bare MAX(col); applying SAFE.TIMESTAMP to a
# native-TIMESTAMP MAX result raises `SAFE with function timestamp is
# not supported` because (a) TIMESTAMP() has no (TIMESTAMP)->TIMESTAMP
# overload and (b) SAFE. prefix is not supported with aggregates.
_STRING_DATE_TIMESTAMP_COLS = {
    ("paper_trades", "created_at"),                  # STRING (RFC3339)
    ("paper_portfolio_snapshots", "snapshot_date"),  # STRING (YYYY-MM-DD)
}


def _bq_max_event_age(bq: Any, table_logical: str, time_col: str) -> Optional[float]:
    """
    Method 1 per Metaplane: SELECT MAX(time_col) FROM table. Returns age in
    seconds, or None if the query fails / table empty. Uses the BQ client's
    existing _pt_table() helper for table name resolution.

    phase-23.2.20 introduced `SAFE.TIMESTAMP(MAX({time_col}))` to coerce
    STRING-typed timestamp columns (paper_trades.created_at as RFC3339,
    paper_portfolio_snapshots.snapshot_date as bare YYYY-MM-DD) to TIMESTAMP.
    phase-43.0 cycle-14 fix: the SAFE.TIMESTAMP wrapper is REQUIRED for
    STRING/DATE columns but BREAKS for already-TIMESTAMP-typed columns
    (historical_prices.ingested_at, historical_fundamentals.ingested_at,
    historical_macro.ingested_at, signals_log.recorded_at). On a native
    TIMESTAMP column, `SAFE.TIMESTAMP(MAX(...))` returns BQ 400 BadRequest
    `SAFE with function timestamp is not supported` -- the broad except
    swallowed it, returning None, and the band stayed "unknown" indefinitely
    (DoD-5 cycle-12 FAIL). The fix: branch on column type via
    `_STRING_DATE_TIMESTAMP_COLS` -- known STRING/DATE columns still use
    SAFE.TIMESTAMP; everything else uses bare MAX(time_col).
    """
    try:
        table = bq._pt_table(table_logical)
        needs_coerce = (table_logical, time_col) in _STRING_DATE_TIMESTAMP_COLS
        max_expr = (
            f"SAFE.TIMESTAMP(MAX({time_col}))" if needs_coerce
            else f"MAX({time_col})"
        )
        sql = (
            f"SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), "
            f"{max_expr}, SECOND) AS age "
            f"FROM `{table}`"
        )
        rows = list(bq.client.query(sql).result(timeout=30))
        if not rows:
            return None
        age = rows[0].get("age") if hasattr(rows[0], "get") else rows[0][0]
        return float(age) if age is not None else None
    except Exception as e:
        # phase-23.2.20: was logger.debug -- silent at default INFO level.
        # Bumped to WARNING so future schema regressions surface in the
        # backend log without operators having to enable DEBUG.
        logger.warning(
            "bq_max_event_age(%s.%s) failed: %s", table_logical, time_col, e
        )
        return None


def compute_freshness(
    bq: Any, cycle_interval_sec: float, *, emit_alarm: bool = True
) -> dict:
    """
    Build the freshness strip payload. Reads:
      - data plane: per-source MAX(time_col) via BQ
      - control plane: .cycle_heartbeat.json age
      - BQ ingest lag: age of the most recent trade row (proxy)

    phase-82.10: `emit_alarm` is keyword-only and defaults to True, so the
    three HTTP call sites (paper_trading.py, observability_api.py x2) are
    behaviourally unchanged. The scheduled evaluator in
    `backend/services/freshness_cron.py` passes `emit_alarm=False` because it
    owns a STATE-TRANSITION gate that the level-triggered `_fire_freshness_alarm`
    path does not have. Without this opt-out a cron tick would page twice for
    the same red table (once from here, once from the gate), and -- worse --
    the inner path would page on EVERY tick for a permanently-red source: a
    P1 bypasses AlertDeduper's consecutive threshold but NOT its repeat
    window, so it re-fires every `alert_repeat_hours` forever. Measured
    2026-08-05: five back-to-back P1 should_fire calls give
    [True, False, False, False, False], but rewinding last_fired_at by 1h1m
    returns True again.
    """
    heartbeat = _log.read_heartbeat() or {}
    hb_updated = _parse_iso(heartbeat.get("updated_at"))
    hb_age_sec: Optional[float] = None
    if hb_updated:
        hb_age_sec = (datetime.now(timezone.utc) - hb_updated).total_seconds()

    trade_age = _bq_max_event_age(bq, "paper_trades", "created_at")
    snap_age = _bq_max_event_age(bq, "paper_portfolio_snapshots", "snapshot_date")
    # phase-25.A7: extend coverage to 4 historical/log tables.
    hist_prices_age = _bq_max_event_age(bq, "historical_prices", "ingested_at")
    hist_fund_age = _bq_max_event_age(bq, "historical_fundamentals", "ingested_at")
    hist_macro_age = _bq_max_event_age(bq, "historical_macro", "ingested_at")
    signals_age = _bq_max_event_age(bq, "signals_log", "recorded_at")
    bq_ingest_lag = trade_age  # trades are the write-hottest table

    # phase-25.A7: snapshot SLA -- distinct from per-cycle cadence (was using
    # cycle_interval_sec; now uses 26h per research finding).
    snap_interval = _TABLE_MAX_AGE_SEC["paper_portfolio_snapshots"]
    hist_prices_interval = _TABLE_MAX_AGE_SEC["historical_prices"]
    hist_fund_interval = _TABLE_MAX_AGE_SEC["historical_fundamentals"]
    hist_macro_interval = _TABLE_MAX_AGE_SEC["historical_macro"]

    sources = {
        "paper_trades": {
            "last_tick_age_sec": trade_age,
            "interval_sec": cycle_interval_sec,
            "ratio": (trade_age / cycle_interval_sec) if (trade_age and cycle_interval_sec) else None,
            "band": _band(trade_age, cycle_interval_sec),
        },
        # phase-25.A7: rename "paper_snapshots" -> "paper_portfolio_snapshots"
        # so the key matches the BQ table name exactly (operators grepping
        # for the table will find the freshness key).
        "paper_portfolio_snapshots": {
            "last_tick_age_sec": snap_age,
            "interval_sec": snap_interval,
            "ratio": (snap_age / snap_interval) if (snap_age is not None and snap_interval > 0) else None,
            "band": _band(snap_age, snap_interval),
        },
        "historical_prices": {
            "last_tick_age_sec": hist_prices_age,
            "interval_sec": hist_prices_interval,
            "ratio": (hist_prices_age / hist_prices_interval) if (hist_prices_age is not None and hist_prices_interval > 0) else None,
            "band": _band(hist_prices_age, hist_prices_interval),
        },
        "historical_fundamentals": {
            "last_tick_age_sec": hist_fund_age,
            "interval_sec": hist_fund_interval,
            "ratio": (hist_fund_age / hist_fund_interval) if (hist_fund_age is not None and hist_fund_interval > 0) else None,
            "band": _band(hist_fund_age, hist_fund_interval),
        },
        "historical_macro": {
            "last_tick_age_sec": hist_macro_age,
            "interval_sec": hist_macro_interval,
            "ratio": (hist_macro_age / hist_macro_interval) if (hist_macro_age is not None and hist_macro_interval > 0) else None,
            "band": _band(hist_macro_age, hist_macro_interval),
        },
        "signals_log": {
            "last_tick_age_sec": signals_age,
            "interval_sec": cycle_interval_sec,
            "ratio": (signals_age / cycle_interval_sec) if (signals_age and cycle_interval_sec) else None,
            "band": _band(signals_age, cycle_interval_sec),
        },
    }

    # phase-25.A7: aggregate worst band across all monitored tables.
    overall_band = _worst_band([v.get("band", "unknown") for v in sources.values()])

    # Fire a P1 Slack alert per table in critical band (dedup via AlertDeduper).
    # phase-82.10: suppressible -- see the `emit_alarm` note in the docstring.
    if emit_alarm and overall_band == "red":
        _fire_freshness_alarm(sources)

    return {
        "sources": sources,
        "overall_band": overall_band,
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
