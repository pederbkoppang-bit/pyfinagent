"""phase-10.6 Monthly Champion/Challenger Sortino gate with HITL approval.

Fires on the last trading Friday of each month (NYSE calendar). Evaluates a
challenger strategy against the current champion on three quality gates:

  1. Sortino delta     challenger - champion >= 0.3
  2. PBO               challenger         <  0.2
  3. DD ratio          challenger / champion <= 1.2

If all three pass, opens a HITL approval window (48h). Approval state is
persisted as JSON at `handoff/logs/monthly_approval_state.json` (or
injectable path in tests). A stale (>48h) pending state auto-transitions to
`expired`.

**Hard invariant:** `actual_replacement` is always `False`. This module
never promotes real-capital strategies -- paper-only until an explicit
SR 11-7 compliance pass wires the real-capital layer.

Pure library: no live Slack posting (inject `slack_fn` for side effects).
"""
from __future__ import annotations

import calendar
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from backend.autoresearch.gate import PromotionGate
from backend.metrics.sortino import sortino

logger = logging.getLogger(__name__)

_APPROVAL_WINDOW_HOURS = 48
_DEFAULT_STATE_PATH = Path("handoff/logs/monthly_approval_state.json")
_DEFAULT_MIN_CHALLENGER_DAYS = 20
_DEFAULT_SORTINO_DELTA = 0.3
_DEFAULT_PBO_THRESHOLD = 0.2
_DEFAULT_DD_RATIO_THRESHOLD = 1.2


def run_monthly_sortino_gate(
    eval_date: date,
    *,
    champion_returns: list[float],
    challenger_returns: list[float],
    champion_max_dd: float,
    challenger_max_dd: float,
    challenger_pbo: float,
    challenger_id: str = "challenger",
    challenger_min_days: int = _DEFAULT_MIN_CHALLENGER_DAYS,
    sortino_delta_threshold: float = _DEFAULT_SORTINO_DELTA,
    pbo_threshold: float = _DEFAULT_PBO_THRESHOLD,
    dd_ratio_threshold: float = _DEFAULT_DD_RATIO_THRESHOLD,
    periods_per_year: int = 252,
    slack_fn: Callable[[str, dict[str, Any]], None] | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the monthly champion/challenger Sortino gate.

    `actual_replacement` is hard-coded False -- paper-only promotion.
    """
    spath = Path(state_path) if state_path is not None else _DEFAULT_STATE_PATH
    now_dt = now or datetime.now(timezone.utc)
    month_key = f"{eval_date.year:04d}-{eval_date.month:02d}"

    result: dict[str, Any] = {
        "fired": False,
        "gate_pass": False,
        "approval_pending": False,
        "approved": False,
        "expired": False,
        "actual_replacement": False,
        "reason": None,
        "sortino_delta": None,
        "dd_ratio": None,
        "pbo": float(challenger_pbo),
        "month": month_key,
    }

    # Criterion 1: fires_on_last_trading_friday_of_month
    if not is_last_trading_friday(eval_date):
        result["reason"] = "not_last_trading_friday"
        return result
    result["fired"] = True

    # Criterion 6 early-check: if a pending state from a prior month exists
    # and is past its expiry, transition it to `expired` and overwrite.
    state = _load_state(spath)
    prior = state.get(month_key)
    if prior is not None:
        prior_status = prior.get("status")
        prior_expires = _parse_iso(prior.get("expires_at_iso"))
        if prior_status == "pending":
            if prior_expires is not None and now_dt >= prior_expires:
                # Expire it.
                prior["status"] = "expired"
                state[month_key] = prior
                _save_state(state, spath)
                result["expired"] = True
                result["reason"] = "prior_pending_expired"
                # Do NOT return early -- continue evaluation to open a new pending if the gate passes.
            else:
                # Still pending; return the current state.
                result["approval_pending"] = True
                result["sortino_delta"] = prior.get("sortino_delta")
                result["dd_ratio"] = prior.get("dd_ratio")
                result["gate_pass"] = True
                result["reason"] = "prior_pending_not_expired"
                return result
        elif prior_status == "approved":
            result["approved"] = True
            result["gate_pass"] = True
            result["reason"] = "prior_approved"
            return result

    # Challenger min days check (short-circuits below-min days before quality gates).
    if len(challenger_returns) < challenger_min_days:
        result["reason"] = f"challenger_days<{challenger_min_days}"
        return result

    # Criterion 3: requires_sortino_delta_ge_0_3
    try:
        s_champ = sortino(
            champion_returns, mar=0.0, periods_per_year=periods_per_year
        )
        s_chall = sortino(
            challenger_returns, mar=0.0, periods_per_year=periods_per_year
        )
    except Exception as exc:
        result["reason"] = f"sortino_eval_error: {exc!r}"
        return result

    # NaN handling: if either Sortino is NaN (zero downside or < 2 samples),
    # treat as fail-closed for the delta check.
    import math as _m
    if _m.isnan(s_champ) or _m.isnan(s_chall):
        result["reason"] = "sortino_nan"
        return result

    delta = s_chall - s_champ
    result["sortino_delta"] = float(delta)
    if delta < sortino_delta_threshold:
        result["reason"] = f"sortino_delta<{sortino_delta_threshold}"
        return result

    # Criterion 4: requires_pbo_lt_0_2 (strict less-than).
    if float(challenger_pbo) >= pbo_threshold:
        result["reason"] = f"pbo>={pbo_threshold}"
        return result

    # Criterion 5: requires_dd_ratio_le_1_2 (challenger dd / champion dd).
    champ_dd = abs(float(champion_max_dd))
    chall_dd = abs(float(challenger_max_dd))
    if champ_dd <= 0.0:
        # No champion drawdown -> ratio is undefined. Fail-closed.
        result["reason"] = "champion_dd_zero"
        return result
    dd_ratio = chall_dd / champ_dd
    result["dd_ratio"] = float(dd_ratio)
    if dd_ratio > dd_ratio_threshold:
        result["reason"] = f"dd_ratio>{dd_ratio_threshold}"
        return result

    # All three gates passed. Open HITL window.
    result["gate_pass"] = True
    result["approval_pending"] = True
    result["reason"] = "opened_hitl_window"

    expires_at = now_dt + timedelta(hours=_APPROVAL_WINDOW_HOURS)
    state[month_key] = {
        "month": month_key,
        "created_at_iso": now_dt.isoformat(),
        "expires_at_iso": expires_at.isoformat(),
        "status": "pending",
        "sortino_delta": float(delta),
        "dd_ratio": float(dd_ratio),
        "pbo": float(challenger_pbo),
        "challenger_id": challenger_id,
    }
    _save_state(state, spath)

    if slack_fn is not None:
        try:
            slack_fn(
                f"Monthly Champion/Challenger gate fired for {month_key}. "
                f"Sortino delta {delta:.3f} (>= {sortino_delta_threshold}), "
                f"PBO {challenger_pbo:.3f} (< {pbo_threshold}), "
                f"DD ratio {dd_ratio:.3f} (<= {dd_ratio_threshold}). "
                f"Expires {expires_at.isoformat()}.",
                state[month_key],
            )
        except Exception as exc:
            logger.warning("monthly_gate: slack_fn fail-open: %r", exc)

    return result


def record_approval(
    month_key: str,
    *,
    status: str,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Transition a pending approval to approved / rejected.

    Returns the updated state row, or {} if not found / already terminal.
    `status` must be one of "approved" / "rejected".
    """
    if status not in ("approved", "rejected"):
        raise ValueError(f"status must be approved|rejected, got {status!r}")
    spath = Path(state_path) if state_path is not None else _DEFAULT_STATE_PATH
    now_dt = now or datetime.now(timezone.utc)
    state = _load_state(spath)
    row = state.get(month_key)
    if row is None or row.get("status") != "pending":
        return {}
    expires = _parse_iso(row.get("expires_at_iso"))
    if expires is not None and now_dt >= expires:
        row["status"] = "expired"
        state[month_key] = row
        _save_state(state, spath)
        return row
    row["status"] = status
    row["resolved_at_iso"] = now_dt.isoformat()
    state[month_key] = row
    _save_state(state, spath)
    return row


def is_last_trading_friday(d: date) -> bool:
    """True iff `d` is the last NYSE trading-Friday of its month.

    Uses `exchange_calendars` XNYS when installed; pure-Python fallback
    computes the last Friday of the calendar month and requires `d` to equal
    that Friday (doesn't handle holiday-shifted last-sessions without xcals,
    but matches the common case and the monthly_anchor rule as defined).
    """
    if d.weekday() != 4:  # not Friday
        return False
    try:
        import exchange_calendars as xcals  # type: ignore
        cal = xcals.get_calendar("XNYS")
        # Last session of the month on-or-before `d`, filtered to Fridays.
        _, last_day = calendar.monthrange(d.year, d.month)
        month_end = date(d.year, d.month, last_day)
        sessions = cal.sessions_in_range(
            f"{d.year:04d}-{d.month:02d}-01", month_end.isoformat()
        )
        # Find last Friday session in the month.
        fridays = [
            s for s in sessions
            if s.weekday() == 4
        ]
        if not fridays:
            return False
        last_fri = fridays[-1].date() if hasattr(fridays[-1], "date") else fridays[-1]
        return d == last_fri
    except Exception as exc:
        logger.info("is_last_trading_friday: xcals fallback: %r", exc)
        # Pure-Python: last Friday of the calendar month.
        _, last_day = calendar.monthrange(d.year, d.month)
        month_end = date(d.year, d.month, last_day)
        offset = (month_end.weekday() - 4) % 7
        last_friday = month_end - timedelta(days=offset)
        return d == last_friday


def _load_state(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("monthly_gate: state load fail-open: %r", exc)
        return {}


def _save_state(state: dict[str, Any], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("monthly_gate: state save fail-open: %r", exc)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


__all__ = [
    "run_monthly_sortino_gate",
    "record_approval",
    "is_last_trading_friday",
]
