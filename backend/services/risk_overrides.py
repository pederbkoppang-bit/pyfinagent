"""
risk_overrides -- Runtime operator control of the paper-trading deployment /
concentration caps, WITHOUT a backend restart (phase-49.1, P7 "risk limits").

The four deployment knobs below are read AT-DECIDE-TIME in
`portfolio_manager.build_trade_decisions` via `get_effective(key, settings.X)`.
An override set here is therefore picked up by the NEXT daily cycle with no
restart -- the loop runs in the backend's own APScheduler process, the same
process as the API (see research_brief phase-49.1, Q1).

Design mirrors `kill_switch.py` (the proven file-backed pattern):
  - module singleton + threading.Lock
  - append-only JSONL audit at handoff/risk_overrides_audit.jsonl
  - replay-on-init so overrides survive a restart

Safety (grounded in SEC Rule 15c3-5 + Fed FEDS-2025-034 + Knight Capital
$440M post-mortem + OneUptime/Unleash hot-reload guidance):
  - ALLOWED_KEYS is a strict allowlist. The kill-switch loss-limit breach
    checks (daily_loss_limit_pct / trailing_dd_limit_pct) are deliberately
    NOT in this surface -- they live in kill_switch.py and must never be
    mutable/disable-able through here (Knight Capital lesson).
  - Every mutation is BOUNDED (validate-before-accept; reject out-of-range
    with a clear error) and AUDITED (key, old_value, new_value, reason, ts).
  - "cleared" means "revert to the settings default" -- unambiguous, never
    a stale/ambiguous flag.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.utils import json_io

logger = logging.getLogger(__name__)

_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "risk_overrides_audit.jsonl"
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Allowlist + bounds ──────────────────────────────────────────────
# key -> (python type, min_inclusive, max_inclusive, human description).
# These are the ONLY keys this surface can touch. Bounds mirror the
# settings.py Field(ge/le) constraints where they exist + add sane caps
# to prevent fat-finger (Knight Capital). The kill-switch loss limits are
# intentionally absent.
BOUNDS: dict[str, tuple] = {
    "paper_max_per_sector": (int, 0, 20, "Max BUY positions in any single GICS sector (0 = no limit)"),
    "paper_max_per_sector_nav_pct": (float, 0.0, 100.0, "Max NAV percentage per single GICS sector (0 = no limit)"),
    "paper_min_cash_reserve_pct": (float, 0.0, 50.0, "Minimum cash reserve as % of NAV"),
    "paper_max_positions": (int, 1, 50, "Maximum simultaneous open positions"),
}
ALLOWED_KEYS = frozenset(BOUNDS.keys())


class RiskOverrideError(ValueError):
    """Raised when an override request is for a disallowed key or an
    out-of-bounds / non-coercible value. Carries an operator-readable message."""


def _coerce_and_validate(key: str, value: Any) -> float | int:
    """Coerce `value` to the key's declared type and assert it is within
    bounds. Raises RiskOverrideError on any problem (validate-before-accept)."""
    if key not in BOUNDS:
        raise RiskOverrideError(
            f"'{key}' is not an adjustable risk limit. Allowed keys: {sorted(ALLOWED_KEYS)}"
        )
    typ, lo, hi, _desc = BOUNDS[key]
    try:
        coerced = typ(value)
    except (TypeError, ValueError):
        raise RiskOverrideError(f"value {value!r} for '{key}' is not a valid {typ.__name__}")
    if not (lo <= coerced <= hi):
        raise RiskOverrideError(
            f"value {coerced} for '{key}' is out of bounds [{lo}, {hi}]"
        )
    return coerced


class RiskOverrideState:
    """Module-level thread-safe override store. Persisted across restarts via
    the audit log: replaying `set` lines re-applies the override, `clear`
    removes it, `clear_all` wipes them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._overrides: dict[str, float | int] = {}
        self._load_from_audit()

    def _load_from_audit(self) -> None:
        if not _AUDIT_PATH.exists():
            return
        try:
            with _AUDIT_PATH.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json_io.parse_json_line(line)
                    except Exception:
                        continue
                    event = row.get("event")
                    key = row.get("key")
                    if event == "set" and key in BOUNDS:
                        # Re-validate on replay; skip rows that no longer pass
                        # bounds (e.g. if BOUNDS tightened since they were written).
                        try:
                            self._overrides[key] = _coerce_and_validate(key, row.get("new_value"))
                        except RiskOverrideError:
                            continue
                    elif event == "clear" and key in self._overrides:
                        self._overrides.pop(key, None)
                    elif event == "clear_all":
                        self._overrides.clear()
        except Exception as e:
            logger.warning(f"risk_overrides: audit load failed: {e}")

    @staticmethod
    def _append_audit(event: str, **fields: Any) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            with _AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"risk_overrides: audit write failed: {e}")

    # ── Read ────────────────────────────────────────────────────────
    def get_effective(self, key: str, default: Any) -> Any:
        """Return the active override for `key` if one is set, else `default`
        (the settings value the caller passes). Unknown keys always return
        the default -- this surface never invents limits."""
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]
        return default

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._overrides)

    # ── Mutations (bounded + audited) ───────────────────────────────
    def set_override(self, key: str, value: Any, reason: str = "manual") -> dict:
        coerced = _coerce_and_validate(key, value)  # raises before any state change
        with self._lock:
            old = self._overrides.get(key)
            self._overrides[key] = coerced
            self._append_audit("set", key=key, old_value=old, new_value=coerced, reason=reason)
            snap = dict(self._overrides)
        logger.info("risk_overrides: SET %s %s -> %s (reason=%s)", key, old, coerced, reason)
        return snap

    def clear_override(self, key: str, reason: str = "manual") -> dict:
        with self._lock:
            old = self._overrides.pop(key, None)
            if old is not None:
                self._append_audit("clear", key=key, old_value=old, new_value=None, reason=reason)
            snap = dict(self._overrides)
        if old is not None:
            logger.info("risk_overrides: CLEAR %s (was %s, reason=%s)", key, old, reason)
        return snap

    def clear_all(self, reason: str = "manual") -> dict:
        with self._lock:
            had = dict(self._overrides)
            self._overrides.clear()
            if had:
                self._append_audit("clear_all", old_value=had, reason=reason)
            snap = dict(self._overrides)
        if had:
            logger.info("risk_overrides: CLEAR_ALL (was %s, reason=%s)", had, reason)
        return snap


_state = RiskOverrideState()


def get_state() -> RiskOverrideState:
    return _state


# ── Module-level convenience API (used by portfolio_manager + the router) ──
def get_effective(key: str, default: Any) -> Any:
    return _state.get_effective(key, default)


def set_override(key: str, value: Any, reason: str = "manual") -> dict:
    return _state.set_override(key, value, reason=reason)


def clear_override(key: str, reason: str = "manual") -> dict:
    return _state.clear_override(key, reason=reason)


def clear_all(reason: str = "manual") -> dict:
    return _state.clear_all(reason=reason)


def snapshot() -> dict:
    return _state.snapshot()


def describe() -> dict:
    """Effective-state view for the GET endpoint: per allowed key, its bound
    spec, whether it's overridden, and the override value (if any)."""
    snap = _state.snapshot()
    out = {}
    for key, (typ, lo, hi, desc) in BOUNDS.items():
        out[key] = {
            "type": typ.__name__,
            "min": lo,
            "max": hi,
            "description": desc,
            "overridden": key in snap,
            "override_value": snap.get(key),
        }
    return out
