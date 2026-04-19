"""phase-6.6 calendar event normalization -- event_id + window mapping.

Deterministic `event_id = sha256(event_type + "|" + ticker + "|" +
(fiscal_period_end or DATE(scheduled_at)))`. Re-fetches of the same event
from the same or different sources produce the same id -> dedup in watcher.
"""
from __future__ import annotations

import hashlib


# Map source-specific timing tokens to the canonical `window` enum.
# `bmo` = before market open, `amc` = after market close, `dmh` = during
# market hours. Missing or unknown -> `all_day`.
_FINNHUB_HOUR_TO_WINDOW = {
    "bmo": "pre_open",
    "amc": "post_close",
    "dmh": "intraday",
    "": "all_day",
    None: "all_day",
}

_VALID_WINDOWS = frozenset({"pre_open", "post_close", "intraday", "all_day"})


def normalize_window(window: str | None) -> str:
    """Map raw window/hour string to canonical enum.

    Accepts already-canonical values (pre_open / post_close / ...) and passes
    them through; maps Finnhub-style (`bmo` / `amc` / `dmh`) to their
    canonical form; otherwise returns `all_day`.
    """
    if window in _VALID_WINDOWS:
        return window  # type: ignore[return-value]
    return _FINNHUB_HOUR_TO_WINDOW.get(window, "all_day")


def compute_event_id(
    *,
    event_type: str,
    ticker: str | None,
    fiscal_period_end: str | None,
    scheduled_at: str,
) -> str:
    """Deterministic id across sources and re-fetches.

    For earnings: prefers `fiscal_period_end` (stable once quarter closes).
    For FOMC/macro: falls back to `DATE(scheduled_at)`.
    """
    t = (event_type or "").strip().lower()
    tk = (ticker or "").strip().upper()
    if fiscal_period_end:
        anchor = str(fiscal_period_end)[:10]
    else:
        anchor = str(scheduled_at)[:10]
    payload = f"{t}|{tk}|{anchor}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "compute_event_id",
    "normalize_window",
]
