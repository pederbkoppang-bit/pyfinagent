"""phase-25.L: drawdown alarm with tiered thresholds.

Reads a list of paper-portfolio snapshots (each carrying a `total_nav`
or `nav` value), computes the current drawdown vs the all-time peak,
and fires per-tier Slack alerts via the existing `raise_cron_alert_sync`
infrastructure.

Tiers (canonical retail-trader thresholds; see research_brief.md):

    warn_3pct        : -3%   - P2 (moderate; logged but not paging)
    warn_5pct        : -5%   - P1 (serious; pages operator)
    critical_10pct   : -10%  - P1 (critical; pages operator)

Each tier has a distinct dedup error_type
(`drawdown_warn_3pct` / `drawdown_warn_5pct` / `drawdown_critical_10pct`)
so AlertDeduper suppresses repeated same-tier alerts inside the
configured repeat-hours window while still firing a fresh alert if a
deeper tier breaches later.

Module is intentionally side-effect-free at import time. The wired
caller (autonomous_loop.py) invokes `emit_drawdown_alarms` from inside
its finally block; this module never calls Slack directly outside that
entry point.

Closes audit bucket 24.5 F-5(c) + 24.8.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# (tier_name, drawdown_threshold_pct (negative), severity)
DRAWDOWN_TIERS: list[tuple[str, float, str]] = [
    ("warn_3pct", -0.03, "P2"),
    ("warn_5pct", -0.05, "P1"),
    ("critical_10pct", -0.10, "P1"),
]


def _snapshot_nav(snapshot: dict) -> float | None:
    """Pull NAV from a snapshot dict, accepting `total_nav` or `nav`."""
    for key in ("total_nav", "nav", "portfolio_value"):
        v = snapshot.get(key)
        if v is None:
            continue
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return None


def compute_drawdown_from_snapshots(snapshots: Iterable[dict]) -> float | None:
    """Return the most-recent drawdown vs all-time peak (negative for losses).

    Returns None when the snapshot list has fewer than 2 valid NAV rows
    or all rows have NAV<=0. A return of 0.0 means current NAV equals
    or exceeds the all-time peak (no drawdown).
    """
    # phase-66.2 hotfix (2026-07-07): "current" was navs[-1], which assumes
    # ASC order -- but the production caller feeds get_paper_snapshots(),
    # which is ORDER BY snapshot_date DESC (bigquery_client.py:1042), so the
    # OLDEST row in the window was treated as current NAV. On 2026-07-06 this
    # paged a phantom "-61.51% drawdown" P1 against a book UP 20% (the
    # phase-47.4 DESC-trap class). Fix: order by the snapshot's own date key
    # when present; refuse to guess (return None) when no date key exists
    # and the sequence order is therefore unknowable.
    dated: list[tuple] = []
    undated_navs: list[float] = []
    for s in snapshots or []:
        nav = _snapshot_nav(s) if isinstance(s, dict) else None
        if nav is None:
            continue
        ts = None
        for key in ("snapshot_date", "date", "created_at", "updated_at", "ts"):
            v = s.get(key)
            if v is not None:
                ts = str(v)
                break
        if ts is not None:
            dated.append((ts, nav))
        else:
            undated_navs.append(nav)

    if dated:
        dated.sort()  # ISO strings sort chronologically
        navs = [nav for _, nav in dated]
    else:
        navs = undated_navs
        if len(navs) >= 2:
            logger.warning(
                "drawdown: snapshots carry no date key; order unknowable -- "
                "refusing to compute (fail-safe, no alarm)"
            )
            return None

    if len(navs) < 2:
        return None
    peak = max(navs)
    current = navs[-1]
    if peak <= 0:
        return None
    return (current - peak) / peak


def check_drawdown_alarms(snapshots: Iterable[dict]) -> list[tuple[str, float, str]]:
    """Return the list of currently-breached tiers as (tier_name, dd_pct, severity).

    Empty list when no breach OR when drawdown can't be computed
    (insufficient snapshots).
    """
    dd = compute_drawdown_from_snapshots(snapshots)
    if dd is None or dd > -0.03:
        return []
    breached: list[tuple[str, float, str]] = []
    for tier_name, threshold, severity in DRAWDOWN_TIERS:
        if dd <= threshold:
            breached.append((tier_name, dd, severity))
    return breached


def emit_drawdown_alarms(
    snapshots: Iterable[dict],
    *,
    source: str = "autonomous_loop",
) -> int:
    """Fire per-tier Slack alerts. Returns the number of alerts fired
    (after dedup -- AlertDeduper may suppress some).

    Fully fail-open: if anything goes wrong (alerting unavailable,
    snapshots malformed, deduper raises), the function logs at WARNING
    and returns 0 instead of raising.
    """
    try:
        breached = check_drawdown_alarms(snapshots)
        if not breached:
            return 0
        from backend.services.observability.alerting import raise_cron_alert_sync
        n_fired = 0
        for tier_name, dd_pct, severity in breached:
            error_type = f"drawdown_{tier_name}"
            title = f"Portfolio drawdown {dd_pct * 100:.2f}% breached tier {tier_name}"
            details = {
                "tier": tier_name,
                "drawdown_pct": f"{dd_pct * 100:.4f}",
                "severity": severity,
                "snapshot_count": len(list(snapshots)) if isinstance(snapshots, (list, tuple)) else "n/a",
            }
            fired = raise_cron_alert_sync(
                source=source,
                error_type=error_type,
                severity=severity,
                title=title,
                details=details,
            )
            if fired:
                n_fired += 1
        return n_fired
    except Exception as exc:
        logger.warning(
            "emit_drawdown_alarms fail-open (source=%s): %r",
            source,
            exc,
        )
        return 0


__all__ = [
    "DRAWDOWN_TIERS",
    "compute_drawdown_from_snapshots",
    "check_drawdown_alarms",
    "emit_drawdown_alarms",
]
