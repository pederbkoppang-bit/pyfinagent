"""phase-66.2 hotfix tests: drawdown alarm DESC-order trap.

The production caller feeds get_paper_snapshots() (ORDER BY snapshot_date
DESC); compute_drawdown_from_snapshots assumed ASC and took navs[-1] (the
OLDEST row) as current NAV -- paging a phantom "-61.51%" P1 on a book UP 20%
(2026-07-06 20:05Z). The fix orders by the snapshot's own date key and
refuses to guess when no date key exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.drawdown_alarm import (  # noqa: E402
    check_drawdown_alarms,
    compute_drawdown_from_snapshots,
)


def _snap(date, nav):
    return {"snapshot_date": date, "total_nav": nav}


GROWING_ASC = [_snap(f"2026-06-{d:02d}", 20000 + d * 200) for d in range(1, 21)]


def test_desc_order_growing_nav_is_zero_drawdown_phantom_regression():
    """The 2026-07-06 phantom: DESC input + growing NAV must NOT report a
    drawdown (current == peak == the NEWEST row)."""
    desc = list(reversed(GROWING_ASC))
    assert compute_drawdown_from_snapshots(desc) == 0.0
    assert check_drawdown_alarms(desc) == []


def test_asc_order_unchanged():
    assert compute_drawdown_from_snapshots(GROWING_ASC) == 0.0


def test_real_drawdown_detected_regardless_of_order():
    asc = [_snap("2026-06-01", 24000), _snap("2026-06-02", 25000),
           _snap("2026-06-03", 22000)]  # -12% off the 25000 peak
    dd_asc = compute_drawdown_from_snapshots(asc)
    dd_desc = compute_drawdown_from_snapshots(list(reversed(asc)))
    assert dd_asc == dd_desc
    assert dd_asc is not None and abs(dd_asc - (-0.12)) < 1e-9
    tiers = check_drawdown_alarms(asc)
    assert [t[0] for t in tiers] == ["warn_3pct", "warn_5pct", "critical_10pct"]


def test_no_date_key_refuses_to_guess():
    undated = [{"total_nav": 25000}, {"total_nav": 9000}]
    assert compute_drawdown_from_snapshots(undated) is None
    assert check_drawdown_alarms(undated) == []


def test_insufficient_rows_none():
    assert compute_drawdown_from_snapshots([_snap("2026-06-01", 20000)]) is None
    assert compute_drawdown_from_snapshots([]) is None
