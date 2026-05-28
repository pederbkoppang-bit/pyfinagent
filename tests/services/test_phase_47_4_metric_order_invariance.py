"""phase-47.4: NAV-series order-invariance guards for Sharpe + max-drawdown.

`get_paper_snapshots` returns rows newest-first (ORDER BY snapshot_date DESC).
Both `compute_sharpe_from_snapshots` and `paper_go_live_gate._snapshot_max_dd_pct`
walk the NAV series; a correct Sharpe/drawdown is ORDER-INVARIANT. Before the
fix the cockpit Sharpe read -5.72 (sign-flipped) and the gate maxDD read 60.08%
(phantom -- portfolio growth read backwards as a crash). These guards FAIL on
the pre-fix code (reversed order changes both) and PASS once each helper sorts
chronologically.
"""
from __future__ import annotations

from backend.services.paper_go_live_gate import _snapshot_max_dd_pct
from backend.services.perf_metrics import compute_sharpe_from_snapshots


def _snaps(navs: list[float]) -> list[dict]:
    """Build chronological snapshot dicts (snapshot_date ascending in list order)."""
    return [
        {"snapshot_date": f"2026-01-{i + 1:02d}", "total_nav": float(n)}
        for i, n in enumerate(navs)
    ]


# A growing fund: the only dip is 10100->10050 (-0.495%); no real crash.
_GROWTH = [10000, 10100, 10050, 10300, 10500, 10800, 11000]


def test_sharpe_is_order_invariant_and_positive_for_growth():
    chron = _snaps(_GROWTH)
    desc = list(reversed(chron))  # newest-first, as get_paper_snapshots returns
    s_chron = compute_sharpe_from_snapshots(chron)
    s_desc = compute_sharpe_from_snapshots(desc)
    assert s_chron == s_desc, f"Sharpe must be order-invariant: {s_chron} vs {s_desc}"
    assert s_chron > 0, f"a growing fund must have a positive Sharpe, got {s_chron}"


def test_max_dd_is_order_invariant_and_small_for_growth():
    chron = _snaps(_GROWTH)
    desc = list(reversed(chron))
    dd_chron = _snapshot_max_dd_pct(chron)
    dd_desc = _snapshot_max_dd_pct(desc)
    assert abs(dd_chron - dd_desc) < 1e-9, (
        f"maxDD must be order-invariant: {dd_chron} vs {dd_desc}"
    )
    # The only dip is 0.495%; a growth series must not show a >5% (let alone 60%) drawdown.
    assert dd_chron < 5.0, f"growth-series maxDD should be tiny, got {dd_chron}"
