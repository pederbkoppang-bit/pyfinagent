"""phase-30.4 tests for paper_metrics_v2._nav_to_returns GIPS-canonical
external-flow subtraction.

Audit basis: handoff/archive/phase-30.0/experiment_results.md Anomaly A
(5/13 $5K deposit produced +32.12% phantom daily return, polluted the
Sharpe denominator -> Sharpe -6.26 anomaly). phase-30.4 fix: canonical
sub-period TWR `r_t = (V_t - F_t - V_{t-1}) / V_{t-1}` per Wikipedia
TWR + CFA L1 worked example.

Test plan (4 cases per research_brief.md Section "Test design"):
  1. test_no_flow_matches_legacy -- snapshots without external_flow_today
     produce identical returns to raw diff (regression guard, satisfies
     `no_regression_in_existing_metrics_v2_test`).
  2. test_deposit_excluded_from_return -- 5/13 case: V0=17818, V1=23541,
     flow=+5000 -> r ~ 4%, NOT 32% (satisfies
     `nav_to_returns_subtracts_external_flow_before_diff`).
  3. test_none_flow_fail_safe -- explicit `external_flow_today=None`
     treated as 0.0 (no crash).
  4. test_withdrawal_excluded -- negative external flow handled
     correctly (canonical formula is signed).

Plus regression guard:
  5. test_legacy_minimal_two_obs -- prior behavior for any pre-30.4
     test that passed snapshots without the new field still passes.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.services.paper_metrics_v2 import _nav_to_returns


def test_no_flow_matches_legacy():
    """Snapshots without external_flow_today -> same returns as raw diff."""
    snaps = [
        {"snapshot_date": "2026-01-01", "total_nav": 10000.0},
        {"snapshot_date": "2026-01-02", "total_nav": 10100.0},
        {"snapshot_date": "2026-01-03", "total_nav": 10200.0},
    ]
    r = _nav_to_returns(snaps)
    assert r[0] == pytest.approx(0.01)
    assert r[1] == pytest.approx(0.0099, rel=1e-3)


def test_deposit_excluded_from_return():
    """5/13 case: V0=17818.31, V1=23541.77, flow=+$5000 -> r ~ 4.06%, NOT 32.12%.

    This is the canonical-bug reproducer from phase-30.0 Anomaly A.
    """
    snaps = [
        {
            "snapshot_date": "2026-05-12",
            "total_nav": 17818.31,
            "external_flow_today": 0.0,
        },
        {
            "snapshot_date": "2026-05-13",
            "total_nav": 23541.77,
            "external_flow_today": 5000.0,
        },
    ]
    r = _nav_to_returns(snaps)
    assert len(r) == 1
    # Canonical: (23541.77 - 5000 - 17818.31) / 17818.31 = 0.04063...
    assert r[0] == pytest.approx(0.0406, rel=1e-2)
    # And explicitly NOT the pre-fix 32%+ phantom value.
    assert r[0] < 0.10, (
        f"phase-30.4 fix FAILED: deposit still polluted return ({r[0]:.4f}); "
        f"expected ~0.0406, NOT 0.3212 (pre-fix phantom)"
    )


def test_none_flow_fail_safe():
    """external_flow_today is None -> treated as 0.0, no crash.

    Legacy snapshots pre-30.4 backfill carry NULL in this field; the
    fix must not crash on None values.
    """
    snaps = [
        {
            "snapshot_date": "2026-01-01",
            "total_nav": 10000.0,
            "external_flow_today": None,
        },
        {
            "snapshot_date": "2026-01-02",
            "total_nav": 10100.0,
            "external_flow_today": None,
        },
    ]
    r = _nav_to_returns(snaps)
    assert r[0] == pytest.approx(0.01)


def test_withdrawal_excluded():
    """Negative external flow (withdrawal) handled correctly.

    Canonical formula is signed: V_t - F_t covers both deposits
    (F positive) and withdrawals (F negative). Pre-fix raw diff would
    show a phantom -10% loss on a withdrawal day; post-fix shows the
    true market move only.
    """
    snaps = [
        {
            "snapshot_date": "2026-01-01",
            "total_nav": 10000.0,
            "external_flow_today": 0.0,
        },
        {
            "snapshot_date": "2026-01-02",
            "total_nav": 8900.0,
            "external_flow_today": -1000.0,
        },
    ]
    r = _nav_to_returns(snaps)
    # Canonical: (8900 - (-1000) - 10000) / 10000 = -0.01
    assert r[0] == pytest.approx(-0.01, rel=1e-3)


def test_legacy_minimal_two_obs_no_field():
    """Regression: any pre-30.4 caller that passed snapshots without the
    new field still produces the same returns."""
    # Identical to test_no_flow_matches_legacy but explicit: matches the
    # original raw `np.diff(navs) / navs[:-1]` shape.
    snaps = [
        {"snapshot_date": "2026-01-01", "total_nav": 1000.0},
        {"snapshot_date": "2026-01-02", "total_nav": 1050.0},
    ]
    r = _nav_to_returns(snaps)
    assert np.isclose(r[0], 0.05)
