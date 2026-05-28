"""phase-43.0 cycle-17 -- windowed-Sharpe helper (DoD-2) tests.

Covers the cycle-16 helpers in backend/services/perf_metrics.py:
  - compute_paper_sharpe_window (lines 118-169)
  - compute_sharpe_gap(window_days=...) (lines 240-349)

Q/A verdict a30ae6755518b9ced (cycle-16) flagged four NOTEs:
  1. window_days < 6 early-return guard (line 145)
  2. len(window) < 6 post-slice insufficiency (line 161-162)
  3. windowed value differs from legacy on a synthetic snapshot set
  4. compute_sharpe_gap(window_days=None) is byte-identical to legacy

Fixture pattern mirrors backend/tests/test_dod4_tier1_coverage_investment.py:
in-test MagicMock() construction, no shared conftest fixture.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _snap(day: int, nav: float, year_month: str = "2026-04") -> dict:
    """One mock paper-portfolio snapshot row.

    Uses the canonical key shape (total_nav + snapshot_date) that
    compute_paper_sharpe_window expects for both NAV access and the
    sort step at perf_metrics.py:157.
    """
    return {
        "total_nav": float(nav),
        "snapshot_date": f"{year_month}-{day:02d}",
    }


# ---------- Case 1: window_days < 6 early-return guard ----------

def test_compute_paper_sharpe_window_returns_none_when_window_too_small():
    """window_days < 6 hits the early-return guard at perf_metrics.py:145.

    The helper must return None BEFORE calling bq.get_paper_snapshots,
    so we verify the mock was not called.
    """
    from backend.services.perf_metrics import compute_paper_sharpe_window

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = [
        _snap(i + 1, 100.0 + i) for i in range(30)
    ]

    for n in (0, 1, 5):
        result = compute_paper_sharpe_window(bq, window_days=n)
        assert result is None, f"expected None for window_days={n}, got {result}"

    bq.get_paper_snapshots.assert_not_called()


# ---------- Case 2: post-slice insufficiency ----------

def test_compute_paper_sharpe_window_returns_none_when_window_slice_too_short():
    """Even with window_days >= 6, if BQ returns < 6 rows the post-slice
    len(window) < 6 guard at perf_metrics.py:161-162 must fire."""
    from backend.services.perf_metrics import compute_paper_sharpe_window

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = [
        _snap(i + 1, 100.0 + i) for i in range(5)
    ]

    result = compute_paper_sharpe_window(bq, window_days=30)
    assert result is None
    bq.get_paper_snapshots.assert_called_once_with(limit=60)  # max(30*2, 60)


# ---------- Case 3: windowed differs from legacy on synthetic data ----------

def test_compute_paper_sharpe_window_differs_from_legacy_on_synthetic_set():
    """When the trailing window has a different return distribution from
    the all-time series, the windowed Sharpe must differ from the legacy
    all-snapshot Sharpe -- proving both branches are exercised, not aliases.
    """
    from backend.services.perf_metrics import (
        compute_paper_sharpe_window,
        compute_sharpe_from_snapshots,
    )

    # Build a synthetic 60-snapshot series with controlled volatility on
    # both halves so neither Sharpe overflows the [-100, 100] clamp at
    # compute_sharpe_from_snapshots and gets remapped to None:
    #   - first 30 days (month 03): high-variance bounces around 97 (low/neg Sharpe)
    #   - last 30 days (month 04): mild uptrend with noise (moderate positive Sharpe)
    import random
    random.seed(43)
    legacy_snaps = []
    for i in range(30):
        nav = 97.0 + (2.5 if i % 2 == 0 else -2.5) + random.uniform(-0.5, 0.5)
        legacy_snaps.append(_snap(i + 1, nav, year_month="2026-03"))
    for i in range(30):
        nav = 100.0 + i * 0.5 + random.uniform(-0.3, 0.3)
        legacy_snaps.append(_snap(i + 1, nav, year_month="2026-04"))

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = legacy_snaps

    windowed = compute_paper_sharpe_window(bq, window_days=30)
    legacy = compute_sharpe_from_snapshots(legacy_snaps)

    assert windowed is not None, "trailing window of 30 should compute"
    assert legacy != 0.0, "60-snapshot series should compute a Sharpe"

    # They must differ -- proves the slice path is exercised, not a no-op.
    assert windowed != legacy, (
        f"windowed Sharpe ({windowed}) must differ from legacy ({legacy}) "
        "on this divergent synthetic set"
    )


# ---------- Case 4: compute_sharpe_gap(window_days=None) byte-identical ----------

def test_compute_sharpe_gap_window_none_byte_identical_to_legacy():
    """compute_sharpe_gap(window_days=None) must preserve the
    pre-cycle-16 behaviour byte-for-byte: same BQ call shape
    (limit=365), same output dict shape, same live_sharpe value
    given the same mock snapshots.
    """
    from backend.services.perf_metrics import (
        compute_sharpe_gap,
        compute_sharpe_from_snapshots,
    )

    snaps = [_snap(i + 1, 100_000.0 + i * 50.0) for i in range(30)]

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = snaps

    out = compute_sharpe_gap(bq)  # window_days defaults to None

    # 4a. BQ call shape: the legacy all-time pull uses limit=365.
    # Note: compute_sharpe_gap may call get_paper_snapshots a second time
    # via _shadow_curve_sharpe fallback when optimizer_best.json is absent;
    # we only assert at least one call with limit=365 (the live-Sharpe arm).
    calls = bq.get_paper_snapshots.call_args_list
    assert any(
        c.kwargs.get("limit") == 365 or (c.args and c.args[0] == 365)
        for c in calls
    ), f"expected at least one call with limit=365, got {calls}"

    # 4b. Live Sharpe matches the direct compute_sharpe_from_snapshots
    #     value -- proves the window_days=None branch routes through the
    #     legacy primitive, not the new windowed helper.
    expected_live = compute_sharpe_from_snapshots(snaps)
    if expected_live == 0.0:
        # legacy "could not compute" -> compute_sharpe_gap remaps to None
        assert out["live_sharpe"] is None
    else:
        assert out["live_sharpe"] == expected_live

    # 4c. Output dict shape -- full key set unchanged.
    expected_keys = {
        "live_sharpe", "backtest_sharpe", "gap_abs", "gap_rel",
        "threshold", "gap_within_threshold", "source", "note",
        "proxy_fallback", "computed_at",
    }
    assert set(out.keys()) == expected_keys

    # 4d. threshold is the SR_GAP_THRESHOLD constant (0.30).
    assert out["threshold"] == 0.30
