"""phase-82.5 -- exit-quality tiles must not be division blowups.

THE MEAN OF THESE RATIOS DOES NOT EXIST. `capture = pnl/MFE` and `edge = MFE/|MAE|`
have denominators that can reach zero, which makes the per-trade distribution
Cauchy-like. Franz (arXiv:0710.2024): "Neither the expected value nor the variance
exist", and the mean of N such variables follows the SAME distribution as one of them.
So the shipped `avg_capture_ratio` of -42.08 was not a bad estimate of a true value --
there is no true value, and more trades would never make it converge.

That is why these tests do not check "the outlier is clipped". Winsorizing would also
produce a finite number, and it would estimate nothing. The ESTIMATOR changed.

The fixture is the REAL 32 closed round-trips
(`backend/tests/fixtures/exit_quality_32_roundtrips.json`), pulled from
`financial_reports.paper_trades`. It reproduces the numbers in the masterplan step
exactly: legacy mean capture -42.0785, legacy mean edge 86.9218, 8 rows with mfe==0,
6 with mae==0.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.services.perf_metrics import (
    MIN_MFE_PCT,
    aggregate_exit_quality,
    compute_capture_ratio,
    compute_edge_ratio,
)

FIXTURE = Path(__file__).parent / "fixtures" / "exit_quality_32_roundtrips.json"


@pytest.fixture(scope="module")
def real_rows() -> list[dict]:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(rows) == 32, f"fixture drifted: {len(rows)} rows, expected 32"
    return rows


def _ordinary(n: int = 20) -> list[dict]:
    """Well-behaved trades: MFE comfortably above the floor, real adverse excursion."""
    return [
        {"mfe_pct": 5.0 + i * 0.5, "mae_pct": -2.0 - i * 0.1,
         "realized_pnl_pct": 3.0 + i * 0.3}
        for i in range(n)
    ]


# ── criterion 4: pin the PRE-FIX behaviour on the real fixture ───────

def test_the_old_mean_formula_blows_up_on_the_committed_real_data(real_rows):
    """Criterion 4. If this stops reproducing, the fixture no longer represents
    the defect and every other test here becomes a claim about nothing."""
    legacy = [r["capture_ratio_legacy"] for r in real_rows]
    old_mean = sum(legacy) / len(legacy)
    assert abs(old_mean) > 40, (
        f"the committed fixture no longer reproduces the defect: old mean={old_mean}"
    )
    assert old_mean == pytest.approx(-42.0785, abs=1e-3), old_mean
    # And the single row that causes it is present and unmodified.
    worst = min(legacy)
    assert worst < -1000, f"the 000660.KS blowup row is missing: min={worst}"


def test_the_old_edge_mean_also_blows_up_on_the_real_data(real_rows):
    """Both tiles were broken; pinning only one would let the other regress."""
    er = [r["mfe_pct"] / abs(r["mae_pct"]) for r in real_rows if abs(r["mae_pct"]) > 0]
    assert len(er) == 26, f"expected 6 mae==0 rows dropped by the old filter, got {32-len(er)}"
    assert sum(er) / len(er) == pytest.approx(86.9218, abs=1e-3)


def test_the_new_aggregate_is_sane_on_the_same_real_data(real_rows):
    """Measured, not predicted. The research gate guessed n_defined=24; the 1.0pp
    floor actually removes 12 rows, not 8, because four more sit in 0 < mfe < 1."""
    a = aggregate_exit_quality(real_rows)
    assert a["capture_median"] == pytest.approx(0.6304, abs=1e-3), a["capture_median"]
    assert a["capture_n_defined"] == 20
    assert a["capture_n_undefined"] == 12
    assert a["edge_median"] == pytest.approx(3.0900, abs=1e-3), a["edge_median"]
    assert a["edge_n_infinite"] == 6
    # The headline a human reads must be in a believable range.
    assert 0.0 <= a["capture_median"] <= 1.5
    assert math.isfinite(a["edge_median"])


# ── criterion 1: MFE == 0 ────────────────────────────────────────────

def test_mfe_zero_does_not_produce_an_unbounded_or_sign_flipped_capture(real_rows):
    """Criterion 1. The old code substituted 0.0 for undefined; 8 of 32 real rows
    hit that path and were averaged as though they were observations."""
    zero_rows = [r for r in real_rows if r["mfe_pct"] == 0.0]
    assert len(zero_rows) == 8, f"fixture drifted: {len(zero_rows)} mfe==0 rows"
    for r in zero_rows:
        assert compute_capture_ratio(r["realized_pnl_pct"], r["mfe_pct"]) is None

    a = aggregate_exit_quality(real_rows)
    assert a["capture_median"] is not None
    assert math.isfinite(a["capture_median"])
    assert a["capture_median"] > 0, (
        "a sign-flipped headline: the legacy mean was -42.08 because losing trades "
        "with a near-zero denominator dominated"
    )


def test_an_undefined_capture_is_distinguishable_from_a_real_zero():
    """The second half of the same defect, and the subtler one. 0.0 is a genuine
    outcome -- captured nothing of a real move. Returning 0.0 for 'undefined' made
    the two indistinguishable inside the average."""
    assert compute_capture_ratio(0.0, 10.0) == 0.0        # real: captured none of a 10pp move
    assert compute_capture_ratio(-0.13, 0.0) is None      # undefined: no move to grade
    assert compute_capture_ratio(-0.13, 0.0001) is None   # undefined: below the floor


def test_an_all_undefined_fixture_reports_none_not_zero():
    """INV3. Rendering 0% for 'nothing gradeable' tells the operator the exits
    captured nothing, which is a different and false claim."""
    a = aggregate_exit_quality([
        {"mfe_pct": 0.0, "mae_pct": -1.0, "realized_pnl_pct": -1.0} for _ in range(5)
    ])
    assert a["capture_median"] is None
    assert a["capture_ratio_of_sums"] is None
    assert a["capture_n_defined"] == 0 and a["capture_n_undefined"] == 5


# ── criterion 2: MAE == 0 handled, never silently dropped ────────────

def test_mae_zero_is_handled_explicitly_and_not_dropped(real_rows):
    """Criterion 2. `if mae_abs > 0` deleted the 6 trades that never traded
    against us -- the BEST trades. Survivorship bias with the sign pointing the
    wrong way."""
    zero_mae = [r for r in real_rows if abs(r["mae_pct"]) == 0.0]
    assert len(zero_mae) == 6, f"fixture drifted: {len(zero_mae)} mae==0 rows"
    for r in zero_mae:
        e = compute_edge_ratio(r["mfe_pct"], r["mae_pct"])
        assert e is not None, "a perfect trade was dropped instead of ranked"
        assert math.isinf(e) or e == 0.0

    a = aggregate_exit_quality(real_rows)
    # INV2: nothing silently vanishes from either denominator.
    assert a["edge_n_defined"] + a["edge_n_undefined"] == a["n_points"] == 32
    assert a["capture_n_defined"] + a["capture_n_undefined"] == 32
    assert a["edge_n_infinite"] == 6


def test_edge_ratio_degenerate_cases_are_each_defined():
    assert math.isinf(compute_edge_ratio(5.0, 0.0))       # no adverse move at all
    assert compute_edge_ratio(0.0, 0.0) is None           # neither excursion happened
    assert compute_edge_ratio(0.0, -3.0) == 0.0           # a real, meaningful zero
    assert compute_edge_ratio(6.0, -2.0) == 3.0


def test_an_infinite_median_is_reported_as_none_never_as_infinity():
    """If most trades are degenerate the median lands on +inf. That is a real
    answer but not a renderable one; a tile must never print Infinity."""
    a = aggregate_exit_quality([
        {"mfe_pct": 5.0, "mae_pct": 0.0, "realized_pnl_pct": 4.0} for _ in range(5)
    ])
    assert a["edge_median"] is None
    assert a["edge_n_infinite"] == 5


# ── criterion 3: robustness to a single extreme outlier ──────────────

def test_one_extreme_outlier_moves_each_headline_by_under_20_percent():
    """Criterion 3, verbatim: adding one round-trip with mfe/mae = 1e-4 to a
    fixture of ordinary trades must move the reported value by < 20%."""
    base = _ordinary()
    before = aggregate_exit_quality(base)
    poisoned = base + [{"mfe_pct": 1e-4, "mae_pct": -1e-4, "realized_pnl_pct": -0.13}]
    after = aggregate_exit_quality(poisoned)

    for key in ("capture_median", "edge_median"):
        b, a = before[key], after[key]
        assert b is not None and a is not None, key
        drift = abs(a - b) / abs(b)
        assert drift < 0.20, f"{key} moved {drift:.1%} ({b} -> {a})"


def test_the_median_itself_is_load_bearing_not_just_the_floor():
    """Criterion 3 names an outlier at mfe=1e-4 -- but that row is EXCLUDED by the
    1.0pp floor, so the criterion-3 test above passes identically whether the
    headline is a mean or a median. The floor is doing that work, not the estimator.

    Measured: reverting the headline to a mean kills only ONE test in this suite,
    which is too little coverage for the change the step is actually about. So this
    test poisons with a row ABOVE the floor -- mfe=1.5, a legitimate small move with
    a large adverse pnl -- where the floor cannot help and only a robust estimator
    can. A mean fails here; the median does not.
    """
    base = _ordinary()
    poison = {"mfe_pct": 1.5, "mae_pct": -0.2, "realized_pnl_pct": -900.0}

    before = aggregate_exit_quality(base)
    after = aggregate_exit_quality(base + [poison])
    drift = abs(after["capture_median"] - before["capture_median"]) / abs(before["capture_median"])
    assert drift < 0.20, f"capture median moved {drift:.1%}"

    # The poison row IS admitted (above the floor) -- otherwise this proves nothing.
    assert after["capture_n_defined"] == before["capture_n_defined"] + 1
    assert compute_capture_ratio(poison["realized_pnl_pct"], poison["mfe_pct"]) < -500

    # And a mean over that same admitted set is destroyed by it.
    caps = [compute_capture_ratio(r["realized_pnl_pct"], r["mfe_pct"]) for r in base]
    mean_before = sum(caps) / len(caps)
    caps_after = caps + [compute_capture_ratio(poison["realized_pnl_pct"], poison["mfe_pct"])]
    mean_after = sum(caps_after) / len(caps_after)
    assert abs(mean_after - mean_before) / abs(mean_before) > 1.0, (
        "the poison row does not break a mean, so this test cannot discriminate "
        "between the two estimators"
    )


def test_the_same_outlier_destroys_the_old_mean():
    """Proves the test above is not vacuous: the poison row IS extreme enough to
    matter, and the previous estimator does move ~40x on it."""
    base = _ordinary()
    poison = {"mfe_pct": 1e-4, "mae_pct": -1e-4, "realized_pnl_pct": -0.13}

    def old_mean(rows):
        return sum((r["realized_pnl_pct"] / r["mfe_pct"]) if r["mfe_pct"] > 0 else 0.0
                   for r in rows) / len(rows)

    b, a = old_mean(base), old_mean(base + [poison])
    assert abs(a - b) / abs(b) > 20, (
        f"the poison row is not extreme enough to be a real test: {b} -> {a}"
    )


def test_the_edge_headline_is_robust_to_an_extreme_outlier():
    """Criterion 3 says BOTH tiles. Until now only the capture tile had a real
    robustness guard.

    Q/A cycle-2 [WARN]: reverting the EDGE headline to a mean was killed by only two
    tests -- a value pin (edge_median == 3.09) and a non-finite guard -- neither of
    which is about outlier robustness. And the verbatim criterion-3 poison cannot
    cover it: mfe/|mae| = 1e-4/1e-4 = exactly 1.0, an ordinary ratio sitting inside
    the base range, so it IS admitted (there is no floor on the edge path) and is
    simply not extreme. My earlier write-up blamed the floor for that survival; the
    floor is the reason for the CAPTURE half only.

    This poisons with a genuinely extreme EDGE value -- mfe=50 against mae=-0.01,
    an edge ratio of 5000 -- and asserts the same row destroys a mean, so the test
    discriminates between the estimators instead of passing under both.
    """
    base = _ordinary()
    poison = {"mfe_pct": 50.0, "mae_pct": -0.01, "realized_pnl_pct": 25.0}

    before = aggregate_exit_quality(base)
    after = aggregate_exit_quality(base + [poison])
    assert after["edge_n_defined"] == before["edge_n_defined"] + 1, (
        "the poison row must be ADMITTED, or this proves nothing about the estimator"
    )
    assert compute_edge_ratio(poison["mfe_pct"], poison["mae_pct"]) == 5000.0

    drift = abs(after["edge_median"] - before["edge_median"]) / abs(before["edge_median"])
    assert drift < 0.20, f"edge median moved {drift:.1%}"

    edges = [compute_edge_ratio(r["mfe_pct"], r["mae_pct"]) for r in base]
    mean_before = sum(edges) / len(edges)
    mean_after = (sum(edges) + 5000.0) / (len(edges) + 1)
    assert abs(mean_after - mean_before) / abs(mean_before) > 1.0, (
        "the poison does not break a mean, so this cannot tell the estimators apart"
    )


def test_deleting_the_most_extreme_row_barely_moves_the_headline(real_rows):
    """INV4. A mean fails this by ~40."""
    a_all = aggregate_exit_quality(real_rows)
    worst = min(real_rows, key=lambda r: r["capture_ratio_legacy"])
    a_less = aggregate_exit_quality([r for r in real_rows if r is not worst])
    assert abs(a_less["capture_median"] - a_all["capture_median"]) < 0.10


# ── INV5: the two surfaces must agree ────────────────────────────────

def test_performance_and_scatter_surfaces_share_one_definition(real_rows):
    """INV5. The formula used to be duplicated at three sites, so /performance and
    /mfe-mae-scatter could report different numbers for the same trades. Patching
    only the endpoint would leave that split in place."""
    import inspect

    from backend.api import paper_trading
    from backend.services import paper_round_trips, paper_trader

    for mod in (paper_trading, paper_round_trips, paper_trader):
        src = inspect.getsource(mod)
        assert "realized_pnl_pct / mfe" not in src, (
            f"{mod.__name__} still computes capture inline instead of calling "
            "perf_metrics -- the surfaces can drift apart again"
        )

    # And the shared helper is what both aggregate paths actually call.
    assert "aggregate_exit_quality" in inspect.getsource(paper_trading)
    assert "aggregate_exit_quality" in inspect.getsource(paper_round_trips)


def test_min_mfe_floor_is_surfaced_not_buried(real_rows):
    """MIN_MFE_PCT is a judgement call, not a measurement. It must travel with the
    number so a reader can audit it."""
    a = aggregate_exit_quality(real_rows)
    assert a["min_mfe_pct"] == MIN_MFE_PCT == 1.0


def test_the_floor_is_load_bearing_not_decorative(real_rows):
    """Lowering the floor to 0 must change the answer -- otherwise the floor is
    doing nothing and the near-zero denominators are back."""
    strict = aggregate_exit_quality(real_rows, min_mfe_pct=1.0)
    loose = aggregate_exit_quality(real_rows, min_mfe_pct=0.0)
    assert loose["capture_n_defined"] > strict["capture_n_defined"]
    assert loose["capture_median"] != strict["capture_median"]
