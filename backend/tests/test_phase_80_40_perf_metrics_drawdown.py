"""phase-80.40 -- the paper drawdown field that never existed.

MEASURED 2026-07-26 against a healthy backend: GET /api/paper-trading/performance
returned sharpe_ratio=3.44 but NO `max_drawdown_pct` key at all, so the cockpit's
drawdown row had never measured anything since it was written. Pre-80.36 the
absence rendered an emerald SAFE via `?? 0` (0 > -10); 80.36's presence check
turned the same absence into an honest NO DATA and exposed the gap.

FILENAME IS LOAD-BEARING. The step's immutable verification command is
`pytest backend/tests/ -q -k 'perf_metrics or drawdown'`, and pytest's `-k`
matches the MODULE FILENAME as well as function names (proved: `-k dod4` selects
72/72 in test_dod4_tier1_coverage_investment.py, none of whose function names
contain "dod4"). A file named e.g. test_phase_80_40_killswitch_indicator.py would
NOT be selected and every assertion below would pass vacuously. Keep
`perf_metrics` and `drawdown` in the name.

WHAT THESE TESTS EXIST TO CATCH, in the order the mutation matrix runs them:
  1. A stub / reverted computation returning 0.0 (criterion 3 + criterion 6).
  2. A SIGN INVERSION. paper_go_live_gate._snapshot_max_dd_pct is the only other
     snapshots->max-drawdown adapter in the repo and it returns POSITIVE
     magnitude, so it is the most tempting thing to import -- and importing it
     makes `maxDd > -10` true at EVERY depth, rendering emerald SAFE forever.
  3. A missing ASC sort. This exact series has caused TWO production incidents
     (phase-47.4 60.08% phantom, phase-66.2 phantom -61.51% P1 page on a book UP
     20%) because get_paper_snapshots is ORDER BY snapshot_date DESC.
  4. 0.0-instead-of-None on degraded paths, which reintroduces the fabricated
     SAFE one layer deeper and behind a passing test.
  5. A nan reaching the payload, which does not merely leak -- FastAPI refuses to
     serialize it and 500s the WHOLE cockpit payload.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.backtest.analytics import compute_max_drawdown  # noqa: E402
from backend.services.paper_go_live_gate import _snapshot_max_dd_pct  # noqa: E402
from backend.services.perf_metrics import (  # noqa: E402
    compute_max_drawdown_from_snapshots,
)


def _snap(date: str, nav):
    return {"snapshot_date": date, "total_nav": nav}


# A fixed NAV series with an arithmetically KNOWN answer: the peak is 150 and the
# trough after it is 105, so (105 - 150) / 150 * 100 == -30.0 exactly. The later
# rise to 160 sets a new high-water mark and must NOT reduce the historical max.
KNOWN_DD_ASC = [
    _snap("2026-01-01", 100.0),
    _snap("2026-01-02", 150.0),
    _snap("2026-01-03", 105.0),
    _snap("2026-01-04", 160.0),
]
KNOWN_DD_PCT = -30.0

# The phantom-drawdown fixture: monotonically GROWING NAV. Read chronologically
# this is a 0% drawdown; read backwards it looks like a ~35% crash.
GROWING_ASC = [_snap(f"2026-06-{d:02d}", 20000.0 + d * 500.0) for d in range(1, 21)]


# ── criterion 3: pinned against a fixed series with a known drawdown ──


def test_known_nav_series_pins_the_exact_drawdown():
    """MUST FAIL against a stub returning 0.0 (and against any sign flip)."""
    assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) == KNOWN_DD_PCT


def test_a_new_high_water_mark_does_not_erase_the_historical_max():
    """MAX, not CURRENT: the series ends at its all-time high (160 > 150) yet the
    published figure must still be the -30% trough. A CURRENT-drawdown
    implementation returns 0.0 here and would pass a naive presence check."""
    assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) == KNOWN_DD_PCT
    assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) != 0.0


def test_deeper_trough_later_in_the_series_wins():
    """Guards a first-drawdown-only (early-return) implementation."""
    series = KNOWN_DD_ASC + [_snap("2026-01-05", 80.0)]
    # Peak 160 -> trough 80 == -50%, deeper than the earlier -30%.
    assert compute_max_drawdown_from_snapshots(series) == -50.0


# ── the sign convention, pinned by assertion rather than docstring ──


def test_sign_is_negative_percent():
    v = compute_max_drawdown_from_snapshots(KNOWN_DD_ASC)
    assert v is not None
    assert -100.0 <= v <= 0.0, f"drawdown must be a negative percent, got {v!r}"


def test_sign_is_the_exact_negation_of_the_positive_magnitude_sibling():
    """THE INVERSION GUARD. paper_go_live_gate._snapshot_max_dd_pct measures the
    same thing with the opposite sign. If someone imports or copies it here, this
    is the assertion that fires -- nothing else in the suite would."""
    ours = compute_max_drawdown_from_snapshots(KNOWN_DD_ASC)
    sibling = _snapshot_max_dd_pct(KNOWN_DD_ASC)
    assert sibling == pytest.approx(30.0), "fixture drifted; sibling is positive magnitude"
    assert ours == pytest.approx(-sibling)


def test_sign_and_value_match_the_backtest_convention():
    """Reality-gap comparability: the backtest side publishes
    max_drawdown_pct via backtest_engine._max_drawdown ->
    analytics.compute_max_drawdown. Feeding the same curve to that primitive must
    give the same number, or the cockpit's 'Max DD: paper / backtest' row is
    comparing two different definitions on one axis."""
    navs = np.array([s["total_nav"] for s in KNOWN_DD_ASC], dtype=float)
    assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) == pytest.approx(
        round(compute_max_drawdown(navs), 2)
    )


# ── order invariance: the guard that would have caught 47.4 AND 66.2 ──


def test_desc_input_gives_an_identical_result_to_asc():
    asc = compute_max_drawdown_from_snapshots(KNOWN_DD_ASC)
    desc = compute_max_drawdown_from_snapshots(list(reversed(KNOWN_DD_ASC)))
    assert asc == desc == KNOWN_DD_PCT


def test_growing_nav_in_desc_order_is_not_a_phantom_crash():
    """phase-47.4 / phase-66.2 regression. bq.get_paper_snapshots is ORDER BY
    snapshot_date DESC; walking it backwards read growth as a crash and paged a
    -61.51% P1 against a book UP 20%."""
    desc = list(reversed(GROWING_ASC))
    assert compute_max_drawdown_from_snapshots(desc) == 0.0
    assert compute_max_drawdown_from_snapshots(GROWING_ASC) == 0.0


def test_shuffled_input_is_still_chronological():
    import random

    shuffled = list(KNOWN_DD_ASC)
    random.Random(80_40).shuffle(shuffled)
    assert compute_max_drawdown_from_snapshots(shuffled) == KNOWN_DD_PCT


# ── None (never 0.0) on every degraded path ──


@pytest.mark.parametrize(
    "series, why",
    [
        ([], "empty series"),
        ([_snap("2026-01-01", 100.0)], "single row -- nothing to compare against"),
        ([_snap("2026-01-01", 0.0), _snap("2026-01-02", 0.0)], "all-NAV-zero (nan upstream)"),
        ([_snap("2026-01-01", None), _snap("2026-01-02", None)], "NAV column missing"),
        ([{"snapshot_date": "2026-01-01"}, {"snapshot_date": "2026-01-02"}], "no nav_key at all"),
        ([_snap("2026-01-01", 100.0), _snap("2026-01-02", 0.0)], "one zero leaves <2 usable"),
    ],
)
def test_degraded_paths_return_none_never_zero(series, why):
    """0.0 IS NOT A SAFE FALLBACK HERE. phase-80.36 deliberately discriminates on
    PRESENCE, so a numeric 0.0 renders a real emerald SAFE -- i.e. a 0.0 fallback
    fabricates the reassuring verdict this whole step exists to delete.
    analytics.compute_max_drawdown returns 0.0 for empty AND single-element input,
    so the adapter, not the primitive, has to make this distinction."""
    got = compute_max_drawdown_from_snapshots(series)
    assert got is None, f"{why}: expected None, got {got!r}"


def test_non_finite_navs_are_dropped_and_never_reach_the_result():
    """nan/inf NAV rows must not poison the answer. compute_max_drawdown returns
    nan if ANY element is nan, and +inf yields a spurious -100.0."""
    poisoned = [
        _snap("2026-01-01", 100.0),
        _snap("2026-01-02", float("nan")),
        _snap("2026-01-03", 150.0),
        _snap("2026-01-04", float("inf")),
        _snap("2026-01-05", 105.0),
        _snap("2026-01-06", 160.0),
    ]
    got = compute_max_drawdown_from_snapshots(poisoned)
    assert got is not None
    assert math.isfinite(got)
    assert got == KNOWN_DD_PCT, "the nan/inf rows should be dropped, not absorbed"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_result_from_the_primitive_returns_none(monkeypatch, bad):
    """DEFENSE-IN-DEPTH, GIVEN A REAL FAILURE MODE.

    The `nav > 0 and isfinite` filter above means the second guard
    (`if not math.isfinite(dd_pct)`) is unreachable through the public argument
    -- and a guard that cannot fail does not count. So this test drives the
    primitive directly: analytics.compute_max_drawdown DOES return nan for an
    all-zero series and for any series containing a nan, so a future edit that
    loosens the NAV filter (or a change inside analytics.py) makes this path
    live. Without this test, deleting the isfinite guard passes the whole suite
    -- measured: mutation M6 survived 64/64 before this was added."""
    monkeypatch.setattr(
        "backend.services.perf_metrics.compute_max_drawdown", lambda _arr: bad
    )
    assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) is None


def test_a_nan_on_the_payload_really_does_take_the_whole_endpoint_down():
    """The premise behind returning None rather than nan, measured on THIS
    endpoint rather than asserted from documentation: FastAPI's serializer
    rejects nan ("Out of range float values are not JSON compliant"), so one bad
    NAV row would 500 the entire cockpit payload -- nav, sharpe_ratio and
    round_trip_summary included. Per project memory every 500 here is reported to
    the operator as "backend down"."""
    bq = _FakeBQ(list(KNOWN_DD_ASC))
    client = _client_for(KNOWN_DD_ASC)
    with patch("backend.api.paper_trading.get_bq_client", lambda: bq), patch(
        "backend.api.paper_trading.compute_max_drawdown_from_snapshots",
        lambda *a, **k: float("nan"),
    ):
        with pytest.raises(ValueError, match="not JSON compliant"):
            client.get("/api/paper-trading/performance")


def test_unparseable_nav_values_are_skipped_not_raised():
    series = [
        _snap("2026-01-01", "not-a-number"),
        _snap("2026-01-02", 100.0),
        _snap("2026-01-03", 150.0),
        _snap("2026-01-04", 105.0),
        _snap("2026-01-05", 160.0),
    ]
    assert compute_max_drawdown_from_snapshots(series) == KNOWN_DD_PCT


def test_negative_zero_is_normalized():
    """`-0.0` in JSON is legal but reads as a bug in the cockpit.

    VACUOUS AS ORIGINALLY WRITTEN (2026-07-26 adversarial verification): GROWING_ASC
    is strictly monotonically increasing, so the underlying primitive returns an
    exact `0.0` for it -- never `-0.0` -- and deleting the normalize line left this
    test green. A tiny real dip is what actually produces `-0.0`: on the real book
    (nav ~23838), a 19-cent-and-16-cent wobble measures
    round(-0.0006711927430634515, 2) == -0.0. Kept the monotonic case too, since
    that path (0.0 from the start, never entering the -0.0 branch at all) is a
    distinct code path worth pinning."""
    tiny_dip = [
        _snap("2026-07-24", 23838.16),
        _snap("2026-07-25", 23838.00),
        _snap("2026-07-26", 23838.16),
    ]
    got = compute_max_drawdown_from_snapshots(tiny_dip)
    assert got == 0.0
    assert math.copysign(1.0, got) > 0, "expected +0.0, got -0.0"

    got_monotonic = compute_max_drawdown_from_snapshots(GROWING_ASC)
    assert got_monotonic == 0.0
    assert math.copysign(1.0, got_monotonic) > 0


def test_the_caller_list_is_not_mutated():
    """The /performance handler reuses this same list for Sharpe and total_cost;
    an in-place sort here would silently reorder those computations."""
    series = list(reversed(KNOWN_DD_ASC))
    before = [s["snapshot_date"] for s in series]
    compute_max_drawdown_from_snapshots(series)
    assert [s["snapshot_date"] for s in series] == before


# ── criterion 1: the field is actually on the payload ──


class _FakeBQ:
    """Only what GET /performance touches."""

    def __init__(self, snapshots: list[dict]):
        self._snapshots = snapshots

    def get_paper_portfolio(self, portfolio_id: str = "default"):
        return {
            "portfolio_id": "default",
            "starting_capital": 100.0,
            "total_nav": 160.0,
            "total_pnl_pct": 60.0,
            "benchmark_return_pct": 10.0,
        }

    def get_paper_snapshots(self, limit: int = 365):
        return list(self._snapshots[:limit])

    def get_paper_trades(self, limit: int = 100):
        return []


def _client_for(snapshots: list[dict]) -> TestClient:
    from backend.api.paper_trading import router
    from backend.services.api_cache import get_api_cache

    # /performance caches for 120s under a fixed key; a stale entry from another
    # test in the same process would make this assert nothing.
    get_api_cache().invalidate("paper:performance")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_performance_endpoint_returns_a_numeric_negative_max_drawdown_pct():
    bq = _FakeBQ(list(reversed(KNOWN_DD_ASC)))  # DESC, exactly as BQ returns it
    client = _client_for(KNOWN_DD_ASC)
    with patch("backend.api.paper_trading.get_bq_client", lambda: bq):
        r = client.get("/api/paper-trading/performance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "max_drawdown_pct" in body, f"field absent -- the 80.40 defect. keys={sorted(body)}"
    assert "sharpe_ratio" in body, "criterion 1 wants it recorded ALONGSIDE sharpe_ratio"
    got = body["max_drawdown_pct"]
    assert isinstance(got, (int, float)) and not isinstance(got, bool)
    assert got == KNOWN_DD_PCT
    assert -100.0 <= got <= 0.0


def test_performance_endpoint_stays_200_when_the_series_is_degenerate():
    """A nan reaching JSONResponse raises 'Out of range float values are not JSON
    compliant' and 500s the WHOLE payload -- taking nav, sharpe_ratio and
    round_trip_summary down with it. Per project memory every 500 on this cockpit
    misreports to the operator as 'backend down'. This is an outage guard."""
    bq = _FakeBQ([_snap("2026-01-01", 0.0), _snap("2026-01-02", 0.0)])
    client = _client_for([])
    with patch("backend.api.paper_trading.get_bq_client", lambda: bq):
        r = client.get("/api/paper-trading/performance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_drawdown_pct"] is None, "absence must propagate, not become 0.0"
    assert "sharpe_ratio" in body, "the rest of the payload must survive"


def test_the_endpoint_does_not_add_a_second_snapshot_query():
    """The series is already in hand; a self-fetching helper would add a BQ round
    trip (~1.2s measured) on every cache miss of the endpoint that feeds the
    entire cockpit."""
    bq = _FakeBQ(list(KNOWN_DD_ASC))
    calls: list[int] = []
    inner = bq.get_paper_snapshots

    def counting(limit: int = 365):
        calls.append(limit)
        return inner(limit=limit)

    bq.get_paper_snapshots = counting  # type: ignore[method-assign]
    client = _client_for(KNOWN_DD_ASC)
    with patch("backend.api.paper_trading.get_bq_client", lambda: bq):
        r = client.get("/api/paper-trading/performance")
    assert r.status_code == 200, r.text
    assert len(calls) == 1, f"expected exactly 1 get_paper_snapshots call, got {calls}"


# ── criterion 4: the two ladders are reconciled, and NEITHER value moved ──


def test_the_two_drawdown_ladders_keep_their_documented_values():
    """80.40 reconciles the LABEL, never a threshold. Ladder A (kill switch,
    HALTS trading) is 10% positive-magnitude trailing DD; ladder B (MCP risk
    constraints, BLOCKS new BUYs) is -15% with -10 derisk / -5 warn. They are
    different mechanisms and legitimately differ. Independently pinned by
    test_phase_75_mcp_truth.py::test_thresholds_are_unchanged -- duplicated here
    so a future edit to THIS step's field cannot drift them."""
    from backend.config.settings import get_settings

    s = get_settings()
    assert float(s.paper_trailing_dd_limit_pct) == 10.0
    assert float(s.paper_daily_loss_limit_pct) == 4.0

    src = (_ROOT / "backend" / "agents" / "mcp_servers" / "signals_server.py").read_text(
        encoding="utf-8"
    )
    assert '"max_drawdown_pct": -15.0,' in src
    assert '"drawdown_derisk_pct": -10.0,' in src
    assert '"drawdown_warning_pct": -5.0,' in src


def test_the_cockpit_row_no_longer_claims_to_be_the_kill_switch():
    """The row renders ladder B's -15/-10 thresholds, so labelling it 'Kill
    switch' attributed ladder A's mechanism to ladder B's numbers -- and the
    field is ALL-TIME MAX drawdown, which the kill switch never trips on
    (it trips on CURRENT trailing DD)."""
    src = (
        _ROOT / "frontend" / "src" / "components" / "paper-trading" / "cockpit-helpers.tsx"
    ).read_text(encoding="utf-8")
    assert '<span className="text-slate-400">Max drawdown (-15%)</span>' in src
    assert '<span className="text-slate-400">Kill switch (-15%)</span>' not in src


def test_phase_80_45_precision_floor_is_documented_and_holds():
    """phase-80.45 disposition (a): ACCEPT the 2dp floor, narrow the CLAIM.

    A real decline shallower than 0.005% rounds to -0.0 and publishes as 0.0, so
    a published 0.0 cannot mean "never fell at all". Rather than widen the payload
    precision -- which would split paper/backtest comparability, the thing this
    function exists to preserve -- the docstring now says "to 2 decimal places".

    This test exists so the DOCSTRING AND THE CODE CANNOT DRIFT APART: it asserts
    both the documented qualifier and the behaviour it describes. Criterion 3.
    """
    from backend.services import perf_metrics

    # (1) The claim is qualified in the docstring.
    doc = perf_metrics.compute_max_drawdown_from_snapshots.__doc__ or ""
    assert "TO 2 DECIMAL PLACES" in doc, (
        "the 0.0-means-never claim must be qualified; a bare 0.0 over-claims"
    )

    # (2) The behaviour that qualifier describes, using the REAL measured dip:
    #     19 cents on a $23,838.19 NAV -> -0.00067...% -> publishes as 0.0.
    snaps = [
        {"snapshot_date": "2026-07-24", "total_nav": 23838.19},
        {"snapshot_date": "2026-07-25", "total_nav": 23838.00},
    ]
    got = perf_metrics.compute_max_drawdown_from_snapshots(snaps)
    assert got == 0.0, f"sub-precision dip should floor to 0.0, got {got!r}"

    # (3) ...and it is a TRUE zero, never a negative zero (the -0.0 normalization
    #     this step was forbidden to remove).
    import math
    assert not math.copysign(1, got) < 0, "a negative zero must never reach the payload"

    # (4) A decline ABOVE the floor still reports, so the floor is a floor and not
    #     a clamp that swallows real drawdowns.
    real = perf_metrics.compute_max_drawdown_from_snapshots([
        {"snapshot_date": "2026-07-24", "total_nav": 23838.19},
        {"snapshot_date": "2026-07-25", "total_nav": 23000.00},
    ])
    assert real is not None and real < -3.0, f"a real 3.5% decline must report, got {real!r}"
