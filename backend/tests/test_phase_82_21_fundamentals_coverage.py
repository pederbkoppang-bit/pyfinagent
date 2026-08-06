"""phase-82.21 -- absent fundamentals must be distinguishable from null ones.

THE MEASURED DEFECT. `financial_reports.historical_fundamentals` holds 4798
rows / 503 tickers and every row is dated 2024-06-30 or later, so ~81% of the
2018-2025 walk-forward window has no fundamentals. Before this step, "no data
for 2018" and "this company lost money so it has no P/E" produced byte-identical
feature vectors, and the result object recorded neither.

WHY THESE GUARDS CAN FAIL (the anti-vacuity argument, which is the whole game
for criterion 2).

* A guard that only checks the UNCOVERED case passes against a fix that
  hardcodes `fundamentals_available = False`. So `test_covered_cutoff_*` drives
  the covered case and asserts REAL NUMERIC ratios, not merely "key present".
* A guard that only checks uncovered-vs-covered passes against a fix that merely
  renames `None` to `"unavailable"` without making the two states separable. So
  `test_covered_but_null_is_distinguishable_from_structural_absence` builds a
  loss-making company -- `pe_ratio` is only assigned when `net_income > 0`
  (`historical_data.py`) -- and asserts `available is True` AND
  `pe_ratio is None`. NEITHER of the other two legs can catch that failure.
* The refusal set is DERIVED by an AST rule, so `test_dependent_set_is_derived_*`
  drives the derivation against a synthetic registry rather than trusting the
  live answer, and recall-tests it with a known negative control.

All three legs go through the same production call, `build_feature_vector`.
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.backtest import fundamentals_coverage as fc
from backend.backtest.fundamentals_coverage import (
    FUNDAMENTALS_COVERAGE_START,
    coverage_start,
    fundamentals_only_feature_keys,
    label_fundamentals_dependent_strategies,
    load_snapshot,
    snapshot_drift,
    window_is_covered,
)

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "handoff" / "current" / "contract_82.21.md"
HISTORICAL_DATA = REPO / "backend" / "backtest" / "historical_data.py"

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Criterion 1 -- the measured coverage start, pinned so a claim cannot drift
# ---------------------------------------------------------------------------


def test_snapshot_records_the_measured_coverage_start():
    """CRITERION 1. The snapshot is the measurement; the constant is what
    production BELIEVES. Both must exist and agree."""
    snap = load_snapshot()
    assert snap["min_report_date"] == "2024-06-30", (
        "the measured earliest report_date; re-measure with "
        "fundamentals_coverage.measure_live_coverage() before changing this")
    assert snap["n_rows"] == 4798 and snap["n_tickers"] == 503, snap
    assert snap["measured_at"] == "2026-08-06", snap
    assert snap["report_date_bq_type"] == "STRING", (
        "if this column ever becomes DATE, the lexicographic comparisons in "
        "fundamentals_coverage must be revisited")


def test_production_constant_matches_the_snapshot_or_drift_is_reported():
    """CRITERION 1's teeth: a later coverage claim made WITHOUT re-measuring
    means the constant and the snapshot disagree, and that must fail."""
    assert FUNDAMENTALS_COVERAGE_START == coverage_start()
    assert snapshot_drift() is None, snapshot_drift()

    # And the drift check must actually be able to see a disagreement -- a
    # detector that cannot report is indistinguishable from one that finds
    # nothing.
    with patch.object(fc, "FUNDAMENTALS_COVERAGE_START", "2025-01-01"):
        drift = fc.snapshot_drift()
    assert drift is not None and "2025-01-01" in drift and "2024-06-30" in drift, drift


def test_lexicographic_min_is_only_valid_because_the_format_is_iso():
    """`report_date` is a BQ STRING, so MIN() is lexicographic. That is correct
    ONLY while every value is zero-padded ISO. Assert the format explicitly --
    and never with `isinstance(v, date)`, which can never fire on a STRING
    column (a documented dead-guard class in this project)."""
    snap = load_snapshot()
    assert _ISO.match(snap["min_report_date"]), snap["min_report_date"]
    assert _ISO.match(snap["max_report_date"]), snap["max_report_date"]
    assert snap["date_format"] == "%Y-%m-%d"
    assert snap["rows_with_non_iso_report_date"] == 0, (
        "a single non-ISO row invalidates the lexicographic MIN this module "
        "relies on")


def test_snapshot_loader_refuses_a_malformed_snapshot(tmp_path):
    """An absent or malformed snapshot must RAISE, never degrade to an
    optimistic default -- the whole point is that nobody re-measured."""
    p = tmp_path / "snap.json"
    p.write_text(json.dumps({"min_report_date": "2024-06-30"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_snapshot(p)

    p.write_text(json.dumps({
        "min_report_date": "2024-6-30", "n_rows": 1, "n_tickers": 1,
        "measured_at": "x", "date_format": "%Y-%m-%d",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not zero-padded ISO"):
        load_snapshot(p)

    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "nope.json")


def test_window_is_covered_boundaries():
    assert window_is_covered("2024-06-30") is True, "the boundary is inclusive"
    assert window_is_covered("2024-06-29") is False
    assert window_is_covered("2018-01-01") is False
    assert window_is_covered("2025-01-01") is True
    with pytest.raises(ValueError, match="zero-padded ISO"):
        window_is_covered("2024-6-30")


# ---------------------------------------------------------------------------
# Criterion 2 -- explicit unavailability, three-way and symmetric
# ---------------------------------------------------------------------------


def _provider_with(fundamentals: dict | None, *, n_days: int = 300):
    """A HistoricalDataProvider whose price history is real enough to reach the
    fundamentals block, with the fundamentals row injected."""
    from backend.backtest.historical_data import HistoricalDataProvider

    p = HistoricalDataProvider.__new__(HistoricalDataProvider)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    prices = pd.DataFrame(
        {"close": [100.0 + i * 0.1 for i in range(n_days)],
         "volume": [1_000_000.0] * n_days},
        index=idx,
    )
    p.get_point_in_time_prices = lambda t, c: prices
    p.get_point_in_time_fundamentals = lambda t, c: ([fundamentals] if fundamentals else [])
    p.get_point_in_time_macro = lambda c: {}
    p._compute_monte_carlo_var = lambda *a, **k: {}
    p._compute_anomaly_count = lambda *a, **k: 0
    p._compute_amihud_illiquidity = lambda *a, **k: 0.0
    return p


_PROFITABLE = {
    "shares_outstanding": 1_000_000.0,
    "total_revenue": 500_000.0,
    "net_income": 50_000.0,
    "total_debt": 200_000.0,
    "total_equity": 400_000.0,
    "total_assets": 900_000.0,
    "operating_cash_flow": 80_000.0,
    "sector": "Technology",
    "industry": "Software",
}
#: Same company, loss-making. `pe_ratio` is only assigned when net_income > 0,
#: so this row yields a GENUINE null -- the case that separates "absent" from
#: "present but null".
_LOSS_MAKING = dict(_PROFITABLE, net_income=-50_000.0)


def test_uncovered_cutoff_reports_fundamentals_explicitly_unavailable():
    """CRITERION 2, leg (a): structural absence."""
    fv = _provider_with(None).build_feature_vector("AAPL", "2020-06-30")
    assert fv["fundamentals_available"] is False
    F = fundamentals_only_feature_keys()
    present = sorted(k for k in F if k in fv)
    assert present == [], f"no fundamentals-derived key may be present: {present}"


def test_covered_cutoff_reports_available_with_real_values():
    """CRITERION 2, leg (b): this is the leg that kills a hardcoded False."""
    fv = _provider_with(_PROFITABLE).build_feature_vector("AAPL", "2025-03-31")
    assert fv["fundamentals_available"] is True
    for key in ("pe_ratio", "roe", "debt_equity", "profit_margin"):
        assert key in fv, key
        assert isinstance(fv[key], float), f"{key} must be a real number, got {fv[key]!r}"
    assert fv["debt_equity"] == pytest.approx(0.5)
    assert fv["profit_margin"] == pytest.approx(0.1)


def test_covered_but_null_is_distinguishable_from_structural_absence():
    """CRITERION 2, leg (c) -- THE load-bearing one.

    A loss-making company has fundamentals but no P/E. Before this step that was
    byte-identical to having no fundamentals at all. Neither leg (a) nor leg (b)
    can catch a fix that merely renames None; this one can, because it asserts
    the two states differ while `pe_ratio` is None in BOTH.
    """
    absent = _provider_with(None).build_feature_vector("AAPL", "2020-06-30")
    null = _provider_with(_LOSS_MAKING).build_feature_vector("AAPL", "2025-03-31")

    assert absent.get("pe_ratio") is None
    assert null.get("pe_ratio") is None, (
        "fixture precondition: a loss-making company must yield a genuine null "
        "pe_ratio, or this guard is testing nothing")

    assert absent["fundamentals_available"] is False
    assert null["fundamentals_available"] is True
    assert absent["fundamentals_available"] != null["fundamentals_available"], (
        "the two states must be distinguishable -- this is criterion 2 verbatim")

    # And the genuine-null case still carries the fundamentals it DOES have.
    assert null["total_equity"] == pytest.approx(400_000.0)
    assert null["roe"] == pytest.approx((-50_000.0 * 4) / 400_000.0)


def test_flag_is_present_on_every_return_path_including_the_early_return():
    """`build_feature_vector` returns early when price history is too short.
    The flag must still be present there, or `fv.get(...)` has three outcomes
    (True / False / absent) and 'absent' reintroduces the ambiguity."""
    fv = _provider_with(_PROFITABLE, n_days=5).build_feature_vector("AAPL", "2025-03-31")
    assert "price_at_analysis" not in fv, "fixture precondition: the early return"
    assert fv["fundamentals_available"] is False


def test_flag_is_not_a_model_input():
    """`fundamentals_available` is a perfect proxy for `date >= 2024-07`. As a
    model input it is a regime dummy and the classifier learns the coverage
    boundary instead of the economics."""
    from backend.backtest.backtest_engine import _NUMERIC_FEATURES

    assert "fundamentals_available" not in _NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Criterion 3 -- refuse for the dependent set, record for everyone else
# ---------------------------------------------------------------------------


def test_dependent_set_is_derived_not_hardcoded():
    """The rule must produce the answer from the source. Driven against a
    SYNTHETIC registry so this fails if the derivation silently returns a
    constant."""
    F = fundamentals_only_feature_keys()
    assert len(F) == 17, f"measured 17 fundamentals-only keys, got {len(F)}: {sorted(F)}"
    assert {"pe_ratio", "roe", "debt_equity", "quality_score"} <= F

    derived = label_fundamentals_dependent_strategies()
    assert derived == {"qarp"}, (
        f"derived label-fundamentals-dependent set, got {sorted(derived)}")

    # RECALL TEST: a rule that finds nothing looks identical to a rule that sees
    # nothing. Point it at a synthetic registry whose answer is known.
    fake = {"reads_pe": "_compute_qarp_label", "reads_none": "_compute_triple_barrier_label"}
    got = label_fundamentals_dependent_strategies(registry=fake)
    assert got == {"reads_pe"}, got


def test_derivation_follows_a_transitive_hop_between_compute_helpers(tmp_path):
    """The docstring promises transitivity; this drives it.

    THE DEFECT THIS GUARD EXISTS FOR (found by the 82.21 Q/A, not by me): an
    earlier revision skipped every callee whose name started with `_compute`.
    EVERY label function in STRATEGY_REGISTRY is named `_compute_*`, so that
    exclusion matched the module's dominant helper convention and could only
    produce FALSE NEGATIVES -- and a false negative here lets a
    fundamentals-dependent strategy run on an uncovered window, which is the
    exact failure the gate exists to prevent.

    The hop below is `_compute_outer_label` -> `_compute_inner_helper`, i.e.
    precisely the shape the old exclusion dropped. No existing guard drove a
    transitive hop at all, which is why the bug survived a 16-mutant matrix.
    """
    engine = tmp_path / "engine.py"
    engine.write_text(
        "def _compute_inner_helper(fv):\n"
        "    return fv.get('pe_ratio')\n"
        "\n"
        "def _compute_outer_label(self, fv):\n"
        "    return _compute_inner_helper(fv)\n"
        "\n"
        "def _compute_clean_label(self, fv):\n"
        "    return fv.get('price_at_analysis')\n",
        encoding="utf-8",
    )
    got = label_fundamentals_dependent_strategies(
        registry={"hops": "_compute_outer_label", "clean": "_compute_clean_label"},
        engine_path=engine,
    )
    assert got == {"hops"}, (
        f"the transitive hop must be followed and the clean one left alone; got {got}")


def test_dependent_set_excludes_the_negative_controls():
    """`triple_barrier` never calls build_feature_vector at all;
    `stretch_regime` LOOKS fundamentals-ish but reads only price/vol. Eyeballing
    would plausibly get the second one wrong; the rule must not."""
    derived = label_fundamentals_dependent_strategies()
    assert "triple_barrier" not in derived
    assert "stretch_regime" not in derived
    assert "mean_reversion" not in derived


def test_derivation_refuses_to_return_a_vacuous_set(tmp_path):
    """If the source stops matching the rule, RAISE -- do not return an empty
    set, which would silently disable the gate for every strategy."""
    stub = tmp_path / "historical_data.py"
    stub.write_text(
        "class P:\n"
        "    def build_feature_vector(self, ticker, cutoff_date):\n"
        "        features = {}\n"
        "        return features\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no `if fundamentals:` block"):
        fundamentals_only_feature_keys(stub)


def test_derivation_refuses_an_EMPTY_derived_set(tmp_path):
    """The second vacuity path, and the one a mutation run caught me missing.

    The `if fundamentals:` block can still be present while the derived set F
    comes out empty -- e.g. every key it assigns is also assigned outside it, so
    `inside - outside` is the empty set. An empty F makes every downstream
    membership test vacuously false and silently disables the refusal gate for
    EVERY strategy. Measured: replacing the `raise` here with `pass` left the
    rest of this suite green, so without this leg the guard could not see it.
    """
    stub = tmp_path / "historical_data.py"
    stub.write_text(
        "class P:\n"
        "    def build_feature_vector(self, ticker, cutoff_date):\n"
        "        features = {}\n"
        "        features['pe_ratio'] = None\n"   # assigned OUTSIDE too
        "        if fundamentals:\n"
        "            features['pe_ratio'] = 1.0\n"
        "        return features\n",
        encoding="utf-8",
    )
    # Precondition: the block IS present, so this is the empty-F path and not
    # the missing-block path already covered above.
    src = stub.read_text(encoding="utf-8")
    assert "if fundamentals:" in src, "fixture precondition"
    with pytest.raises(ValueError, match="EMPTY"):
        fundamentals_only_feature_keys(stub)


def _engine(strategy: str, start_date: str):
    from backend.backtest.backtest_engine import BacktestEngine

    e = BacktestEngine.__new__(BacktestEngine)
    e.strategy = strategy
    e.scheduler = MagicMock()
    e.scheduler.start_date = dt.date.fromisoformat(start_date)
    return e


def test_dependent_strategy_refuses_on_an_uncovered_window():
    """CRITERION 3, the refuse half -- driven through the engine's real method."""
    with pytest.raises(ValueError, match="REFUSED"):
        _engine("qarp", "2018-01-01")._preload_fundamentals_and_record()


def test_dependent_strategy_runs_on_a_covered_window():
    """The refusal must be caused by the WINDOW, not by the strategy name --
    otherwise the gate would block qarp forever and pass for the wrong reason."""
    rec = _engine("qarp", "2025-01-01")._preload_fundamentals_and_record()
    assert rec["fundamentals"] is True
    assert rec["fundamentals_coverage_start"] == FUNDAMENTALS_COVERAGE_START


def test_independent_strategy_records_a_warning_instead_of_refusing():
    """CRITERION 3, the record half. Every strategy is FEATURE-dependent via
    _NUMERIC_FEATURES, so an uncovered run must be labelled even when it is
    allowed to proceed."""
    rec = _engine("triple_barrier", "2018-01-01")._preload_fundamentals_and_record()
    assert rec["fundamentals"] is False
    assert rec["fundamentals_window_start"] == "2018-01-01"
    assert rec["fundamentals_label_dependent"] == ["qarp"]


def test_covered_window_is_not_labelled_unavailable():
    """So the gate cannot pass by always reporting False."""
    rec = _engine("triple_barrier", "2025-01-01")._preload_fundamentals_and_record()
    assert rec["fundamentals"] is True


def test_result_default_carries_fundamentals_and_analytics_surfaces_it():
    """The record must reach BOTH report['data_availability'] and
    report['analytics'] -- two consumers read only the latter, so a top-level
    key alone leaves an uncovered run looking normal to exactly the code that
    compares candidates."""
    from backend.backtest.backtest_engine import BacktestResult

    assert BacktestResult().data_availability == {"macro": True, "fundamentals": True}

    src = (REPO / "backend" / "backtest" / "analytics.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert 'report["analytics"]["fundamentals_available"]' in code, (
        "the analytics surface must carry it, not only the top level")


def test_engine_calls_the_recorder_on_the_real_run_path():
    """A method nothing calls is not a gate. Assert the production run path
    invokes it, structurally."""
    src = (REPO / "backend" / "backtest" / "backtest_engine.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert code.count("self._preload_fundamentals_and_record()") == 1, (
        "run_backtest must call the recorder exactly once")
    tree = ast.parse(src)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run_backtest")
    calls = [n for n in ast.walk(run)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_preload_fundamentals_and_record"]
    assert len(calls) == 1, "the call must live inside run_backtest itself"


# ---------------------------------------------------------------------------
# Criterion 4 -- the operator's decision, verbatim
# ---------------------------------------------------------------------------

OPERATOR_INSTRUCTION = (
    "82.21 fundamentals source: free only -> either accept that fundamentals-\n"
    "  dependent strategies are evaluable from 2024-07 on, or build SEC EDGAR XBRL.\n"
    "  Do NOT adopt a paid source."
)


def test_operator_decision_recorded_verbatim_in_the_artifact():
    """CRITERION 4."""
    assert CONTRACT.is_file(), f"the step artifact is missing: {CONTRACT}"
    text = CONTRACT.read_text(encoding="utf-8")
    assert OPERATOR_INSTRUCTION in text, (
        "the operator's verbatim instruction must appear in the step artifact")
    for option in ("paid source", "evaluable from 2024-07", "SEC EDGAR XBRL"):
        assert option.lower() in text.lower(), f"the artifact must address {option!r}"
    assert "**DECISION: build SEC EDGAR XBRL.**" in text, (
        "the artifact must state which option was chosen, not merely list them")
