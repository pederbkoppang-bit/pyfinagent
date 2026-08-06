"""phase-82.43 -- a bare `if macro:` hid THREE states behind one boolean.

WHAT THE STEP DESCRIBES, and what the research gate found instead. The step says
an EMPTY macro dict silently drops all six macro features. True, and measured
on these fixtures through the production `_NUMERIC_FEATURES` filter: the matrix
goes from 18 features to 12, and the narrower one reaches
`GradientBoostingClassifier.fit` -- nothing re-pads it.

But the LIVE case is worse and the step does not name it. `cache.cached_macro`
resolves each series INDEPENDENTLY, so `if macro:` is TRUE while individual
series are missing. Measured on the real biweekly grid 2018-01-01..2019-12-31
(n=53 cutoffs): 1 empty, 3 PARTIAL (2 of 6 series), 49 full. On a partial
cutoff the columns SURVIVE -- most rows have them -- and the missing cells are
median-imputed downstream. Silent fabrication, no trace.

THAT IS WHY THIS IS A COUNT AND NOT A BOOLEAN, and why the phase-82.21 template
deliberately does NOT carry over here: a boolean `macro_available` reads True on
exactly the degraded cutoffs. `test_partial_macro_is_distinguishable_from_both`
is the leg a boolean cannot pass, and it is this file's whole reason for
existing.

WHY THE SIX KEYS ARE NOT EMITTED AS EXPLICIT NULLS (criterion 1 offers that as
an alternative and it is measurably WORSE): an all-NaN column survives the
`feature_cols` filter, `X.median()` is NaN, `fillna(train_medians)` is a no-op,
and the trailing `fillna(0)` makes it a CONSTANT 0.0 -- in-range for
`yield_curve_spread` and `cpi_yoy`. It converts a visible width change into an
invisible fabricated constant. `test_explicit_nulls_would_have_been_worse` pins
that measurement so nobody "improves" the fix into a regression.

FIXTURE HAZARD, from the gate: `cached_macro` memoises, so a test that calls it
twice for one cutoff never re-exercises the branch. Every leg here drives
`build_feature_vector` with an injected provider instead, so memoisation cannot
mask a state -- and `test_cached_macro_memoises` pins the hazard itself.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend.backtest.historical_data import (
    _MACRO_FEATURES,
    _MACRO_SERIES,
    HistoricalDataProvider,
)

REPO = Path(__file__).resolve().parents[2]
HD = REPO / "backend" / "backtest" / "historical_data.py"
CACHE = REPO / "backend" / "backtest" / "cache.py"

_FULL = {s: {"value": float(i + 1)} for i, s in enumerate(_MACRO_SERIES)}
#: 2 of 6 -- the shape measured live on the real grid (fast-publishing series
#: resolve, slow ones like CPIAUCSL do not).
_PARTIAL = {"DGS10": {"value": 3.0}, "T10Y2Y": {"value": 0.5}}


def _provider(macro: dict, *, n_days: int = 300, fundamentals=None):
    p = HistoricalDataProvider.__new__(HistoricalDataProvider)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    prices = pd.DataFrame(
        {"close": [100.0 + i * 0.1 for i in range(n_days)],
         "volume": [1_000_000.0] * n_days},
        index=idx,
    )
    p.get_point_in_time_prices = lambda t, c: prices
    p.get_point_in_time_fundamentals = lambda t, c: ([fundamentals] if fundamentals else [])
    p.get_point_in_time_macro = lambda c: macro
    p._compute_monte_carlo_var = lambda *a, **k: {}
    p._compute_anomaly_count = lambda *a, **k: 0
    p._compute_amihud_illiquidity = lambda *a, **k: 0.0
    return p


def _fv(macro: dict, **kw):
    return _provider(macro, **kw).build_feature_vector("AAPL", "2025-03-31")


# ---------------------------------------------------------------------------
# Criterion 1 -- the absence is recorded explicitly
# ---------------------------------------------------------------------------


def test_empty_macro_records_the_absence_explicitly():
    """CRITERION 1. Against the pre-fix tree there was NO signal at all -- the
    six keys were simply absent and `macro_series_count` did not exist."""
    fv = _fv({})
    assert fv["macro_series_count"] == 0
    present = [k for k in _MACRO_FEATURES if k in fv]
    assert present == [], f"no macro feature may be present: {present}"


def test_full_macro_is_unaffected():
    """CRITERION 3. The fix must not pass by always reporting degradation."""
    fv = _fv(_FULL)
    assert fv["macro_series_count"] == len(_MACRO_SERIES) == 6
    for key in _MACRO_FEATURES:
        assert key in fv, key
        assert isinstance(fv[key], float), f"{key} must carry a real value, got {fv[key]!r}"


def test_partial_macro_is_distinguishable_from_both():
    """THE LEG A BOOLEAN CANNOT PASS, and the reason this file exists.

    A partial cutoff is the live degraded case: `if macro:` is TRUE, the columns
    survive, and the missing cells get median-imputed downstream. A boolean
    `macro_available` reports True here -- identical to a fully-covered window.
    The count separates all three states.
    """
    empty, partial, full = _fv({}), _fv(_PARTIAL), _fv(_FULL)
    assert (empty["macro_series_count"],
            partial["macro_series_count"],
            full["macro_series_count"]) == (0, 2, 6)

    # Precondition: partial really is the trap -- a boolean would collapse it
    # into the full case.
    assert bool(_PARTIAL) is True and bool(_FULL) is True, "fixture precondition"
    assert partial["macro_series_count"] != full["macro_series_count"], (
        "partial must not read as fully covered")
    assert partial["macro_series_count"] != empty["macro_series_count"]

    # And the degraded series really are the missing ones.
    assert partial["treasury_10y"] == 3.0 and partial["yield_curve_spread"] == 0.5
    assert partial["cpi_yoy"] is None and partial["fed_funds_rate"] is None


def test_count_is_present_on_every_return_path():
    """Including the short-price early return. An ABSENT count would be a third
    state and reintroduce the ambiguity the count removes."""
    fv = _fv(_FULL, n_days=5)
    assert "price_at_analysis" not in fv, "fixture precondition: the early return"
    assert fv["macro_series_count"] == 0


def test_count_is_not_a_model_input():
    """Macro coverage is a step function at ~2018-02-20, so as a model input the
    count is an even more degenerate regime dummy than 82.21's flag. Note this
    deliberately departs from Perez-Lebel 2022 (missingness-mask-as-feature);
    the departure is argued in the contract, not silent."""
    from backend.backtest.backtest_engine import _NUMERIC_FEATURES

    assert "macro_series_count" not in _NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Criterion 2 -- the feature count REACHING THE MODEL, measured both ways
# ---------------------------------------------------------------------------


def _matrix_width(rows: list[dict]) -> tuple[int, list[str]]:
    """The production transform at `_build_training_data`: DataFrame ->
    feature_cols filter -> X. Reproduced on the same two lines so the widths
    below are the widths the model receives."""
    from backend.backtest.backtest_engine import _NUMERIC_FEATURES

    df = pd.DataFrame(rows)
    cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
    return df[cols].shape[1], cols


def test_feature_count_reaching_the_model_differs_by_six():
    """CRITERION 2. The two numbers recorded in the step artifact."""
    full_rows = [_fv(_FULL) for _ in range(3)]
    empty_rows = [_fv({}) for _ in range(3)]
    w_full, cols_full = _matrix_width(full_rows)
    w_empty, cols_empty = _matrix_width(empty_rows)

    assert w_full - w_empty == len(_MACRO_SERIES) == 6, (
        f"macro-present={w_full} macro-absent={w_empty}; the gap must be exactly "
        "the six macro features")

    # Criterion 2 asks for the two NUMBERS, not just their difference. The
    # cycle-1 Q/A refuted an earlier pair (35/29) that I had carried over from
    # the research brief without deriving it myself -- and which had already
    # propagated into a production comment. These are measured on THESE
    # fixtures; if the fixture or _NUMERIC_FEATURES changes, re-measure rather
    # than loosening the assertion.
    assert (w_full, w_empty) == (18, 12), (
        f"measured widths changed: macro-present={w_full}, macro-absent={w_empty}")

    # The number that makes the whole step necessary: a PARTIAL window has the
    # SAME width as a full one, because the six keys are still assigned (as
    # None) whenever any series resolves. Width cannot see the degraded case;
    # only the count can.
    w_partial, _ = _matrix_width([_fv(_PARTIAL) for _ in range(3)])
    assert w_partial == w_full == 18, (
        f"partial={w_partial} must equal full={w_full} -- if these ever differ, "
        "the argument for a count over a width check needs revisiting")
    assert set(cols_full) - set(cols_empty) == set(_MACRO_FEATURES)
    # The narrower matrix is what the model actually gets -- nothing re-pads it.
    assert all(k not in cols_empty for k in _MACRO_FEATURES)


def test_explicit_nulls_would_have_been_worse():
    """Pins the measurement that made criterion 1's OTHER branch a regression.

    If the six features were emitted as explicit nulls to keep the width stable,
    the all-NaN column survives `feature_cols`, its median is NaN,
    `fillna(train_medians)` is a no-op, and `fillna(0)` leaves a CONSTANT 0.0 --
    a plausible in-range value for a flat yield curve. Visible narrowing becomes
    invisible fabrication.
    """
    from backend.backtest.backtest_engine import _NUMERIC_FEATURES

    rows = [dict(_fv({}), **{k: None for k in _MACRO_FEATURES}) for _ in range(3)]
    df = pd.DataFrame(rows)
    cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
    X = df[cols].copy()
    assert "yield_curve_spread" in cols, "the all-NaN column SURVIVES the filter"
    med = X.median()
    assert pd.isna(med["yield_curve_spread"]), "its median is NaN"
    X = X.fillna(med).fillna(0)
    assert (X["yield_curve_spread"] == 0.0).all(), (
        "and it lands as a constant 0.0 -- an in-range fabricated value")


def test_partial_macro_is_median_imputed_which_is_why_it_must_be_recorded():
    """The live degraded case, end to end: a minority of partial rows keeps the
    column alive and the gap is filled with the training median."""
    from backend.backtest.backtest_engine import _NUMERIC_FEATURES

    rows = [_fv(_FULL) for _ in range(9)] + [_fv(_PARTIAL)]
    df = pd.DataFrame(rows)
    cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
    X = df[cols].copy()
    assert "cpi_yoy" in cols, "the column survives because most rows carry it"
    assert pd.isna(X["cpi_yoy"].iloc[-1]), "fixture precondition: the partial row is NaN"
    med = X.median()
    X = X.fillna(med).fillna(0)
    assert X["cpi_yoy"].iloc[-1] == med["cpi_yoy"], (
        "the partial row is silently filled with the training median")
    assert not np.isnan(X["cpi_yoy"].iloc[-1])


def test_engine_records_matrix_coverage():
    """The result-level half: a run must carry what its matrix actually got."""
    from backend.backtest.backtest_engine import BacktestEngine, BacktestResult

    assert "macro_coverage" not in BacktestResult().data_availability, (
        "seeded at run time, not on the dataclass default")
    assert isinstance(BacktestEngine._last_matrix_coverage, dict), (
        "class-level default so reading before a window runs is not an error")

    # EXECUTE the seam, do not scan its source. An earlier version of this guard
    # scanned backtest_engine.py for token names and a mutation run defeated it
    # TWICE: replacing `int(X.shape[1])` with a literal survived (the scan
    # checked the key, not the value), and deleting the `macro_series_min` entry
    # survived because that same string still appeared in a logger call. Both
    # are the source-scan trap. `compute_matrix_coverage` was extracted to make
    # this assertable on VALUES.
    from backend.backtest.backtest_engine import compute_matrix_coverage

    full = pd.DataFrame([{"macro_series_count": 6, "a": 1.0} for _ in range(4)])
    cov_full = compute_matrix_coverage(full, full[["a"]])
    assert cov_full["macro_series_min"] == 6
    assert cov_full["macro_rows_degraded"] == 0
    assert cov_full["n_samples"] == 4

    mixed = pd.DataFrame(
        [{"macro_series_count": 6, "a": 1.0, "b": 2.0} for _ in range(3)]
        + [{"macro_series_count": 2, "a": 4.0, "b": 5.0}]
    )
    cov_mixed = compute_matrix_coverage(mixed, mixed[["a", "b"]])
    assert cov_mixed["macro_series_min"] == 2, "the MIN is what exposes a partial window"
    assert cov_mixed["macro_series_max"] == 6
    assert cov_mixed["macro_rows_degraded"] == 1
    # n_features must track the ACTUAL matrix, not a literal.
    assert cov_full["n_features"] == 1 and cov_mixed["n_features"] == 2, (
        "n_features must be derived from X.shape, so a hardcoded value fails here")

    # A frame with no coverage column must not crash or fabricate a number.
    bare = pd.DataFrame([{"a": 1.0}])
    cov_bare = compute_matrix_coverage(bare, bare[["a"]])
    assert cov_bare["macro_series_min"] is None
    assert cov_bare["macro_rows_degraded"] == 0

    src = (REPO / "backend" / "backtest" / "backtest_engine.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "self._last_matrix_coverage = compute_matrix_coverage(" in code, (
        "the production path must USE the extracted seam, not a private copy")
    assert 'result.data_availability["macro_coverage"]' in code, (
        "the record must reach the result, not just the engine")

    an = (REPO / "backend" / "backtest" / "analytics.py").read_text(encoding="utf-8")
    an_code = "\n".join(l for l in an.splitlines() if not l.lstrip().startswith("#"))
    assert 'report["analytics"]["macro_coverage"]' in an_code, (
        "two consumers read only report['analytics'], so a top-level key alone "
        "leaves a degraded run looking normal to the code that compares runs")


# ---------------------------------------------------------------------------
# Criterion 4 -- the causes of an empty macro dict, DERIVED from the source
# ---------------------------------------------------------------------------

#: Each entry: (cause, file, the source anchor that proves it, classification).
#: `line` is re-derived at run time from `anchor` -- never hardcoded, because
#: every step that edits these files moves them.
MACRO_EMPTY_CAUSES = [
    ("macro load REFUSED (stale or unparseable)", "cache.py",
     "_REFUSAL_OUTCOMES = frozenset(",
     "expected -- 82.13 detects this and suppresses the per-cutoff fallback"),
    ("0-row macro table (outcome 'empty')", "cache.py",
     '"warm", "empty", "refused_unparseable"',
     "DEFECTIVE -- 'empty' is NOT in _REFUSAL_OUTCOMES, so it never arms "
     "set_macro_unavailable and every cutoff falls through to a BQ query"),
    ("cutoff earlier than any macro row", "cache.py",
     'if str(entry["date"]) > cutoff_date:',
     "expected -- there is genuinely no data to serve"),
    ("series not yet PUBLISHED at that vintage (point-in-time predicate)",
     "cache.py", "vintage is None or vintage.isoformat() > cutoff_date",
     "expected per phase-82.15, and the direct cause of the PARTIAL case "
     "measured live (fast series resolve, slow ones do not)"),
    # Anchored on a UNIQUE string. The cycle-1 Q/A caught the earlier anchor,
    # the bare "except Exception", matching FOUR sites in cache.py -- and
    # `hits[0]` resolved to a settings helper, not the BQ fallback this cause
    # names. An anchor that resolves to the wrong site does not establish a
    # file:line, which is what criterion 4 demands.
    ("per-cutoff BQ fallback raises (timeout, quota, auth)", "cache.py",
     "BQ macro query timed out",
     "DEFECTIVE -- swallowed into rows=[] and returned as {}"),
]


def test_empty_macro_causes_are_enumerated_from_the_source():
    """CRITERION 4. Every cause must be anchored in real source, and the
    enumeration itself asserted non-empty -- an empty list would satisfy a naive
    'all causes classified' check vacuously."""
    assert MACRO_EMPTY_CAUSES, "the enumeration must not be empty"
    assert len(MACRO_EMPTY_CAUSES) >= 4, (
        "the step names four causes (refusal, early cutoff, vintage miss, BQ "
        "timeout); fewer means the derivation lost one")

    cache_src = CACHE.read_text(encoding="utf-8").splitlines()
    seen_lines = set()
    for cause, fname, anchor, classification in MACRO_EMPTY_CAUSES:
        assert fname == "cache.py", cause
        hits = [i + 1 for i, l in enumerate(cache_src) if anchor in l]
        assert hits, f"anchor for {cause!r} not found in cache.py: {anchor!r}"
        seen_lines.add(hits[0])
        assert classification.startswith(("expected", "DEFECTIVE")), classification

    # EXACT equality, not >=. The cycle-1 Q/A showed `>= 4` over 5 causes
    # tolerates precisely the collision it exists to catch.
    assert len(seen_lines) == len(MACRO_EMPTY_CAUSES), (
        f"the anchors must point at DISTINCT sites: {len(MACRO_EMPTY_CAUSES)} "
        f"causes resolved to {len(seen_lines)} lines {sorted(seen_lines)} -- one "
        "line is doing duty for several causes")

    # And each anchor must be UNIQUE in the file, or hits[0] is arbitrary.
    for cause, _f, anchor, _cl in MACRO_EMPTY_CAUSES:
        n = sum(1 for l in cache_src if anchor in l)
        assert n == 1, (
            f"anchor for {cause!r} matches {n} lines in cache.py; a non-unique "
            "anchor makes the derived file:line arbitrary")

    defective = [c for c, _f, _a, cl in MACRO_EMPTY_CAUSES if cl.startswith("DEFECTIVE")]
    assert defective, "at least one cause is defective; classifying all as expected would be a whitewash"


def test_empty_is_not_a_refusal_outcome():
    """The hole this step found in 82.13, pinned so the classification above is
    a measurement rather than an opinion. Queued as its own step, not fixed
    here."""
    from backend.backtest import cache as c

    assert "empty" not in c._REFUSAL_OUTCOMES, (
        "if 'empty' becomes a refusal outcome, the classification in "
        "MACRO_EMPTY_CAUSES is stale and must be re-derived")
    assert c._REFUSAL_OUTCOMES == {"refused_unparseable", "refused_stale"}


def test_cached_macro_memoises_which_would_make_a_naive_guard_vacuous():
    """The fixture hazard, pinned. A test that called `cached_macro` twice for
    one cutoff would get the memoised result and never re-exercise the branch --
    so every guard in this file injects the provider instead."""
    src = CACHE.read_text(encoding="utf-8")
    fn = src.split("def cached_macro(", 1)[1].split("\ndef ", 1)[0]
    assert "_macro_cache[cutoff_date]" in fn, (
        "cached_macro no longer memoises; the hazard note above is stale")
    assert "if cutoff_date in _macro_cache:" in fn


def test_macro_series_tuple_matches_what_the_block_actually_reads():
    """DERIVED, not typed. If a seventh series is added to the macro block
    without extending `_MACRO_SERIES`, the count silently under-reports coverage
    -- so the tuple is checked against the block's own `.get("SERIES")` calls."""
    src = HD.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_feature_vector")
    body = ast.get_source_segment(src, fn) or ""
    macro_block = body.split("# ── Macro", 1)[1] if "# ── Macro" in body else body
    # NOTE the trailing `)` is deliberately NOT in this pattern: the production
    # calls are `_macro.get("FEDFUNDS", {})`, with a default. An earlier version
    # required `")` and matched NOTHING -- the derivation would have been
    # vacuous, and the `assert read` below is exactly what caught it.
    read = set(re.findall(r'_macro\.get\("([A-Z0-9]+)"', macro_block))
    assert read, "no macro series reads found -- the derivation would be vacuous"
    assert read == set(_MACRO_SERIES), (
        f"_MACRO_SERIES={sorted(_MACRO_SERIES)} but the block reads {sorted(read)}")
    assert len(_MACRO_FEATURES) == len(_MACRO_SERIES) == 6
