"""phase-82.2 -- three overpriced-market candidate strategies.

Guards the four immutable criteria. Measurements run against a committed,
seeded, BQ-free fixture (`fixtures/phase_82_2_label_fixture.py`) so every claim
below is reproducible rather than a lucky sample.
"""
from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest

from backend.backtest import backtest_engine as BE
from backend.backtest.backtest_engine import STRATEGY_REGISTRY, BacktestEngine
from backend.tests.fixtures import phase_82_2_label_fixture as F

NEW = ["stretch_regime", "qarp", "reversion_sigma"]

# A label set can look healthy on three rows. Non-degeneracy is only meaningful
# above a floor, so the two assertions always travel together.
MIN_LABELLED_ROWS = 200
MAX_CLASS_SHARE = 0.95


class _Provider:
    def build_feature_vector(self, ticker, entry_date):
        return F.build_feature_vector(ticker, entry_date)


class _Trader:
    transaction_cost_pct = 0.1


def _engine(holding_days: int = 60, mr_holding_days: int = 15) -> BacktestEngine:
    eng = BacktestEngine.__new__(BacktestEngine)
    eng.data_provider = _Provider()
    eng.trader = _Trader()
    eng.holding_days = holding_days
    eng.mr_holding_days = mr_holding_days
    return eng


def _labels(strategy: str, eng=None, dates=None) -> list[int]:
    eng = eng or _engine()
    dates = dates or F.entry_dates()
    method = getattr(eng, STRATEGY_REGISTRY[strategy])
    out = []
    with patch.object(BE.cache, "cached_prices", F.cached_prices):
        for t in F.TICKERS:
            for d in dates:
                lab = method(t, d)
                if lab is not None:
                    out.append(lab)
    return out


# ── criterion 1: registered with real label methods ──────────────────

@pytest.mark.parametrize("name", NEW)
def test_candidate_is_registered_with_a_real_method(name):
    assert name in STRATEGY_REGISTRY, f"{name} missing from STRATEGY_REGISTRY"
    method_name = STRATEGY_REGISTRY[name]
    assert hasattr(BacktestEngine, method_name), (
        f"{name} maps to {method_name}, which does not exist on BacktestEngine"
    )
    assert callable(getattr(BacktestEngine, method_name))


def test_three_new_strategies_were_added():
    assert len(set(NEW) & set(STRATEGY_REGISTRY)) == 3


def test_no_candidate_aliases_an_existing_label_method():
    """`meta_label` aliases triple_barrier; a candidate that merely re-points at
    an existing method would satisfy 'registered' without being a strategy."""
    existing = {
        "_compute_triple_barrier_label", "_compute_quality_momentum_label",
        "_compute_mean_reversion_label", "_compute_factor_label",
    }
    for name in NEW:
        assert STRATEGY_REGISTRY[name] not in existing, (
            f"{name} aliases the pre-existing {STRATEGY_REGISTRY[name]}"
        )


# ── criterion 2: non-degenerate on the committed fixture ─────────────

@pytest.mark.parametrize("name", NEW)
def test_candidate_labels_are_not_degenerate(name):
    labs = _labels(name)
    assert len(labs) >= MIN_LABELLED_ROWS, (
        f"{name} labelled only {len(labs)} rows; below the floor the "
        "no-class-above-95% assertion is trivially satisfiable"
    )
    counts = Counter(labs)
    top = max(counts.values()) / len(labs)
    assert top <= MAX_CLASS_SHARE, (
        f"{name} is degenerate: {dict(counts)} -> top class {top:.1%}"
    )


@pytest.mark.parametrize("name", NEW)
def test_candidate_produces_both_directions(name):
    """A label set that is 60/40 between 0 and +1 passes the 95% bar while
    still being unable to express a downside view."""
    counts = Counter(_labels(name))
    assert counts.get(1, 0) > 0, f"{name} never emits +1"
    assert counts.get(-1, 0) > 0, f"{name} never emits -1"


# ── criterion 3: the incumbent mean_reversion, MEASURED ──────────────

def test_existing_mean_reversion_degeneracy_is_measured():
    """The '~all-neutral' report was a claim. This measures it.

    On the committed fixture the existing label is degenerate at the boundary
    and, crucially, labels EVERY row -- it returns 0 rather than None when there
    is no signal, which is the mechanism that floods the neutral class.
    """
    labs = _labels("mean_reversion")
    counts = Counter(labs)
    top = max(counts.values()) / len(labs)

    # Bar set at 90%, not at the 95% non-degeneracy threshold, deliberately.
    # Re-seeding the fixture moves this figure over 93.4%-95.0% (five seeds
    # measured: 822 -> 95.0, 1 -> 94.1, 7 -> 93.6, 12345 -> 94.1, 99999 -> 93.4).
    # An assertion pinned at >= 95% passes only on the committed seed and would
    # then fail while claiming "not degenerate" at 93.4%, which plainly still is.
    # The finding is "overwhelmingly neutral", and it must survive re-seeding.
    assert counts[0] / len(labs) >= 0.90, (
        f"the neutral class is {counts[0]/len(labs):.1%}; the degeneracy report "
        "does not reproduce on this fixture and should be re-examined"
    )
    assert top >= 0.90, (
        f"mean_reversion top class {top:.1%} is below the degeneracy bar"
    )

    # The flooding mechanism: it labels everything.
    total_rows = len(F.TICKERS) * len(F.entry_dates())
    assert len(labs) == total_rows, (
        "mean_reversion is expected to label every row (0-on-no-signal); it "
        f"labelled {len(labs)} of {total_rows}"
    )


def test_reversion_sigma_fixes_the_degeneracy_it_replaces():
    """The candidate must beat the thing it is replacing ON THE SAME fixture,
    or it is a rename rather than a fix."""
    old = Counter(_labels("mean_reversion"))
    new = Counter(_labels("reversion_sigma"))
    old_top = max(old.values()) / sum(old.values())
    new_top = max(new.values()) / sum(new.values())
    assert new_top < old_top - 0.20, (
        f"reversion_sigma top class {new_top:.1%} vs mean_reversion {old_top:.1%} "
        "-- not a material improvement"
    )
    # And it must EXCLUDE no-signal rows rather than labelling them 0.
    assert sum(new.values()) < sum(old.values()), (
        "reversion_sigma should return None on no-signal, labelling fewer rows"
    )


# ── criterion 4: the (d) cash-timing overlay changes exposure ────────

def test_stretch_overlay_reduces_buy_signals_as_turbulence_rises():
    """Lens (d). The overlay lives inside `stretch_regime`: as market stretch
    rises the up-barrier widens, so fewer rows earn +1 and the trained model
    buys less -- holding cash by being harder to convince."""
    eng = _engine()
    dates = F.entry_dates()

    def _rate(stretch_value):
        with patch.object(BE.cache, "cached_prices", F.cached_prices), \
             patch.object(BacktestEngine, "_market_stretch", lambda self, d: stretch_value):
            labs = []
            for t in F.TICKERS:
                for d in dates:
                    lab = eng._compute_stretch_regime_label(t, d)
                    if lab is not None:
                        labs.append(lab)
        return sum(1 for x in labs if x == 1) / len(labs), len(labs)

    calm_rate, n_calm = _rate(0.6)
    turbulent_rate, n_turb = _rate(2.4)

    assert n_calm >= MIN_LABELLED_ROWS and n_turb >= MIN_LABELLED_ROWS
    assert turbulent_rate < calm_rate, (
        f"+1 rate did not fall with turbulence (calm {calm_rate:.1%} -> "
        f"turbulent {turbulent_rate:.1%}); the (d) overlay is inert"
    )


def test_market_stretch_actually_varies_across_the_sample():
    """Q/A A1: the overlay test patches `_market_stretch` out, and the
    look-ahead test only inspects the requested end date -- so replacing the
    method body with a constant 1.0 left every test green. A regime proxy that
    is silently flat would make the (d) overlay inert in production while the
    suite stayed green. This exercises the UNPATCHED method."""
    eng = _engine()
    with patch.object(BE.cache, "cached_prices", F.cached_prices):
        vals = [eng._market_stretch(d) for d in F.entry_dates()]
    vals = [v for v in vals if v is not None]

    assert len(vals) >= MIN_LABELLED_ROWS // 4, f"only {len(vals)} stretch values"
    lo, hi = min(vals), max(vals)
    assert hi / lo >= 2.0, (
        f"market stretch spans only {hi/lo:.2f}x ({lo:.3f}..{hi:.3f}) -- the "
        "regime proxy is effectively constant, so the overlay cannot modulate "
        "anything"
    )


def test_overlay_changes_exposure_without_patching_market_stretch():
    """Q/A A1, end-to-end: the calm third vs the turbulent third of the sample
    must differ in +1 rate using the REAL regime proxy, not an injected one."""
    eng = _engine()
    dates = F.entry_dates()
    with patch.object(BE.cache, "cached_prices", F.cached_prices):
        scored = [(d, eng._market_stretch(d)) for d in dates]
        scored = [(d, s) for d, s in scored if s is not None]
        scored.sort(key=lambda x: x[1])
        third = max(1, len(scored) // 3)
        calm = [d for d, _ in scored[:third]]
        turbulent = [d for d, _ in scored[-third:]]

        def _rate(ds):
            labs = [eng._compute_stretch_regime_label(t, d)
                    for t in F.TICKERS for d in ds]
            labs = [x for x in labs if x is not None]
            return sum(1 for x in labs if x == 1) / len(labs), len(labs)

        calm_rate, n_c = _rate(calm)
        turb_rate, n_t = _rate(turbulent)

    assert n_c > 50 and n_t > 50
    assert turb_rate < calm_rate, (
        f"with the REAL regime proxy the +1 rate did not fall "
        f"({calm_rate:.1%} calm -> {turb_rate:.1%} turbulent)"
    )


def test_sigma_barriers_include_the_round_trip_cost():
    """Q/A A2: zeroing the cost adjustment left every test green, although
    cost-adjustment is claimed prominently in the docstrings. The existing
    `mean_reversion` label has no cost adjustment at all, so this property is
    exactly the kind that rots silently."""
    eng = _engine()
    fv = {"annualized_volatility": 30.0}
    bars = eng._sigma_barriers(fv, 60)
    assert bars is not None
    sigma_h, cost = bars
    expected = 2 * eng.trader.transaction_cost_pct / 100
    assert cost == pytest.approx(expected), (
        f"round-trip cost is {cost}, expected {expected} "
        f"(2 x transaction_cost_pct/100)"
    )
    assert cost > 0, "cost adjustment is zero -- barriers are not cost-adjusted"


def test_qarp_excludes_non_candidates_rather_than_labelling_them_neutral():
    """Q/A A3: making `qarp` return 0 for non-candidates left every test green
    (880 rows at 67.7% top class, still under the 95% bar). The None-on-no-signal
    property was guarded for reversion_sigma but not here -- and it is the
    property that keeps the label set from being swamped by non-candidates."""
    labs_labelled = len(_labels("qarp"))
    total_rows = len(F.TICKERS) * len(F.entry_dates())
    assert labs_labelled < total_rows, (
        f"qarp labelled all {total_rows} rows; names failing the QARP gate must "
        "return None (excluded), not 0 (neutral)"
    )

    # And the exclusion must be driven by the gate, not by missing data.
    eng = _engine()
    with patch.object(BE.cache, "cached_prices", F.cached_prices):
        # HIVOL_DOWN has pe_ratio=-8.0 (negative earnings) -> never a candidate.
        assert all(
            eng._compute_qarp_label("HIVOL_DOWN", d) is None
            for d in F.entry_dates()[:20]
        ), "a name with negative earnings passed the QARP gate"


def test_market_stretch_is_backward_looking_only():
    """The regime proxy must not read prices after entry_date, or the overlay
    smuggles look-ahead into every label."""
    seen = {}

    def _spy(ticker, start, end):
        seen["end"] = end
        return F.cached_prices(ticker, start, end)

    eng = _engine()
    entry = F.entry_dates()[10]
    with patch.object(BE.cache, "cached_prices", _spy):
        eng._market_stretch(entry)
    assert seen["end"] == entry, (
        f"_market_stretch requested prices through {seen['end']} for an entry "
        f"on {entry} -- that is look-ahead"
    )


# ── all three must be FORWARD-LOOKING (the 82.16 defect) ─────────────

@pytest.mark.parametrize("name", NEW)
def test_candidate_label_depends_on_post_entry_prices(name):
    """`quality_momentum` and `factor_model` classify on features at entry_date
    and never read a forward price, so a model trained on them learns a function
    of its own inputs (step 82.16). Mutating the future must move these labels.
    """
    eng = _engine()
    dates = F.entry_dates()
    method = getattr(eng, STRATEGY_REGISTRY[name])

    def _run(mutate: bool):
        def _prices(ticker, start, end):
            df = F.cached_prices(ticker, start, end).copy()
            if mutate and not df.empty and ticker != "SPY" and len(df) > 1:
                # Collapse everything after the entry bar.
                df.iloc[1:, df.columns.get_loc("close")] = float(df["close"].iloc[0]) * 0.5
            return df
        with patch.object(BE.cache, "cached_prices", _prices):
            return [method(t, d) for t in F.TICKERS for d in dates]

    base, mutated = _run(False), _run(True)
    assert base != mutated, (
        f"{name} produced identical labels after the post-entry price path was "
        "destroyed -- it carries no forward information"
    )
