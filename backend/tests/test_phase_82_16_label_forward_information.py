"""phase-82.16 -- every registered strategy's label must carry FORWARD information.

THE DEFECT. `quality_momentum` and `factor_model` built the feature vector AT
entry_date and classified on `momentum_6m` / `quality_score` -- both entry-date
features, both members of `_NUMERIC_FEATURES`. Neither fetched a post-entry price. So
the GradientBoostingClassifier was trained to reproduce a deterministic threshold rule
over its OWN inputs: in-sample accuracy near-tautological, and any Sharpe/DSR computed
from it not evidence about the future. This is CIRCULAR ANALYSIS (Kriegeskorte et al.
2009), not look-ahead bias -- and note that a shuffled-label permutation test, the
reflex check for a bad target, would PASS both of them.

THE TRAP THIS FILE EXISTS TO AVOID. A mutation test alone is not enough. Measured on
the 82.2 fixture, `factor_model` produced **0 labels at all**, so "no label changed
when I destroyed the future" was indistinguishable from "there was nothing to change".
Every strategy therefore has to clear a COVERAGE precondition -- at least one non-None
label -- before its mutation result means anything. Without that, any strategy that
silently returns None passes criterion 1 for free.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.backtest import backtest_engine as BE
from backend.backtest.backtest_engine import (
    NON_COMPARABLE_STRATEGIES,
    STRATEGY_REGISTRY,
    BacktestEngine,
)
from backend.tests.fixtures import phase_82_2_label_fixture as F


class _Provider:
    def build_feature_vector(self, ticker, entry_date):
        return F.build_feature_vector(ticker, entry_date)


class _Trader:
    transaction_cost_pct = 0.1


def _engine() -> BacktestEngine:
    """The 82.2 builder, plus the tp_pct/sl_pct the research gate measured missing --
    without them 2 of the strategies raise AttributeError, which would look like a
    strategy defect rather than a fixture gap."""
    eng = BacktestEngine.__new__(BacktestEngine)
    eng.data_provider = _Provider()
    eng.trader = _Trader()
    eng.holding_days = 60
    eng.mr_holding_days = 15
    eng.tp_pct = 10.0
    eng.sl_pct = 5.0
    return eng


def _run(method, mutate: bool):
    """Label every fixture row, optionally after destroying the post-entry path."""
    def _prices(ticker, start, end):
        df = F.cached_prices(ticker, start, end).copy()
        if mutate and not df.empty and ticker != "SPY" and len(df) > 1:
            df.iloc[1:, df.columns.get_loc("close")] = float(df["close"].iloc[0]) * 0.5
        return df

    with patch.object(BE.cache, "cached_prices", _prices):
        return [method(t, d) for t in F.TICKERS for d in F.entry_dates()]


# ---------------------------------------------------------------------------
# CRITERION 3 -- the enumeration itself, asserted non-empty
# ---------------------------------------------------------------------------

def test_registry_is_non_empty():
    """An empty parametrize collects ZERO tests and reports green, so the registry
    being non-empty is a precondition for every guard below meaning anything."""
    assert STRATEGY_REGISTRY, "STRATEGY_REGISTRY is empty -- every guard below is vacuous"
    assert len(STRATEGY_REGISTRY) >= 3, (
        f"only {len(STRATEGY_REGISTRY)} strategies registered; the demotion in this "
        "step was supposed to leave the forward-looking ones behind, not empty the pool"
    )


def test_parametrization_is_not_empty():
    """Pins the collect-zero-tests failure mode directly: if the parametrize source
    ever empties, this fails instead of the suite silently shrinking to nothing."""
    assert list(STRATEGY_REGISTRY.keys()), "the parametrize source is empty"


# ---------------------------------------------------------------------------
# CRITERIA 1 + 2 -- enumerate the REGISTRY, not a hand-written list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY.keys()))
def test_label_has_coverage_on_the_fixture(name):
    """THE PRECONDITION. Measured: factor_model produced 0 labels on this fixture, so
    its 'nothing changed under mutation' was VACUOUS. Any strategy that cannot label
    anything must fail here rather than sail through the mutation test."""
    method = getattr(_engine(), BE.STRATEGY_REGISTRY[name])
    labelled = [x for x in _run(method, mutate=False) if x is not None]
    assert labelled, (
        f"{name} produced ZERO non-None labels on the fixture. Its mutation result "
        "would be meaningless -- 'no label changed' cannot be distinguished from "
        "'no label existed'."
    )


@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY.keys()))
def test_label_depends_on_post_entry_prices(name):
    """CRITERION 1. Enumerated from STRATEGY_REGISTRY itself, so a strategy added
    later cannot skip the check.

    'at least one fixture row' is deliberate, not lax: mean_reversion was measured
    changing on only 119 of 880 rows, so requiring most rows would fail a genuinely
    forward-looking strategy.
    """
    method = getattr(_engine(), BE.STRATEGY_REGISTRY[name])
    base = _run(method, mutate=False)
    mutated = _run(method, mutate=True)

    assert [x for x in base if x is not None], (
        f"{name} has no coverage -- see test_label_has_coverage_on_the_fixture"
    )
    changed = sum(1 for a, b in zip(base, mutated) if a != b)
    assert changed >= 1, (
        f"{name} produced IDENTICAL labels after the entire post-entry price path was "
        f"destroyed ({len(base)} rows compared). It carries no forward information, so "
        "a model trained on it learns a function of its own inputs and its Sharpe/DSR "
        "say nothing about the future."
    )


# ---------------------------------------------------------------------------
# CRITERION 4 -- the guard can fail
# ---------------------------------------------------------------------------

def test_guard_detects_a_deliberately_non_forward_stub(monkeypatch):
    """CRITERION 4 -- and it must drive the REAL guard, not a re-computation of it.

    The first version of this test asserted `changed == 0` for a stub using the same
    `_run` helper, but never touched STRATEGY_REGISTRY and never invoked
    `test_label_depends_on_post_entry_prices`. The cycle-1 Q/A proved the hole:
    weakening the real guard's threshold to `>= 0` would have left this test GREEN.
    A negative control that re-implements the check proves nothing about the check.

    So: register a non-forward stub in a COPY of the registry (never the module
    global -- it is imported by strategy_backtest_adapter), point the engine at it,
    and assert the REAL guard raises.
    """
    def _stub_label(self, ticker, entry_date):
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv:
            return None
        m = fv.get("momentum_6m")
        return None if m is None else (1 if m > 0 else -1)

    monkeypatch.setattr(BacktestEngine, "_compute_stub_label", _stub_label, raising=False)
    patched = {**STRATEGY_REGISTRY, "evil_stub": "_compute_stub_label"}
    monkeypatch.setattr(BE, "STRATEGY_REGISTRY", patched)

    # The enumeration must PICK IT UP ...
    assert "evil_stub" in BE.STRATEGY_REGISTRY

    # ... and the REAL guard must reject it.
    with pytest.raises(AssertionError, match="IDENTICAL labels"):
        test_label_depends_on_post_entry_prices("evil_stub")

    # And the same real guard must still ACCEPT a forward-looking strategy, so it
    # cannot pass by rejecting everything.
    test_label_depends_on_post_entry_prices("triple_barrier")


def test_a_forward_looking_strategy_does_change(sample="triple_barrier"):
    """Mirror of the above: the fixture must be able to show a POSITIVE result too,
    else 'changed == 0' everywhere would be a fixture artefact."""
    method = getattr(_engine(), STRATEGY_REGISTRY[sample])
    changed = sum(
        1 for a, b in zip(_run(method, False), _run(method, True)) if a != b
    )
    assert changed >= 1, (
        f"{sample} is known forward-looking but showed no change -- the mutation is "
        "not reaching the label path, so every other result in this file is suspect"
    )


# ---------------------------------------------------------------------------
# CRITERION 2 -- the demotion actually happened, everywhere it matters
# ---------------------------------------------------------------------------

DEMOTED = ("quality_momentum", "factor_model")


@pytest.mark.parametrize("name", DEMOTED)
def test_non_forward_strategy_is_out_of_the_registry(name):
    assert name not in STRATEGY_REGISTRY, (
        f"{name} carries no forward information but is still registered; the "
        "optimizer rotates over the registry as if every member were comparable"
    )
    assert name in NON_COMPARABLE_STRATEGIES, (
        f"{name} was removed from the registry but not recorded in "
        "NON_COMPARABLE_STRATEGIES -- a silent deletion loses the reason"
    )
    assert NON_COMPARABLE_STRATEGIES[name].strip(), f"{name} has no recorded reason"


@pytest.mark.parametrize("name", DEMOTED)
def test_optimizer_cannot_select_a_demoted_strategy(name):
    """Removing from the registry is not enough on its own. The optimizer had its own
    hardcoded list; leaving it would make it log and score a strategy that never ran."""
    from backend.backtest.quant_optimizer import AVAILABLE_STRATEGIES

    assert name not in AVAILABLE_STRATEGIES, (
        f"{name} is demoted but still selectable by the optimizer"
    )


def test_optimizer_list_is_derived_from_the_registry_not_restated():
    """A restated list drifts. This asserts the derivation, so a future registry
    change cannot leave the optimizer behind."""
    from backend.backtest.quant_optimizer import AVAILABLE_STRATEGIES

    for name in STRATEGY_REGISTRY:
        assert name in AVAILABLE_STRATEGIES, (
            f"registry strategy {name} is not offered to the optimizer -- the list "
            "has drifted from the registry"
        )
    # phase-82.46 dropped the `- {"blend"}` carve-out: blend is no longer offered,
    # so the exemption was a dead escape hatch that would have silently tolerated
    # a future non-registry name of the same shape.
    extra = set(AVAILABLE_STRATEGIES) - set(STRATEGY_REGISTRY)
    assert not extra, (
        f"optimizer offers strategies that are not in the registry: {sorted(extra)}"
    )


def test_demoted_methods_are_kept_so_the_demotion_is_reversible():
    """Demotion, not deletion -- the methods stay so nothing referencing them by name
    breaks and either can be restored if given a real forward label."""
    eng = _engine()
    for attr in ("_compute_quality_momentum_label", "_compute_factor_label"):
        assert callable(getattr(eng, attr, None)), (
            f"{attr} was deleted rather than demoted"
        )


# ---------------------------------------------------------------------------
# The demotion must not create a NEW silent wrongness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", DEMOTED)
def test_requesting_a_demoted_strategy_is_loud_not_silent(name, caplog):
    """Drives the REAL decision (`resolve_strategy`), not a copy of it.

    The engine has always coerced an unknown strategy to triple_barrier. Demoting
    these two made that coercion REACHABLE, and a run reported as quality_momentum
    that actually ran triple_barrier is worse than the tautological label it replaced
    -- the provenance is wrong too. A caller really can ask: archetype_library still
    lists both names in its own IMPLEMENTED_STRATEGY_IDS.
    """
    from backend.backtest.backtest_engine import resolve_strategy

    with caplog.at_level("WARNING"):
        effective, was_demoted = resolve_strategy(name)

    assert was_demoted is True, f"{name} is demoted but resolve_strategy did not say so"
    assert effective == "triple_barrier"
    assert any(name in r.getMessage() for r in caplog.records), (
        f"requesting demoted strategy {name!r} produced no warning naming it -- the "
        "fallback is silent, so a caller cannot tell the results are not its own"
    )


@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_a_registered_strategy_resolves_to_itself_without_warning(name, caplog):
    """Mirror: resolve_strategy must not flag or warn on a healthy strategy."""
    from backend.backtest.backtest_engine import resolve_strategy

    with caplog.at_level("WARNING"):
        effective, was_demoted = resolve_strategy(name)

    assert effective == name
    assert was_demoted is False
    assert not [r for r in caplog.records if "NON-COMPARABLE" in r.getMessage()], (
        f"{name} is registered but resolve_strategy warned about it"
    )


def test_engine_source_flags_a_demoted_request(caplog):
    """Structural companion: the PRODUCTION __init__ must set the flag and warn.

    The test above pins the contract; this pins that backtest_engine.py actually
    implements it, so the contract cannot be satisfied only by the test's own
    re-statement of it.
    """
    import ast as _ast
    from pathlib import Path as _P

    src = (_P(BE.__file__)).read_text(encoding="utf-8")
    tree = _ast.parse(src)
    fn = [n for n in _ast.walk(tree)
          if isinstance(n, _ast.FunctionDef) and n.name == "resolve_strategy"]
    assert fn, "resolve_strategy not found in backtest_engine.py"
    body = _ast.unparse(fn[0])
    assert "NON_COMPARABLE_STRATEGIES" in body, (
        "resolve_strategy never consults NON_COMPARABLE_STRATEGIES"
    )
    init = [n for n in _ast.walk(tree)
            if isinstance(n, _ast.FunctionDef) and n.name == "__init__"]
    assert any("resolve_strategy" in _ast.unparse(n) for n in init), (
        "BacktestEngine.__init__ does not call resolve_strategy, so the production "
        "path bypasses the demotion check entirely"
    )


def test_autoresearch_seed_for_a_demoted_strategy_is_excluded_not_silently_run():
    """A LIVE consequence of the demotion, verified rather than assumed.

    `backend/autoresearch/strategy_registry.py` carries a seed whose strategy is
    `quality_momentum`. After the demotion the adapter's own guard raises on a name
    that is not in STRATEGY_REGISTRY, and the candidate producer catches and SKIPS.
    Net effect: the non-comparable strategy is excluded from the candidate pool
    rather than silently backtested as triple_barrier -- which is exactly what
    Bailey & Lopez de Prado require of a non-comparable candidate.

    Pinned here because the alternative failure (the seed silently running under a
    different label) would corrupt the trial pool that DSR deflation depends on.
    """
    from backend.autoresearch.strategy_backtest_adapter import _default_param_grid

    for name in DEMOTED:
        try:
            _default_param_grid({"strategy": name}, 1)
        except ValueError as exc:
            assert name in str(exc), (
                f"the adapter rejected {name!r} but its message does not name it"
            )
        else:
            pytest.fail(
                f"the adapter ACCEPTED demoted strategy {name!r} -- it would be "
                "backtested as triple_barrier and scored under the wrong name, "
                "corrupting the trial pool DSR deflation depends on"
            )

    # And a REGISTERED strategy must still be accepted, else this guard would pass
    # by rejecting everything.
    assert _default_param_grid({"strategy": "triple_barrier"}, 1), (
        "the adapter rejected a registered strategy -- the guard is over-broad"
    )


def test_unregistered_non_demoted_names_are_also_loud(caplog):
    """Cycle-1 Q/A: the demoted names were not the only route to the silent
    coercion. `blend` is offered by the optimizer, has NO implementation in
    backtest_engine, and resolved to triple_barrier with no signal -- then got
    scored under the requested name."""
    from backend.backtest.backtest_engine import resolve_strategy

    for name in ("blend", "totally_made_up"):
        caplog.clear()
        with caplog.at_level("WARNING"):
            effective, was_demoted = resolve_strategy(name)
        assert effective == "triple_barrier"
        assert was_demoted is False, f"{name} is not a DEMOTED strategy"
        assert any(name in r.getMessage() for r in caplog.records), (
            f"{name!r} silently became triple_barrier with no warning -- it would be "
            "scored under the requested name"
        )


def test_optimizer_trial_pool_composition_is_pinned():
    """The derivation ENLARGED the selectable set, and that is a real change.

    The old hand-written literal had drifted and omitted the three 82.2 candidates,
    so deriving from the registry makes qarp / reversion_sigma / stretch_regime newly
    selectable. Pinned here so the change to the offered set is visible and
    intentional rather than incidental.

    phase-82.46 CORRECTION. This docstring used to justify the pin by saying "the
    trial pool is a direct input to DSR deflation (Bailey & Lopez de Prado)".
    THAT IS FALSE: `compute_deflated_sharpe` has no pool parameter and
    `num_trials` increments once per ITERATION. The pin is still worth having --
    a silent change to what the optimizer may select is worth catching -- but the
    reason is attribution and wasted iterations, not pool-size deflation. See
    `test_phase_82_46_trial_pool_composition.py::test_pool_size_does_not_enter_dsr`.
    """
    from backend.backtest.quant_optimizer import AVAILABLE_STRATEGIES

    previously_offered = {
        "triple_barrier", "quality_momentum", "mean_reversion",
        "factor_model", "meta_label", "blend",
    }
    now = set(AVAILABLE_STRATEGIES)

    assert now - previously_offered == {"qarp", "reversion_sigma", "stretch_regime"}, (
        f"the optimizer's newly-selectable set changed unexpectedly: "
        f"{sorted(now - previously_offered)}"
    )
    # phase-82.46 removed `blend` as well, deliberately: it was never a registry
    # key, had no implementation, and resolved to triple_barrier while being
    # SCORED under the requested name. The intent of this assertion is unchanged
    # -- the offered set must not lose anything unexplained -- so the expected
    # removals are widened by exactly that one decided name, not loosened.
    assert previously_offered - now == {"quality_momentum", "factor_model", "blend"}, (
        f"the optimizer lost strategies other than the two demoted ones plus the "
        f"82.46-removed `blend`: {sorted(previously_offered - now)}"
    )
