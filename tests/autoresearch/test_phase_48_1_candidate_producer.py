"""phase-48.1: guards for the per-strategy producer + the registry->producer->selector spine.

All tests use an IN-MEMORY fixture backtest_fn -- ZERO real backtests, ZERO BQ,
ZERO LLM, ZERO macro preload. They prove: the producer emits exactly the selector
contract, SKIPS malformed candidates (so the gate never silently drops them), and
the full bake-off composes (first_selection + gate-veto + anti-churn retain).
"""
from __future__ import annotations

import pytest

from backend.autoresearch.strategy_candidate_producer import (
    build_per_strategy_candidates,
    run_strategy_bakeoff,
)

_BASE = {"strategy": "triple_barrier", "target_annual_vol": 0.0}


def _fixture_backtest_fn(params):
    """Map a strategy's params -> canned metrics (no real backtest).

    mr (0.99/0.05) > qm (0.96/0.10) > tb_baseline (0.953/0.12) all pass the
    DSR>=0.95/PBO<=0.20 gate; tb_risk_managed (0.94/0.30) is gate-vetoed on BOTH
    legs. tb_baseline vs tb_risk_managed (both strategy=triple_barrier) are
    distinguished by target_annual_vol (the risk-managed override sets it >0).
    """
    strat = params.get("strategy")
    if strat == "mean_reversion":
        return {"dsr": 0.99, "pbo": 0.05, "sharpe": 1.40}
    if strat == "quality_momentum":
        return {"dsr": 0.96, "pbo": 0.10, "sharpe": 1.20}
    if strat == "triple_barrier" and float(params.get("target_annual_vol") or 0) > 0:
        return {"dsr": 0.94, "pbo": 0.30, "sharpe": 0.90}  # tb_risk_managed -> vetoed
    return {"dsr": 0.953, "pbo": 0.12, "sharpe": 1.17}  # tb_baseline


def _configs():
    from backend.autoresearch.strategy_registry import load_seed_strategies
    return load_seed_strategies(base_params=_BASE)


def test_producer_emits_exact_selector_contract():
    cands = build_per_strategy_candidates(_configs(), _fixture_backtest_fn)
    assert len(cands) == 4
    for c in cands:
        assert set(c.keys()) == {"strategy", "dsr", "pbo", "params", "sharpe"}
        assert isinstance(c["dsr"], float) and isinstance(c["pbo"], float)
        assert isinstance(c["sharpe"], float)
        assert isinstance(c["params"], dict)
    # the 'strategy' value is the REGISTRY ID (what the selector's _strategy_id reads)
    assert {c["strategy"] for c in cands} == {
        "tb_baseline", "mr_short_horizon", "qm_trend_tilt", "tb_risk_managed",
    }


def test_producer_skips_when_backtest_fn_raises():
    def fn(params):
        if params.get("strategy") == "quality_momentum":
            raise RuntimeError("simulated backtest failure")
        return _fixture_backtest_fn(params)

    cands = build_per_strategy_candidates(_configs(), fn)
    ids = {c["strategy"] for c in cands}
    assert "qm_trend_tilt" not in ids  # the raising config is skipped
    assert {"tb_baseline", "mr_short_horizon", "tb_risk_managed"} <= ids  # others survive
    assert len(cands) == 3


def test_producer_skips_when_pbo_missing():
    def fn(params):
        m = dict(_fixture_backtest_fn(params))
        if params.get("strategy") == "mean_reversion":
            m.pop("pbo")  # omit pbo -> gate would silently drop; producer must skip first
        return m

    cands = build_per_strategy_candidates(_configs(), fn)
    ids = {c["strategy"] for c in cands}
    assert "mr_short_horizon" not in ids
    # no emitted candidate ever carries a None pbo/dsr
    assert all(c["pbo"] is not None and c["dsr"] is not None for c in cands)


def test_producer_skips_non_numeric_metrics():
    def fn(params):
        m = dict(_fixture_backtest_fn(params))
        if params.get("strategy") == "quality_momentum":
            m["dsr"] = "not-a-number"
        return m

    cands = build_per_strategy_candidates(_configs(), fn)
    assert "qm_trend_tilt" not in {c["strategy"] for c in cands}


def test_bakeoff_first_selection_picks_top_dsr_gate_passer():
    verdict = run_strategy_bakeoff(_fixture_backtest_fn, incumbent=None, base_params=_BASE)
    assert verdict["reason"] == "first_selection"
    assert verdict["switched"] is True
    assert verdict["selected_id"] == "mr_short_horizon"  # dsr 0.99, top passer
    # ranked = gate-passers DSR-desc; tb_risk_managed (0.94/0.30) is vetoed out
    assert verdict["ranked"] == ["mr_short_horizon", "qm_trend_tilt", "tb_baseline"]
    assert "tb_risk_managed" not in verdict["ranked"]
    assert verdict["num_trials"] == 4  # count of seed configs


def test_bakeoff_anti_churn_retains_incumbent_below_min_improvement():
    # incumbent is a DIFFERENT strategy than the top passer, with dsr only
    # marginally below -> delta < min_improvement (0.01) -> retain (no whipsaw)
    incumbent = {"strategy": "qm_trend_tilt", "dsr": 0.985}
    verdict = run_strategy_bakeoff(_fixture_backtest_fn, incumbent=incumbent, base_params=_BASE)
    assert verdict["reason"] == "below_min_improvement"
    assert verdict["switched"] is False
    assert verdict["selected_id"] == "qm_trend_tilt"  # incumbent retained
    assert verdict["delta_dsr"] == pytest.approx(0.99 - 0.985, abs=1e-6)


def test_bakeoff_switches_on_material_improvement():
    # incumbent materially worse than the top passer -> switch
    incumbent = {"strategy": "tb_baseline", "dsr": 0.953}
    verdict = run_strategy_bakeoff(_fixture_backtest_fn, incumbent=incumbent, base_params=_BASE)
    assert verdict["reason"] == "dsr_improvement"
    assert verdict["switched"] is True
    assert verdict["selected_id"] == "mr_short_horizon"
