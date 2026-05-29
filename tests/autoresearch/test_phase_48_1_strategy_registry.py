"""phase-48.1: guards for the config-driven strategy seed registry.

The registry enumerates the rotation candidate SET that the producer scores and
feeds to the 47.6 selector. These guards assert: >=4 distinct seeds spanning
orthogonal strategy-TYPE axes (mean_reversion + quality_momentum + the
triple_barrier baseline), params = base overlaid with param_overrides,
operator-tunability (injected seeds honored), and fail-open behavior.

$0: pure, no engine/BQ/LLM.
"""
from __future__ import annotations

from backend.autoresearch.strategy_registry import (
    SEED_STRATEGIES,
    load_base_params,
    load_seed_strategies,
)

# Deterministic injected base so axis assertions do not couple to the live
# optimizer_best.json (which a future optimizer run may mutate -- see the
# registry/optimizer_best coupling risk).
_BASE = {
    "strategy": "triple_barrier",
    "tp_pct": 10.0,
    "sl_pct": 12.9,
    "holding_days": 90,
    "mr_holding_days": 15,
    "max_positions": 20,
    "target_annual_vol": 0.0,
    "trailing_stop_enabled": False,
}


def test_seed_set_has_at_least_four_distinct_ids():
    configs = load_seed_strategies(base_params=_BASE)
    assert len(configs) >= 4
    ids = [c["id"] for c in configs]
    assert len(ids) == len(set(ids)), f"seed ids must be distinct: {ids}"
    for c in configs:
        assert set(c.keys()) == {"id", "rationale", "params"}
        assert isinstance(c["params"], dict) and "strategy" in c["params"]


def test_seeds_span_at_least_three_strategy_types():
    configs = load_seed_strategies(base_params=_BASE)
    types = {c["params"].get("strategy") for c in configs}
    # orthogonal-axis diversification: distinct strategy TYPES, not param tweaks
    assert {"mean_reversion", "quality_momentum", "triple_barrier"} <= types


def test_param_overrides_apply_on_top_of_base():
    by_id = {c["id"]: c for c in load_seed_strategies(base_params=_BASE)}

    # baseline = verbatim base (empty overrides)
    assert by_id["tb_baseline"]["params"] == _BASE

    # mean-reversion diversifier: type + short holding/turnover regime
    mr = by_id["mr_short_horizon"]["params"]
    assert mr["strategy"] == "mean_reversion"
    assert mr["mr_holding_days"] < 30
    assert mr["holding_days"] == 30

    # quality-momentum trend sleeve
    qm = by_id["qm_trend_tilt"]["params"]
    assert qm["strategy"] == "quality_momentum"
    assert qm["holding_days"] == 120

    # risk-managed variant: same type, differentiated on the risk/exit axis
    rm = by_id["tb_risk_managed"]["params"]
    assert rm["strategy"] == "triple_barrier"
    assert rm["target_annual_vol"] > 0
    assert rm["trailing_stop_enabled"] is True
    assert rm["tp_pct"] == 6
    # base value not overridden stays put
    assert rm["holding_days"] == _BASE["holding_days"]


def test_does_not_mutate_module_constant_or_base():
    base_before = dict(_BASE)
    seeds_overrides_before = [dict(s.get("param_overrides") or {}) for s in SEED_STRATEGIES]
    load_seed_strategies(base_params=_BASE)
    assert _BASE == base_before  # base not mutated
    seeds_overrides_after = [dict(s.get("param_overrides") or {}) for s in SEED_STRATEGIES]
    assert seeds_overrides_before == seeds_overrides_after  # constant not mutated


def test_operator_tunable_injected_seeds():
    custom = [
        {"id": "factor_only", "param_overrides": {"strategy": "factor_model"}},
        {"id": "blend_only", "param_overrides": {"strategy": "blend"}},
    ]
    configs = load_seed_strategies(seeds=custom, base_params=_BASE)
    assert [c["id"] for c in configs] == ["factor_only", "blend_only"]
    assert configs[0]["params"]["strategy"] == "factor_model"
    assert configs[1]["params"]["strategy"] == "blend"


def test_fail_open_empty_base_still_enumerates_ids():
    # empty base (e.g. optimizer_best.json missing) must not raise; ids enumerate
    configs = load_seed_strategies(base_params={})
    assert [c["id"] for c in configs] == [
        "tb_baseline",
        "mr_short_horizon",
        "qm_trend_tilt",
        "tb_risk_managed",
    ]
    # override-only seeds still carry their override even with no base
    by_id = {c["id"]: c for c in configs}
    assert by_id["mr_short_horizon"]["params"]["strategy"] == "mean_reversion"


def test_skips_seed_with_no_id():
    configs = load_seed_strategies(
        seeds=[{"param_overrides": {"strategy": "x"}}, {"id": "ok"}],
        base_params=_BASE,
    )
    assert [c["id"] for c in configs] == ["ok"]


def test_real_base_load_works_and_enumerates():
    # the default path (reads the live optimizer_best.json) must load >=4 seeds
    # without raising; do NOT assert exact param values (avoid coupling to the
    # live file -- assert structure only).
    configs = load_seed_strategies()
    assert len(configs) >= 4
    assert all(isinstance(c["params"], dict) for c in configs)
    base = load_base_params()
    assert isinstance(base, dict)  # fail-open: dict even if file absent
