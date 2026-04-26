"""phase-10.7.5 unit tests for the API-Credit Reallocator.

Mirrors the tests/meta_evolution/test_cron_allocator.py pattern:
- _write_yaml() helper
- _provider() helper
- 12+ tests covering: proportional allocation, floor/ceiling clamps,
  disabled providers excluded, single-provider edge case, default
  total budget from YAML, infeasible-floor rejection, two-pass
  rebalance under both under-spend and over-spend, rich-data
  introspection, all-disabled edge case, float precision invariants.

No external deps beyond pyyaml + pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.provider_rebalancer import (  # noqa: E402
    PROVIDER_BUDGET_DEFAULT_TOTAL_USD,
    Allocation,
    allocate,
    compute_allocations,
    rebalance,
)


def _write_yaml(tmp_path: Path, providers: list[dict], total: float | None = None) -> Path:
    cfg = {"version": 1, "providers": providers}
    if total is not None:
        cfg["total_daily_usd_budget"] = total
    p = tmp_path / "provider_budget.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _provider(name: str, weight: int, floor: float, ceiling: float, enabled: bool = True) -> dict:
    return {
        "name": name,
        "priority_weight": weight,
        "min_floor_usd": floor,
        "max_ceiling_usd": ceiling,
        "enabled": enabled,
    }


# -----------------------
# Allocation tests
# -----------------------

def test_proportional_basic(tmp_path):
    """3 providers 10/6/2, total $9 -- proportional shares 5/3/1, no clamps."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("google_vertex", 6, 0.0, 10.0),
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=9.0)
    out = allocate(p)
    # weights 10+6+2=18; shares 9*(10/18)=5.0, 9*(6/18)=3.0, 9*(2/18)=1.0
    assert out["anthropic"] == pytest.approx(5.0)
    assert out["google_vertex"] == pytest.approx(3.0)
    assert out["openai"] == pytest.approx(1.0)
    # No clamps -> sum equals total
    assert sum(out.values()) == pytest.approx(9.0)


def test_disabled_excluded(tmp_path):
    """Disabled provider absent from result; remaining weights renormalize."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("google_vertex", 6, 0.0, 10.0, enabled=False),
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=12.0)
    out = allocate(p)
    assert "google_vertex" not in out
    # weights 10+2=12; shares 12*(10/12)=10, 12*(2/12)=2
    assert out["anthropic"] == pytest.approx(10.0)
    assert out["openai"] == pytest.approx(2.0)


def test_min_floor_enforced(tmp_path):
    """Low-priority provider gets lifted to its floor when share < floor."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("github_models", 1, 1.0, 10.0),  # raw share = $0.45 but floor $1.00
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    out = allocate(p)
    # raw: anthropic = 5*(10/11) = 4.5454, github = 5*(1/11) = 0.4545
    # github clamped UP to 1.00; anthropic stays at raw
    assert out["github_models"] == pytest.approx(1.0)
    assert out["anthropic"] == pytest.approx(4.5454, rel=1e-3)


def test_max_ceiling_enforced(tmp_path):
    """High-priority provider clamped to ceiling when share > ceiling."""
    providers = [
        _provider("anthropic", 10, 0.0, 2.0),  # raw share $4.16 but ceiling $2
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    out = allocate(p)
    # raw anthropic = 5*(10/12) = 4.1667; clamped to ceiling 2.0
    assert out["anthropic"] == pytest.approx(2.0)
    # openai unchanged (still at raw share)
    assert out["openai"] == pytest.approx(5.0 * 2 / 12, rel=1e-3)


def test_single_provider_full_budget(tmp_path):
    """Single enabled provider gets the full budget (subject to ceiling)."""
    providers = [
        _provider("anthropic", 10, 0.0, 100.0),
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    out = allocate(p)
    assert len(out) == 1
    assert out["anthropic"] == pytest.approx(5.0)


def test_allocate_uses_yaml_default_budget(tmp_path):
    """When total_budget arg is None, reads total_daily_usd_budget from YAML."""
    providers = [
        _provider("anthropic", 10, 0.0, 100.0),
        _provider("openai", 2, 0.0, 100.0),
    ]
    p = _write_yaml(tmp_path, providers, total=12.0)
    out = allocate(p, total_budget=None)
    assert out["anthropic"] == pytest.approx(10.0)
    assert out["openai"] == pytest.approx(2.0)


def test_default_total_when_yaml_omits(tmp_path):
    """When YAML omits total_daily_usd_budget, falls back to module default."""
    providers = [_provider("anthropic", 10, 0.0, 100.0)]
    p = _write_yaml(tmp_path, providers, total=None)
    out = allocate(p, total_budget=None)
    assert out["anthropic"] == pytest.approx(PROVIDER_BUDGET_DEFAULT_TOTAL_USD)


def test_sum_floors_gt_total_raises(tmp_path):
    """Infeasible: sum(floors) > total budget -- raise ValueError at load."""
    providers = [
        _provider("anthropic", 10, 5.0, 10.0),
        _provider("openai", 2, 3.0, 10.0),  # 5 + 3 = 8 > total 5
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    with pytest.raises(ValueError, match="infeasible"):
        allocate(p)


def test_floor_gt_ceiling_raises(tmp_path):
    """Per-provider floor > ceiling -- ValueError."""
    providers = [_provider("anthropic", 10, 5.0, 1.0)]  # floor 5 > ceiling 1
    p = _write_yaml(tmp_path, providers, total=10.0)
    with pytest.raises(ValueError, match="min_floor_usd"):
        allocate(p)


def test_compute_allocations_returns_rich_data(tmp_path):
    """Allocation dataclass exposes raw_budget, clamped_budget, floor, ceiling, was_clamped."""
    providers = [_provider("anthropic", 10, 1.0, 10.0)]
    p = _write_yaml(tmp_path, providers, total=5.0)
    out = compute_allocations(p)
    assert len(out) == 1
    a = out[0]
    assert isinstance(a, Allocation)
    assert a.provider == "anthropic"
    assert a.weight == 10
    assert a.floor == 1.0
    assert a.ceiling == 10.0
    assert a.raw_budget == pytest.approx(5.0)
    assert a.clamped_budget == pytest.approx(5.0)
    assert a.was_clamped is False


def test_all_providers_disabled_returns_empty(tmp_path):
    """If every provider is disabled, allocate returns {}."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0, enabled=False),
        _provider("openai", 2, 0.0, 10.0, enabled=False),
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    assert allocate(p) == {}


def test_float_precision_not_int(tmp_path):
    """USD allocations are floats (not rounded to int tokens like cron_allocator)."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("openai", 7, 0.0, 10.0),  # weights 10+7=17; non-integer division
    ]
    p = _write_yaml(tmp_path, providers, total=5.0)
    out = allocate(p)
    # 5 * 10/17 = 2.941176... -- must NOT be int(2)
    assert out["anthropic"] == pytest.approx(2.9411764705882355, rel=1e-6)
    assert isinstance(out["anthropic"], float)


# -----------------------
# Rebalance tests (two-pass max-min progressive fill)
# -----------------------

def test_rebalance_underspent_surplus_redistributes(tmp_path):
    """Under-spent provider's surplus flows to others proportionally by weight."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("google_vertex", 6, 0.0, 10.0),
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=9.0)
    allocs = compute_allocations(p)
    # Initial: anthropic=5, gvertex=3, openai=1
    # Anthropic only used $1 today -- $4 surplus.
    # gvertex + openai have headroom; share by weight 6:2 = 3:1
    # gvertex gets 3 + 4*(6/8) = 3 + 3.0 = 6.0
    # openai gets 1 + 4*(2/8) = 1 + 1.0 = 2.0
    # anthropic locks in $1 (its actual usage)
    out = rebalance(allocs, {"anthropic": 1.0, "google_vertex": 3.0, "openai": 1.0})
    assert out["anthropic"] == pytest.approx(1.0)
    assert out["google_vertex"] == pytest.approx(6.0)
    assert out["openai"] == pytest.approx(2.0)


def test_rebalance_overspent_capped_at_ceiling(tmp_path):
    """Over-spent provider stays at min(used, ceiling); does not eat surplus."""
    providers = [
        _provider("anthropic", 10, 0.0, 4.0),  # ceiling $4
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=6.0)
    allocs = compute_allocations(p)
    # Initial: anthropic=4 (clamped to ceiling), openai=1
    # Anthropic tried to spend $7 but ceiling is $4 -- granted $4 (max_ceiling clamp)
    # Openai under-spent at $0.5; surplus from openai = $0.5
    # anthropic is at ceiling so no headroom -- surplus stays with openai (or goes nowhere)
    out = rebalance(allocs, {"anthropic": 7.0, "openai": 0.5})
    assert out["anthropic"] == pytest.approx(4.0)  # capped at ceiling
    assert out["openai"] == pytest.approx(0.5)  # locked in actual usage


def test_rebalance_empty_allocations_returns_empty():
    """Empty input -> empty output."""
    assert rebalance([], {}) == {}


def test_rebalance_no_surplus_passthrough(tmp_path):
    """When everyone uses exactly their budget, no surplus, output == clamped_budget."""
    providers = [
        _provider("anthropic", 10, 0.0, 10.0),
        _provider("openai", 2, 0.0, 10.0),
    ]
    p = _write_yaml(tmp_path, providers, total=12.0)
    allocs = compute_allocations(p)
    # anthropic=10, openai=2; both used exactly that
    out = rebalance(allocs, {"anthropic": 10.0, "openai": 2.0})
    assert out["anthropic"] == pytest.approx(10.0)
    assert out["openai"] == pytest.approx(2.0)


def test_real_yaml_loads(tmp_path):
    """The real .claude/provider_budget.yaml must load + allocate cleanly."""
    real_yaml = REPO_ROOT / ".claude" / "provider_budget.yaml"
    assert real_yaml.exists(), f"missing canonical yaml: {real_yaml}"
    out = allocate(real_yaml)
    assert "anthropic" in out
    assert "google_vertex" in out
    assert "openai" in out
    assert "github_models" in out
    # Total daily budget = $5.00; sum should be <= $5.00 (clamps may reduce)
    assert sum(out.values()) <= 5.01  # tolerance for float
