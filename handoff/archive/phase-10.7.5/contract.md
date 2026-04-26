---
step: phase-10.7.5
title: API-credit reallocator with per-provider floors
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-10.7
deliverables:
  - .claude/provider_budget.yaml (NEW)
  - backend/meta_evolution/provider_rebalancer.py (NEW, ~180 LOC)
  - tests/meta_evolution/test_provider_rebalancer.py (NEW, ~250 LOC, 12 tests)
---

# Sprint Contract -- phase-10.7.5

## Research-gate summary

`handoff/current/phase-10.7.5-research-brief.md`. tier=moderate, 8 in-full,
18 URLs, recency scan present, gate_passed=true. 9 internal files inspected.

Decisive references:
- WFQ (Wikipedia), Max-min fairness (Wikipedia), Dordal Computer Networks
- Enterprise LLM Gateway pattern (Kamya Shah blog 2026)
- StackSpend three-layer governance model (2026)
- LLMGateway weighted scoring docs

## Hypothesis

Per-provider floor + WFQ proportional allocation + max-min progressive-fill
rebalance is the right algorithm. Mirror `cron_allocator.py` pattern
(@dataclass + module-level constants + factory + zero-I/O outside
yaml read), but with FLOAT USD math (not int tokens) and an additional
`rebalance()` function for surplus redistribution.

## Concrete plan

### File 1: `.claude/provider_budget.yaml` (NEW)

```yaml
version: 1
total_daily_usd_budget: 5.0  # matches kill-switch trip in cost_budget_api.py:53
providers:
  - name: anthropic
    priority_weight: 10
    min_floor_usd: 1.00
    max_ceiling_usd: 4.00
    enabled: true
  - name: google_vertex
    priority_weight: 6
    min_floor_usd: 0.50
    max_ceiling_usd: 3.00
    enabled: true
  - name: openai
    priority_weight: 2
    min_floor_usd: 0.10
    max_ceiling_usd: 1.50
    enabled: true
  - name: github_models
    priority_weight: 1
    min_floor_usd: 0.00
    max_ceiling_usd: 0.50
    enabled: true
```

Sum of floors: 1.60 < 5.00 (feasible).

### File 2: `backend/meta_evolution/provider_rebalancer.py` (NEW, ~180 LOC)

Pure module mirroring `cron_allocator.py`. Structure:

```python
"""phase-10.7.5 API-Credit Reallocator with per-provider floors.

Weighted-fair-queueing allocator over .claude/provider_budget.yaml,
with per-provider floor + ceiling clamps + reactive surplus
rebalance. Pure data; no I/O outside yaml read.
"""
PROVIDER_BUDGET_DEFAULT_TOTAL_USD = 5.0

@dataclass(frozen=True)
class Allocation:
    provider: str
    weight: int
    raw_budget: float       # WFQ proportional share before clamping (USD)
    clamped_budget: float   # after floor/ceiling clamp (USD)
    floor: float
    ceiling: float
    was_clamped: bool

def _load_yaml(path) -> dict
def _enabled_providers(cfg) -> list[dict]
def _validate_feasibility(providers, total_budget) -> None  # raises ValueError if sum(floors) > total

def compute_allocations(yaml_path, total_budget=None) -> list[Allocation]
def allocate(yaml_path, total_budget=None) -> dict[str, float]

def rebalance(allocations, used_usd_by_provider) -> dict[str, float]:
    """Two-pass max-min progressive fill.
    Pass 1: each provider gets min(used, ceiling) — locks in actual demand
    Pass 2: surplus from underspent providers redistributes proportionally
            to other providers within their ceilings.
    """
```

USD floats throughout; `round(x, 6)` for monetary precision (NOT `int()`).

### File 3: `tests/meta_evolution/test_provider_rebalancer.py` (~250 LOC, 12 tests)

Mirror `test_cron_allocator.py` structure (`_write_yaml()` + `_provider()` helpers, no live deps). 12 cases:
1. test_proportional_basic
2. test_disabled_excluded
3. test_min_floor_enforced
4. test_max_ceiling_enforced
5. test_single_provider_full_budget
6. test_allocate_uses_yaml_default_budget
7. test_sum_floors_gt_total_raises
8. test_rebalance_underspent_surplus_redistributes
9. test_rebalance_overspent_capped_at_ceiling
10. test_compute_allocations_returns_rich_data
11. test_all_providers_disabled_returns_empty
12. test_float_precision_not_int

## Success Criteria (verbatim, immutable from masterplan)

```
python -m pytest tests/meta_evolution/test_provider_rebalancer.py -v
```

Plus:
- `module_imports`: `from backend.meta_evolution.provider_rebalancer import allocate, rebalance, Allocation` exits 0
- `pure_module`: no logging / BQ / network imports (only stdlib + pyyaml + dataclasses)
- `tests_pass`: 12/12 (or more if useful)
- `yaml_loads`: `python -c "import yaml; yaml.safe_load(open('.claude/provider_budget.yaml'))"` succeeds
- `feasibility_check`: yaml sum(floors) <= total_daily_usd_budget

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0 (12+ tests pass).
2. `provider_rebalancer.py` is PURE (only stdlib + pyyaml).
3. `Allocation` dataclass uses float types for USD (not int).
4. `_validate_feasibility` raises ValueError when sum(floors) > total.
5. `rebalance()` implements two-pass max-min progressive fill (under-spent surplus redistributes to others).
6. Pattern matches cron_allocator.py (dataclass + module-level constants + factory + zero-I/O).
7. .claude/provider_budget.yaml feasible (sum(floors)=1.60 < 5.00).
8. No backend service code modified outside the new module.
