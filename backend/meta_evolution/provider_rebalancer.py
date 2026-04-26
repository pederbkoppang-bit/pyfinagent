"""phase-10.7.5 API-Credit Reallocator with per-provider USD floors.

Weighted-fair-queueing (stride-style) allocator over the per-provider
daily USD budget defined in `.claude/provider_budget.yaml`. Adds a
`rebalance()` function that does two-pass max-min progressive-fill
surplus redistribution when a provider is over- or under-spent.

Algorithm (per Wikipedia WFQ + max-min fairness; mirrors the pattern
in `backend/meta_evolution/cron_allocator.py`):

    weight_i      = provider.priority_weight  (int)
    raw_budget_i  = (weight_i / sum_weights) * total_budget   (float USD)
    allocation_i  = clamp(raw_budget_i, floor_i, ceiling_i)

Disabled providers (`enabled: false`) are excluded from the weight
denominator so their notional share redistributes to active providers
(work-conserving WFQ).

Feasibility invariant: `sum(min_floor_usd) <= total_daily_usd_budget`.
Enforced at load time with `ValueError`.

Pure module: no logging, no BQ, no network. Only stdlib + pyyaml.
USD floats throughout (NOT int tokens like `cron_allocator.py`).
Use `round(x, 6)` for monetary precision; never `int()`.

Pattern mirrors `cron_allocator.py` (10.7.4) and `archetype_library.py`
(10.7.3): @dataclass(frozen=True) + module-level constants + factory
+ zero-I/O outside the YAML read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PROVIDER_BUDGET_DEFAULT_TOTAL_USD = 5.0

REQUIRED_PROVIDER_KEYS: frozenset[str] = frozenset(
    {"name", "priority_weight", "min_floor_usd", "max_ceiling_usd"}
)
USD_PRECISION = 6  # round(x, 6) for $0.000001 dollar precision


@dataclass(frozen=True)
class Allocation:
    """One provider's allocation outcome.

    Float USD throughout (not int tokens). `was_clamped` flips True if
    floor or ceiling activated (raw_budget != clamped_budget).
    """

    provider: str
    weight: int
    raw_budget: float
    clamped_budget: float
    floor: float
    ceiling: float
    was_clamped: bool


def _load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load YAML. Raises FileNotFoundError or yaml.YAMLError on failure."""
    p = Path(yaml_path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _enabled_providers(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return providers with enabled != False (default true)."""
    raw = cfg.get("providers") or []
    return [p for p in raw if p.get("enabled", True)]


def _validate_feasibility(
    providers: list[dict[str, Any]], total_budget: float
) -> None:
    """Sum of floors MUST be <= total budget. Raises ValueError otherwise."""
    floors_sum = sum(float(p.get("min_floor_usd", 0.0)) for p in providers)
    if floors_sum > total_budget:
        raise ValueError(
            f"infeasible provider budget: sum(min_floor_usd)={floors_sum:.2f} "
            f"> total_daily_usd_budget={total_budget:.2f}"
        )


def _validate_provider_keys(provider: dict[str, Any]) -> None:
    """Each provider entry must have all required keys."""
    missing = REQUIRED_PROVIDER_KEYS - set(provider.keys())
    if missing:
        raise ValueError(
            f"provider {provider.get('name', '<?>')} missing required "
            f"keys: {sorted(missing)}"
        )
    floor = float(provider["min_floor_usd"])
    ceiling = float(provider["max_ceiling_usd"])
    if floor > ceiling:
        raise ValueError(
            f"provider {provider['name']}: min_floor_usd ({floor}) > "
            f"max_ceiling_usd ({ceiling})"
        )
    weight = int(provider["priority_weight"])
    if weight < 1:
        raise ValueError(
            f"provider {provider['name']}: priority_weight must be >= 1, got {weight}"
        )


def compute_allocations(
    yaml_path: str | Path,
    total_budget: Optional[float] = None,
) -> list[Allocation]:
    """Return per-provider Allocation list (rich introspection).

    If `total_budget` is None, reads `total_daily_usd_budget` from the YAML
    or falls back to `PROVIDER_BUDGET_DEFAULT_TOTAL_USD`.
    """
    cfg = _load_yaml(yaml_path)
    if total_budget is None:
        total_budget = float(
            cfg.get("total_daily_usd_budget", PROVIDER_BUDGET_DEFAULT_TOTAL_USD)
        )

    enabled = _enabled_providers(cfg)
    if not enabled:
        return []

    for p in enabled:
        _validate_provider_keys(p)
    _validate_feasibility(enabled, total_budget)

    weights = {p["name"]: int(p["priority_weight"]) for p in enabled}
    sum_weights = sum(weights.values())
    if sum_weights <= 0:
        raise ValueError(
            f"sum of priority weights is {sum_weights}; cannot allocate"
        )

    out: list[Allocation] = []
    for p in enabled:
        name = p["name"]
        w = weights[name]
        raw = (w / sum_weights) * total_budget
        floor = float(p["min_floor_usd"])
        ceiling = float(p["max_ceiling_usd"])
        clamped = round(max(floor, min(ceiling, raw)), USD_PRECISION)
        was_clamped = round(clamped, USD_PRECISION) != round(raw, USD_PRECISION)
        out.append(
            Allocation(
                provider=name,
                weight=w,
                raw_budget=round(raw, USD_PRECISION),
                clamped_budget=clamped,
                floor=floor,
                ceiling=ceiling,
                was_clamped=was_clamped,
            )
        )
    return out


def allocate(
    yaml_path: str | Path,
    total_budget: Optional[float] = None,
) -> dict[str, float]:
    """Top-level API: return {provider: clamped_usd_budget}.

    Disabled providers absent from result. Sum may differ from total_budget
    when min/max clamps activate -- intentional (clamps are hard floors
    and ceilings, not advisory). Mirrors cron_allocator.allocate behavior.
    """
    return {a.provider: a.clamped_budget for a in compute_allocations(yaml_path, total_budget)}


def rebalance(
    allocations: list[Allocation],
    used_usd_by_provider: dict[str, float],
) -> dict[str, float]:
    """Two-pass max-min progressive-fill surplus redistribution.

    Pass 1: classify each provider as either "under-spent" (used <
    clamped_budget; surplus contributor) or "demanding" (used >=
    clamped_budget; surplus recipient candidate). Under-spent
    providers lock in at their actual usage; demanding providers
    provisionally lock at min(clamped_budget, ceiling). Surplus =
    sum of (clamped_budget - used) for under-spent.

    Pass 2: distribute surplus to demanding providers that still
    have headroom (provisional < ceiling), proportionally by weight.
    Surplus does NOT flow back to under-spent contributors (they
    proved they don't need more) and does NOT exceed any ceiling.

    Pure function. Returns {provider: new_usd_budget}. No I/O.
    """
    if not allocations:
        return {}

    pass1: dict[str, float] = {}
    surplus = 0.0
    demanding: list[Allocation] = []

    for a in allocations:
        used = float(used_usd_by_provider.get(a.provider, 0.0))
        if used >= a.clamped_budget:
            # Demanding more than budget: provisional grant = clamped_budget
            # (capped at ceiling, which is already enforced in clamped_budget).
            # Eligible for surplus only if there's headroom up to ceiling.
            granted = min(a.clamped_budget, a.ceiling)
            pass1[a.provider] = round(granted, USD_PRECISION)
            if granted < a.ceiling:
                demanding.append(a)
        else:
            # Under-spent: lock at actual usage; the rest becomes surplus.
            pass1[a.provider] = round(used, USD_PRECISION)
            surplus += a.clamped_budget - used

    if surplus <= 0 or not demanding:
        return pass1

    sum_w = sum(a.weight for a in demanding)
    if sum_w <= 0:
        return pass1

    out = dict(pass1)
    for a in demanding:
        share = (a.weight / sum_w) * surplus
        new_total = min(a.ceiling, pass1[a.provider] + share)
        out[a.provider] = round(new_total, USD_PRECISION)
    return out
