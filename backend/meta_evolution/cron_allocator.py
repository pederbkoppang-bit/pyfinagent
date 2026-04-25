"""phase-10.7.4 Cron Budget Allocator (slot governance authority).

Proportional Weighted-Fair-Queueing (stride-style) allocator for the
cron-job token budget defined in `.claude/cron_budget.yaml`.

Algorithm (per OSTEP ch.9 Lottery/Stride and Justitia 2510.17015 2025):

    weight_i      = PRIORITY_WEIGHTS[slot.priority]
    raw_budget_i  = (weight_i / sum_weights) * total_budget
    allocation_i  = clamp(raw_budget_i,
                          slot.min_tokens_per_fire,
                          slot.max_tokens_per_fire)

Disabled slots (`enabled: false`) are excluded from the weight
denominator so their notional share redistributes to active slots
(Agent Contracts arXiv 2601.08815 2026 pool-reclaim pattern).

Pure module: no logging, no BQ, no network. Reads one YAML file.
Pattern mirrors `archetype_library.py` (10.7.3) and
`alpha_velocity.py` (10.7.1): @dataclass + module-level constants +
factory + zero-I/O outside the YAML read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PRIORITY_WEIGHTS: dict[str, int] = {
    "reserved": 10,
    "high": 6,
    "medium": 3,
    "low": 1,
}

DEFAULT_MIN_TOKENS_PER_FIRE = 1000
DEFAULT_MAX_TOKENS_PER_FIRE = 50000
DEFAULT_TOTAL_DAILY_TOKEN_BUDGET = 100000

ALLOWED_PRIORITIES: frozenset[str] = frozenset(PRIORITY_WEIGHTS.keys())
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {"research", "monitoring", "trading", "maintenance"}
)


@dataclass(frozen=True)
class Allocation:
    """One slot's allocation outcome.

    Returned in the richer `compute_allocations` API for test
    introspection. The simpler `allocate()` reduces this to a
    `{job_name: clamped_budget}` dict.
    """

    job_name: str
    priority: str
    weight: int
    raw_budget: float
    clamped_budget: int
    was_clamped: bool
    min_floor: int
    max_ceiling: int


def _load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load YAML. Raises FileNotFoundError or yaml.YAMLError on failure."""
    p = Path(yaml_path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _enabled_slots(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return slots with enabled != False (default true)."""
    raw = cfg.get("slots") or []
    return [s for s in raw if s.get("enabled", True)]


def _slot_weight(slot: dict[str, Any]) -> int:
    """Look up the priority weight; KeyError on bad priority."""
    pri = slot.get("priority")
    if pri not in PRIORITY_WEIGHTS:
        raise KeyError(
            f"slot {slot.get('job_name', '<?>')} has invalid priority "
            f"{pri!r}; expected one of {sorted(PRIORITY_WEIGHTS)}"
        )
    return PRIORITY_WEIGHTS[pri]


def compute_allocations(
    yaml_path: str | Path,
    total_budget: Optional[float] = None,
) -> list[Allocation]:
    """Return per-slot Allocation list (rich introspection).

    If `total_budget` is None, uses `total_daily_token_budget` from the
    YAML, falling back to `DEFAULT_TOTAL_DAILY_TOKEN_BUDGET`.
    """
    cfg = _load_yaml(yaml_path)
    if total_budget is None:
        total_budget = float(
            cfg.get("total_daily_token_budget", DEFAULT_TOTAL_DAILY_TOKEN_BUDGET)
        )

    enabled = _enabled_slots(cfg)
    if not enabled:
        return []

    weights = {s["job_name"]: _slot_weight(s) for s in enabled}
    sum_weights = sum(weights.values())
    if sum_weights <= 0:
        raise ValueError(
            f"sum of priority weights is {sum_weights}; cannot allocate"
        )

    out: list[Allocation] = []
    for slot in enabled:
        name = slot["job_name"]
        w = weights[name]
        raw = (w / sum_weights) * total_budget
        lo = int(slot.get("min_tokens_per_fire", DEFAULT_MIN_TOKENS_PER_FIRE))
        hi = int(slot.get("max_tokens_per_fire", DEFAULT_MAX_TOKENS_PER_FIRE))
        if lo > hi:
            raise ValueError(
                f"slot {name}: min_tokens_per_fire ({lo}) > "
                f"max_tokens_per_fire ({hi})"
            )
        clamped = int(max(lo, min(hi, round(raw))))
        was_clamped = clamped != int(round(raw))
        out.append(
            Allocation(
                job_name=name,
                priority=slot["priority"],
                weight=w,
                raw_budget=raw,
                clamped_budget=clamped,
                was_clamped=was_clamped,
                min_floor=lo,
                max_ceiling=hi,
            )
        )
    return out


def allocate(
    yaml_path: str | Path,
    total_budget: Optional[float] = None,
) -> dict[str, int]:
    """Top-level API: return {job_name: clamped_token_budget} for the day.

    Disabled slots (enabled: false) are excluded from the result.
    Sum of allocations may differ from total_budget when min/max
    clamps are active -- this is intentional (clamps are hard floors
    and ceilings, not advisory). Callers needing the strict-sum
    invariant should use `compute_allocations()` and post-process.
    """
    return {a.job_name: a.clamped_budget for a in compute_allocations(yaml_path, total_budget)}
