---
step: phase-10.7.4
title: Cron Budget Allocator (slot governance authority)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-10.7
deliverables:
  - .claude/cron_budget.yaml (extend in-place to schema v3)
  - backend/meta_evolution/cron_allocator.py
  - scripts/meta/validate_cron_budget.py
  - tests/meta_evolution/test_cron_allocator.py
---

# Sprint Contract -- phase-10.7.4

## Research-gate summary

`handoff/current/phase-10.7.4-research-brief.md`. tier=moderate, 7 in-full,
21 URLs, recency scan present, gate_passed=true. Internal: 12 files
inspected. Decisive sources: Justitia (arXiv 2510.17015 2025), Agent
Contracts (arXiv 2601.08815v3 2026), HiveMind (arXiv 2604.17111 2026),
OSTEP ch.9 (Lottery/Stride scheduling), Dordal Computer Networks ch on
WFQ/DRR.

## Hypothesis

Proportional Weighted-Fair-Queueing (stride-style: `allocation_i =
(weight_i / sum_weights) * total_budget`) is the right algorithm for
pyfinagent's 15-slot, ~$5/day token budget. Min/max clamps protect
against starvation + monopolisation; disabled slots are excluded from
the weight denominator (Agent Contracts 2026 pool-reclaim pattern);
priority-tier vocabulary already exists in `.claude/cron_budget.yaml`
v2 (`reserved/high/medium/low`).

## Schema bump (cron_budget.yaml v2 -> v3)

Backwards-compatible additions only. Existing v2 fields preserved
unchanged. New fields default in code (allocator + validator both apply
sane defaults so the file remains parseable as v2 or v3).

New top-level:
- `total_daily_token_budget: 100000` (~$0.30/day at gemini-2.0-flash)

New per-slot (all optional, defaults injected by allocator):
- `min_tokens_per_fire: int` (default 1000)
- `max_tokens_per_fire: int` (default 50000)
- `category: research | monitoring | trading | maintenance` (default
  inferred from existing `priority`: reserved->trading, high/medium->
  research, low->maintenance)
- `enabled: bool` (default true)

## Algorithm (stride/WFQ proportional)

```
PRIORITY_WEIGHTS = {"reserved": 10, "high": 6, "medium": 3, "low": 1}

def allocate(yaml_path, total_budget) -> dict[str, int]:
  cfg = yaml.safe_load(...)
  enabled = [s for s in cfg["slots"] if s.get("enabled", True)]
  weights = {s["job_name"]: PRIORITY_WEIGHTS[s["priority"]] for s in enabled}
  W = sum(weights.values())
  alloc = {}
  for s in enabled:
    raw = (weights[s.job_name] / W) * total_budget
    lo = s.get("min_tokens_per_fire", 1000)
    hi = s.get("max_tokens_per_fire", 50000)
    alloc[s.job_name] = int(max(lo, min(hi, round(raw))))
  return alloc
```

No renormalisation pass: clamp may cause `sum(alloc) != total_budget`,
but the test asserts the invariant `min(total_budget * 0.9, sum(alloc))
<= sum(alloc) <= total_budget * 1.1` to permit clamp drift while
catching gross math errors.

## Validator (8 checks, exit 0/1)

`python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml`:
1. YAML loads without error
2. Top-level required keys: `version`, `total_slots`, `slots`
3. Per-slot required keys: `slot_id`, `job_name`, `priority`, `cadence`, `surface`
4. Priority in `{reserved, high, medium, low}`
5. No duplicate `job_name`
6. `total_daily_token_budget` (if present) is positive int
7. `min_tokens_per_fire <= max_tokens_per_fire` per slot (where both present)
8. `total_slots` matches `len(slots)` (existing soft invariant)

CLI: `--quiet` for exit-code-only mode (no stdout). Default mode prints
each check + verdict.

## Plan steps

1. Extend `.claude/cron_budget.yaml`:
   - Add top-level `total_daily_token_budget: 100000`
   - Add `category`, `enabled`, `min_tokens_per_fire`, `max_tokens_per_fire`
     to a representative subset of slots (do NOT need to backfill all 15
     since defaults handle the rest)
   - Bump `version: 3`
2. Write `backend/meta_evolution/cron_allocator.py` (~150 LOC):
   - `PRIORITY_WEIGHTS` constant
   - `DEFAULT_MIN_TOKENS_PER_FIRE = 1000`, `DEFAULT_MAX_TOKENS_PER_FIRE = 50000`
   - `Allocation` dataclass (job_name, weight, raw_budget, clamped_budget, was_clamped)
   - `allocate(yaml_path, total_budget) -> dict[str, int]` (top-level API)
   - `compute_allocations(yaml_path, total_budget) -> list[Allocation]`
     (richer return for test introspection)
   - Pure: no I/O outside yaml file read.
3. Write `scripts/meta/__init__.py` (empty package marker) +
   `scripts/meta/validate_cron_budget.py` (~120 LOC):
   - argparse: `path` positional + `--quiet` flag
   - Returns 0 on all 8 checks pass; 1 on any failure with helpful stderr
4. Write `tests/meta_evolution/test_cron_allocator.py` (~180 LOC, 9
   tests minimum):
   - allocation tests (proportional, sum invariant, clamp respect,
     disabled excluded, single job)
   - validator tests (subprocess: valid yaml exits 0; bad yaml exits 1)
5. Run immutable verification:
   `python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml &&
    python -m pytest tests/meta_evolution/test_cron_allocator.py -v`
6. Spawn Q/A.

## Success Criteria (verbatim, immutable from masterplan)

```
python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml && python -m pytest tests/meta_evolution/test_cron_allocator.py -v
```

Plus:
- `cron_budget_yaml_extends_v2`: existing 15 slots preserved; only
  additive fields added; `version: 3` bump.
- `validator_passes_on_real_yaml`: exit 0 on `.claude/cron_budget.yaml`.
- `validator_rejects_bad_yaml`: exit 1 on synthetic bad fixture (test).
- `allocator_proportional`: 3 enabled slots with priorities high/medium/
  low get budgets in ratio 6:3:1 (within clamp tolerance).
- `allocator_clamps_min_max`: a slot with `min_tokens_per_fire=10000`
  receives at least 10000 tokens even if raw share would be smaller.
- `allocator_excludes_disabled`: an `enabled: false` slot is absent
  from the result; remaining slots renormalise.
- `tests_pass`: pytest >= 9/9 PASS.

## What Q/A must audit

1. Immutable verification command exits 0 (deterministic).
2. cron_budget.yaml is still loadable as v2 (back-compat);
   `version: 3` bump is the only schema-version change.
3. No existing slot's required fields removed or renamed.
4. Validator catches all 8 documented failure modes (spot-check at
   least 3 via synthetic fixtures inside the test file).
5. Allocator is pure: no logging, no BQ, no network. Confirmed by
   inspection of imports in `cron_allocator.py`.
6. PRIORITY_WEIGHTS uses the documented `{reserved:10, high:6,
   medium:3, low:1}` mapping (cite HiveMind 2026 + existing yaml
   priority vocabulary).
7. Disabled-slot exclusion implemented per Agent Contracts 2026
   pool-reclaim pattern (filter BEFORE computing weight denominator).
8. `scripts/meta/__init__.py` exists so the script directory is a
   proper Python package (or at least importable as a module path).
9. No mutation to `STRATEGY_REGISTRY`, `_PARAM_BOUNDS`, scheduler.py,
   cost_tracker.py, or any other engine code.
10. Pattern matches 10.7.1/10.7.2/10.7.3 (dataclass + module-level
    constants + factory + zero-I/O outside the yaml read).
