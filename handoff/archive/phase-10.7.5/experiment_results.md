---
step: phase-10.7.5
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - .claude/provider_budget.yaml (NEW, ~36 lines)
  - backend/meta_evolution/provider_rebalancer.py (NEW, ~210 LOC)
  - tests/meta_evolution/test_provider_rebalancer.py (NEW, ~270 LOC, 17 tests)
---

# Experiment Results -- phase-10.7.5

## What was done

Implemented API-credit reallocator with per-provider USD floors and
two-pass max-min progressive-fill rebalance. Pure-Python module
mirroring `cron_allocator.py` (10.7.4) pattern, with USD floats
(not int tokens).

## Deliverables

### `.claude/provider_budget.yaml` (NEW, 36 lines)

4 providers with version=1 schema:
- `anthropic` — weight=10, floor=$1.00, ceiling=$4.00 (primary harness + Layer-2)
- `google_vertex` — weight=6, floor=$0.50, ceiling=$3.00 (28 Layer-1 agents + fallback)
- `openai` — weight=2, floor=$0.10, ceiling=$1.50 (secondary)
- `github_models` — weight=1, floor=$0.00, ceiling=$0.50 (free tier)

`total_daily_usd_budget: 5.0` (matches `cost_budget_api.py:53` kill-switch trip).

Sum of floors = $1.60 < total $5.00 (feasibility invariant holds).

### `backend/meta_evolution/provider_rebalancer.py` (NEW, ~210 LOC)

- `Allocation` frozen dataclass (provider/weight/raw_budget/clamped_budget/floor/ceiling/was_clamped)
- `compute_allocations()` — WFQ proportional allocation + floor/ceiling clamps + feasibility validation
- `allocate()` — top-level API returning `{provider: clamped_usd}`
- `rebalance()` — two-pass max-min progressive fill:
  - Pass 1: classify each provider as under-spent (locks at actual usage; contributes surplus) or demanding (provisional grant = clamped_budget; eligible for surplus)
  - Pass 2: distribute surplus to demanding providers proportionally by weight, capped at ceiling. Surplus does NOT flow back to contributors.

Pure module: only stdlib + pyyaml. USD floats throughout (`round(x, 6)`,
NOT `int()`). Validation raises `ValueError` on infeasible config or
floor > ceiling.

### `tests/meta_evolution/test_provider_rebalancer.py` (NEW, ~270 LOC, 17 tests)

12 from contract + 5 additional defensive tests:
1. test_proportional_basic
2. test_disabled_excluded
3. test_min_floor_enforced
4. test_max_ceiling_enforced
5. test_single_provider_full_budget
6. test_allocate_uses_yaml_default_budget
7. test_default_total_when_yaml_omits (defensive)
8. test_sum_floors_gt_total_raises
9. test_floor_gt_ceiling_raises (defensive)
10. test_compute_allocations_returns_rich_data
11. test_all_providers_disabled_returns_empty
12. test_float_precision_not_int
13. test_rebalance_underspent_surplus_redistributes
14. test_rebalance_overspent_capped_at_ceiling
15. test_rebalance_empty_allocations_returns_empty (defensive)
16. test_rebalance_no_surplus_passthrough (defensive)
17. test_real_yaml_loads (defensive: real .claude/provider_budget.yaml allocates cleanly)

## Verification (verbatim, immutable from masterplan)

```
$ python -m pytest tests/meta_evolution/test_provider_rebalancer.py -v
============================== 17 passed in 0.03s ==============================
```

## Cycle-2 fix applied during implementation

Initial pass-2 logic redistributed surplus to ALL providers with headroom
(including the under-spent contributor itself). Test failures surfaced this
in 2/17 tests on first run:
- `test_rebalance_underspent_surplus_redistributes`: anthropic (the surplus
  source) was wrongly receiving surplus back, getting $3.22 instead of $1.0
- `test_rebalance_overspent_capped_at_ceiling`: openai (under-spent) was
  receiving its own surplus back, getting $1.0 instead of $0.5

Fix: refactored pass-2 to only redistribute surplus to "demanding" providers
(those with `used >= clamped_budget`) — providers that maxed out their
budget and could use more. Under-spent providers exit the recipient list.
Both tests then PASS along with the other 15.

## Files touched

| Path | Action | Note |
|------|--------|------|
| `.claude/provider_budget.yaml` | CREATED | 36 lines, 4 providers |
| `backend/meta_evolution/provider_rebalancer.py` | CREATED | ~210 LOC pure module |
| `tests/meta_evolution/test_provider_rebalancer.py` | CREATED | ~270 LOC, 17 tests |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-10.7.5-research-brief.md` | created (researcher) | -- |

NO backend service code changes. NO new dependencies (pyyaml already installed).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | module imports cleanly | PASS |
| 2 | provider_rebalancer.py is PURE (only stdlib + pyyaml) | PASS |
| 3 | Allocation uses float USD (not int) | PASS |
| 4 | _validate_feasibility raises on infeasible | PASS |
| 5 | rebalance two-pass max-min progressive fill | PASS |
| 6 | pattern matches cron_allocator.py | PASS |
| 7 | yaml feasibility verified | PASS (sum 1.60 < 5.0) |
| 8 | 17/17 pytest | PASS |

## Honest disclosures

1. **17 tests vs 12 in contract** — added 5 defensive: yaml-default
   fallback, floor>ceiling, empty rebalance, no-surplus passthrough,
   real-yaml smoke. Floor exceeded; not a violation.

2. **Cycle-2 fix during impl, not separate Q/A retry.** Found by
   running tests immediately after first write; no Q/A spawn yet.
   Fix logged in this experiment_results explicitly per protocol.

3. **No reactive loop yet.** The `rebalance()` function is pure logic
   — caller (future scheduler/loop in 10.7.6) decides when to invoke
   it with current usage data. This cycle ships the math; wiring is
   10.7.6 scope.

4. **Pattern consistency with 10.7.4** — same Allocation dataclass
   shape + module-level constant + factory + zero-I/O outside yaml
   read. Test helpers (_write_yaml, _provider) mirror
   test_cron_allocator.py.

5. **No BQ table needed** — provider budget is stateless config; no
   migration script.

## Closes

Task list item #74. Masterplan step phase-10.7.5.

## Next

Spawn Q/A.
