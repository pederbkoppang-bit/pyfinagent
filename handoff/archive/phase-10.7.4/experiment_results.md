---
step: phase-10.7.4
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - .claude/cron_budget.yaml (extended in-place to schema v3)
  - backend/meta_evolution/cron_allocator.py
  - scripts/meta/__init__.py
  - scripts/meta/validate_cron_budget.py
  - tests/meta_evolution/test_cron_allocator.py
---

# Experiment Results -- phase-10.7.4

## What was done

Implemented the Cron Budget Allocator (slot governance authority) per
the research-brief design. Proportional Weighted-Fair-Queueing
(stride-style) over the existing 15-slot `.claude/cron_budget.yaml`,
extended in place to schema v3 with backwards-compatible additive
fields. Pure-Python module + CLI validator + 17-test suite.

### Files touched

| Path | Action | Size |
|------|--------|------|
| `.claude/cron_budget.yaml` | EDITED (v2 -> v3, +18 lines: top-level token budget + optional per-slot fields on slots 4, 6, 15) | 184 lines |
| `backend/meta_evolution/cron_allocator.py` | CREATED | 156 lines |
| `scripts/meta/__init__.py` | CREATED | 0 lines (package marker) |
| `scripts/meta/validate_cron_budget.py` | CREATED | 191 lines |
| `tests/meta_evolution/test_cron_allocator.py` | CREATED | 220 lines |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

NO mutation of any backend service, frontend, or engine code.
NO BQ migration (allocator is pure-Python; no table needed).

## Verification (verbatim, immutable from masterplan)

```
$ python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml && python -m pytest tests/meta_evolution/test_cron_allocator.py -v
  [PASS] YAML loads
  [PASS] top-level required keys present
  [PASS] per-slot required keys present
  [PASS] priorities in ['high', 'low', 'medium', 'reserved']
  [PASS] no duplicate job_name
  [PASS] total_daily_token_budget is positive int -- = 100000
  [PASS] min_tokens_per_fire <= max_tokens_per_fire
  [PASS] total_slots matches len(slots) -- = 15
validate_cron_budget: PASS (.claude/cron_budget.yaml)
============================= test session starts ==============================
collected 17 items

tests/meta_evolution/test_cron_allocator.py::test_proportional_basic PASSED
tests/meta_evolution/test_cron_allocator.py::test_disabled_excluded PASSED
tests/meta_evolution/test_cron_allocator.py::test_min_floor_enforced PASSED
tests/meta_evolution/test_cron_allocator.py::test_max_ceiling_enforced PASSED
tests/meta_evolution/test_cron_allocator.py::test_single_slot_gets_full_budget PASSED
tests/meta_evolution/test_cron_allocator.py::test_allocate_uses_yaml_default_budget PASSED
tests/meta_evolution/test_cron_allocator.py::test_invalid_priority_raises PASSED
tests/meta_evolution/test_cron_allocator.py::test_min_gt_max_raises PASSED
tests/meta_evolution/test_cron_allocator.py::test_compute_allocations_returns_richer_data PASSED
tests/meta_evolution/test_cron_allocator.py::test_priority_weights_constants PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_real_yaml_exits_0 PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_duplicate_job_name_exits_1 PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_bad_priority_exits_1 PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_missing_file_exits_2 PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_min_gt_max_exits_1 PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_quiet_flag PASSED
tests/meta_evolution/test_cron_allocator.py::test_validator_total_slots_mismatch_exits_1 PASSED

============================== 17 passed in 0.29s ==============================
```

**Result: PASS.** Validator's 8 checks all PASS on real YAML.
Pytest 17/17 PASS (10 allocator + 7 validator-via-subprocess).
Compound `&&` exits 0.

## Bottom-line

`backend/meta_evolution/cron_allocator.py::allocate(yaml_path, total_budget)`
returns `{job_name: clamped_token_budget}` for the day. Disabled slots
are excluded from the weight denominator (Agent Contracts 2026
pool-reclaim pattern). Min/max clamps protect against starvation +
monopolisation. Validator (`scripts/meta/validate_cron_budget.py`) is
the gate -- 8 checks, three exit codes (0=pass / 1=fail / 2=fs-error),
quiet flag for CI.

Schema v3 is backwards-compatible: every v2 slot still parses; new
per-slot fields (`category`, `enabled`, `min_tokens_per_fire`,
`max_tokens_per_fire`) are all optional with allocator-supplied
defaults.

### Real-yaml allocation preview

For `.claude/cron_budget.yaml` (15 slots, mix of reserved/high/medium/low,
total_daily_token_budget=100000, slot 15 disabled):

- 14 enabled slots compete for 100000 tokens
- Reserved slots (1-5) get the largest share via weight=10
- Slot 15 (`reserved_headroom`, `enabled=false`) gets nothing -- correctly
  excluded

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | cron_budget_yaml_extends_v2 | PASS | version: 3, 15 slots preserved, only additive fields |
| 2 | validator_passes_on_real_yaml | PASS | exit 0; all 8 checks PASS |
| 3 | validator_rejects_bad_yaml | PASS | 5 rejection tests cover dup names, bad priority, missing file, min>max, total_slots mismatch |
| 4 | allocator_proportional | PASS | test_proportional_basic: 6:3:1 ratio for high/medium/low |
| 5 | allocator_clamps_min_max | PASS | test_min_floor_enforced + test_max_ceiling_enforced |
| 6 | allocator_excludes_disabled | PASS | test_disabled_excluded: enabled=false slot absent from result |
| 7 | tests_pass | PASS | 17/17 (target was >=9) |

## Honest disclosures

1. **Sum-invariant intentionally not enforced.** When min/max clamps
   activate, `sum(allocate())` may differ from `total_budget`. This is
   documented in the `allocate()` docstring and the contract's "no
   renormalisation pass" line. The contract's tolerance band (90-110%)
   is NOT explicitly tested -- but the proportional_basic test
   (no clamps) does assert exact equality. Callers needing strict-sum
   should use `compute_allocations()` and post-process.

2. **17 tests, contract floor was 9.** Exceeded by 8. Not a
   contract violation; over-coverage. Subprocess validator tests add
   ~0.2s of CI time (acceptable).

3. **Only 3 of 15 slots got the new optional fields populated** in
   the YAML (slots 4, 6, 15). The other 12 use allocator defaults.
   This is intentional -- defaults work fine; populating all 15 would
   bloat the yaml without adding value. Future cycles can add per-slot
   tuning as needed.

4. **`scripts/meta/__init__.py` is empty.** Marker file only.
   The validator runs as a script (`python scripts/meta/validate_cron_budget.py`),
   not as an importable module, so a marker is sufficient.

5. **No connection to existing cost_tracker / cost_budget_api.**
   Per research-brief pitfall #1, the token allocator is orthogonal to
   the USD cap enforcer. The two layers run independently:
   - Allocator: pre-allocates tokens per job before execution.
   - Cost tracker: post-hoc USD accounting + circuit-break at $5/day.
   Combining them is explicitly out of scope (would create double-trip
   risk where the USD watcher fires AND the allocator throttles).

6. **PRIORITY_WEIGHTS pinned to {reserved:10, high:6, medium:3, low:1}.**
   Test `test_priority_weights_constants` is a regression guard so a
   sneaky edit gets caught.

7. **Frozen Allocation dataclass.** Same defensive pattern as 10.7.3
   archetype_library; mutation after construction raises
   FrozenInstanceError.

## No-regressions

- `git status --short`: only `.claude/cron_budget.yaml`,
  `backend/meta_evolution/cron_allocator.py`,
  `scripts/meta/__init__.py`,
  `scripts/meta/validate_cron_budget.py`,
  `tests/meta_evolution/test_cron_allocator.py`,
  plus rolling handoff/* files.
- No backend service, frontend, scheduler, cost-tracker, or other
  code touched.
- `python -c "import yaml; yaml.safe_load(open('.claude/cron_budget.yaml'))"`
  exits 0 (back-compat parse check).

## Closes

- masterplan step **phase-10.7.4** (immutable verification PASS)

## Next

Spawn Q/A to audit deterministic checks + LLM judgment. If PASS:
log + flip + continue with phase-10.7.5 (API-credit reallocator with
per-provider floors).
