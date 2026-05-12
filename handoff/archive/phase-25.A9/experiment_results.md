---
step: phase-25.A9
cycle: 58
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A9.py'
title: Fix cache-write cost premium 1.25x to 2.0x (P1 prerequisite for 25.A8)
---

# Experiment Results — phase-25.A9

**Action:** GENERATE (real code change). 1-line constant change + comment update.

## Code change

`backend/agents/cost_tracker.py:147`:

```diff
- cache_write_cost = cache_creation * pricing[0] * 1.25 / 1_000_000
+ cache_write_cost = cache_creation * pricing[0] * 2.0 / 1_000_000
```

Comment block updated to reference phase-25.A9 attribution and the underlying llm_client.py:773-779 1h-TTL opt-in.

## New verifier: `tests/verify_phase_25_A9.py`

5 claims (stdlib-only):
1. `cache_write_premium_constant_equals_2_0`
2. `old_1_25_multiplier_removed_from_calculation`
3. `phase_25_A9_attribution_comment_present`
4. `cost_tracker_py_syntax_clean`
5. `cost_tracker_math_for_4096_token_write_at_5_per_mtok_equals_0_04096_usd`

## Verbatim verifier output

```
=== phase-25.A9 (cache-write premium) verifier ===
  [PASS] cache_write_premium_constant_equals_2_0
  [PASS] old_1_25_multiplier_removed_from_calculation
  [PASS] phase_25_A9_attribution_comment_present
  [PASS] cost_tracker_py_syntax_clean
  [PASS] cost_tracker_math_for_4096_token_write_at_5_per_mtok_equals_0_04096_usd
PASS (5/5) EXIT=0
```

5/5 PASS.

## Hypothesis verdict
CONFIRMED. 1-line fix as specified by phase-24.9 audit F-1. Math round-trip: 4096 tokens × $5/MTok × 2.0 ÷ 1M = $0.04096 (was $0.0256 with 1.25x — a 60% under-report closure).

## Live-check
Per masterplan: "Re-process recent BQ cost rows; verify cumulative cost increases by expected ratio". To be filled in post-deployment with a re-aggregation script comparing pre/post 25.A9 cost-attribution.

## Next phase
EVALUATE — Q/A pending. After PASS: log Cycle 58, flip 25.A9, start 25.A8 (depends on 25.A9 for accurate cost data).
