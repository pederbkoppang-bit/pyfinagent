# Night diagnostics — overnight drain 2026-08-16 21:00 → 07:00 Oslo

Required by the goal's §S1: **diagnostics are the deliverable, not a byproduct.**
Every step gets a row whether it CLOSED or PARKED. A park with no diagnosis is a
wasted step.

Columns:

| field | meaning |
|---|---|
| step | masterplan id |
| verdict | the Q/A verdict actually returned (never my summary of it) |
| criterion missed | the IMMUTABLE criterion text, **quoted**, that the verdict says was not met |
| quality gap only | findings that were real but did NOT map to an immutable criterion |
| evaluator's named fix | the concrete repair the evaluator named |
| attempts | Q/A spawns consumed (cap 3, per R1) |
| tokens | measured from the run records, never estimated |

Baseline stamp: `/tmp/pyfin_night_start` = 1786906029 (2026-08-16 20:47:09 CEST).

---

## Preflight — PASSED

All four gates green, health 200, no merge conflicts. Verbatim tails:

```
verify_prompt_render_86_90.mjs     ALL GREEN: 95 passed, 0 failed
verify_research_gate_workflow.mjs  ALL GREEN: 124 passed, 0 failed
verify_escalation_86_78.mjs        ALL CHECKS PASS   (checks run 51, cardinality floor 49, failed 0)
verify_rail_retry.mjs              ALL GREEN: 38 passed, 0 failed
/api/health                        200
git status --short | grep -E "^(UU|AA|DD)"   ->  no merge conflicts
```

Known-red, confirmed as documented and NOT treated as a preflight failure:

```
verify_workflow_args_boundary.mjs  FAILED: 84 passed, 3 failed
```

---

## Rows

<!-- one row per step, appended as each step closes or parks -->

| step | verdict | criterion missed (quoted) | quality gap only | evaluator's named fix | attempts | tokens |
|---|---|---|---|---|---|---|
