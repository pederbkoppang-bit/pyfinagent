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
| 86.92 | **PASS** (cycle 2; cycle 1 CONDITIONAL) | *None at PASS.* Cycle-1's cap was **not** an immutable-criterion miss — the verdict stated all 7 were "met on their letter". It was capped by one executed finding against criterion 5's spirit: *"after the fix the checker exits 0, AND its mutation cells are shown still to KILL — a green checker whose mutants now survive is worse than a red one"* — my newly-added positive control could not fail. | Cycle 1: (a) two blocks declared "verbatim" while eliding lines — truthful, but disclosure-incomplete; (b) provenance `86.28/86.37` vs measured `86.6/86.37`; (c) live_check born-inert marker left at IN PROGRESS. Cycle 2 (all NOTE, none capping): control covers 1 of the stripper's 2 operations; region-slicing mutant equivalent today; a value-tightening rule would still surface as the old prose signature; harness_log dates 86.17 to 08-10 vs masterplan flip 08-09. | Cycle 1, verbatim: *"inject the same comment INSIDE the region, e.g. before `const selfReported` → stripper ON false / OFF true"*. Applied exactly, plus a shared-slicing `verificationFieldsReadNoStrip()` so the differential isolates stripping alone. | 2 of 3 | 636,006 (190,482 research + 224,595 QA-1 + 220,929 QA-2) |
