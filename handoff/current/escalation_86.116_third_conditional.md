# ESCALATION -- step 86.116 -- THIRD CONSECUTIVE CONDITIONAL

**Sequence:** `[CONDITIONAL, CONDITIONAL, CONDITIONAL]`
(`wf_6c5d3dfc-43a`, `wf_10d2c895-28e`, `wf_62e2fe3c-126`)
**Attempts:** 4 of 5 -- **the budget is NOT the constraint.**
**Binding constraint:** CLAUDE.md F1. The next verdict is FAIL by rule.

## The product is correct and three independent evaluators said so

Every one of the three re-derived the headline numbers **from BigQuery
themselves** and all three reproduced exactly. Cycle 3 also ran its own 4-cell
mutation matrix via `sys.modules` injection and confirmed the fix; the one
survivor it found was a **declared-equivalent** mutant (dropping `sort_index`,
where both production queries already `ORDER BY date`).

**No production code has changed since the original fix.**
`backend/backtest/cache.py` is `9f5f1d67...` across all three cycles. Cycles 2
and 3 changed **evidence only**.

## What each cycle actually found -- none of it was a product defect

| cycle | finding |
|---|---|
| 1 | criterion 6 credited a **dead key** (`vol_barrier_multiplier`, zero readers, listed in `_DEAD_KEYS`); the re-runnable command **aborted** because `--base-rev` defaulted to `HEAD`, which the fix's own commit turned into the post-fix tree; the read-path fixtures used byte-identical twins so a value-keyed mutant stayed green |
| 2 | the tripwire added in cycle 1's fix was **vacuous** -- an `or` clause true on the unmutated tree short-circuited it, so it detected a *rename*, not a volatility term. *"The remedy for brittleness introduced the vacuity."* |
| 3 | the cap guard added in cycle 2's fix **could not fire** (`x < 3.0` where `x <= sqrt(2)`); an assertion credited with drift protection was an **algebraic identity**; and **"no restart pending" was FALSE** |

**Three cycles, and each capping finding was in a guard written to close the
previous one.** That pattern is the argument for `guardlib` in the next-session
goal: the tax is being paid per step instead of once.

## The one finding with operational consequence

**RESTART IS PENDING and I had said it was not.** uvicorn **pid 41635**, started
**2026-08-17T15:57:16Z**, no `--reload`, holds the **pre-fix**
`backend.backtest.cache` in `sys.modules` (module-level import at
`backtest_engine.py:25`), and `backend/api/backtest.py` runs `run_backtest`
in-process. **Every API-triggered backtest still reads duplicated frames.**
Fresh processes (harness, CLI, scripts) DO carry the fix. Corrected in
`experiment_results_86.116.md` and in `goal_next_2026-08-19.md`.

## State at the park -- all three fixed, unevaluated

- cap guard replaced by `vol_scale_is_unsaturated_on_both_sides`, and **proven to
  fire** on both of the evaluator's saturating cases while passing the real
  control (vol_scale 0.4487 / 0.3587);
- the identity assertion renamed `census_sql_is_internally_consistent`, false
  prose replaced with an explicit correction;
- the restart claim corrected, with pid and start time, plus a pending-restart list.

Evidence: **33 invariants**, **13 tests**, matrix **11/11 KILLED** across two
targets. No Q/A has seen any of it.

## Carried forward, not absorbed

`_volatility_identifiers` rejects only `vol`-named identifiers, so a `sigma`- or
`_atr_width`-named term would slip past. Cycle 3 confirmed it is **not** vacuous,
but it is narrower than its name. Widening it during EVALUATE would have been a
tree change mid-evaluation; it belongs to the next cycle or to `guardlib`.

## Operator options

1. **Accept and flip** on three evaluators' agreement that the product is
   correct and no criterion is unaddressed. Main cannot do this without a PASS.
2. **Authorise one attempt** knowing it returns FAIL by rule, to reset the
   counter, then a fifth to grade the fixes. Costs both remaining attempts.
3. **Leave parked.** The fix is committed and live for every fresh process.
4. **Decide the rule itself** -- this is the fourth step it has parked.
