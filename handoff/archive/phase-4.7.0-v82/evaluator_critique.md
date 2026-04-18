# Evaluator Critique -- Cycle 82 / phase-4.8 step 4.8.5

Step: 4.8.5 Champion-challenger gradual rollout (5 / 25 / 100%)

## Dual-evaluator run (parallel, anti-rubber-stamp)

## qa-evaluator: PASS

6-point substantive review:
1. **allocation_pct_field_present**: optimizer_best.json line 34
   has `"allocation_pct": 0.05`. Confirmed.
2. **promotion_gate_enforced**: evaluate_stage decision tree
   requires all of psr_ok + pbo_ok + kill_ok + can_advance_by_time
   for the "advance" branch; failures route to hold / regress /
   demote. Audit tests (c) + (d) prove this.
3. **initial_live_allocation_5pct_default**: scripts/risk/
   promotion_gate.py uses `STAGES[0]=0.05` (not a hardcoded
   0.05 elsewhere). Gated by `if "allocation_pct" not in
   existing:` so re-runs preserve existing stage.
4. **Preservation-on-update**: update_optimizer_best reads full
   blob, only sets 3 keys (allocation_pct, stage, optional
   challenger_run_id). Original 7 keys retained.
5. **Decision tree branches real**: traced benign (advance ->
   0.25), psr_failure (regress to 0.05 + psr_below_champion
   reason), insufficient_days (hold + days_at_stage_insufficient
   reason). All dynamic.
6. **No hardcoded 5%**: STAGES[0] is the source; changing
   STAGES[0] would ripple to the default.

## harness-verifier: PASS

6/6 mechanical checks green:
- Immutable verification exits 0.
- Audit clean with 4/4 teeth tests PASS.
- Artifact structure correct; allocation_pct=0.05, stage=0.
- optimizer_best.json has all 9 required keys.
- **Preservation-on-rerun test**: force allocation_pct=0.25; run
  dry-run again; confirm snapshot shows 0.25 preserved (not reset
  to 0.05). Then restore to 0.05 for subsequent runs.
- **Mutation test**: disable PSR check (`psr_ok = True`); audit
  catches with rc=1. File restored.

## Decision: PASS (evaluator-owned)

Both evaluators substantively green with preservation test +
mutation test proving the gate has real teeth and the default
5% canary is principled, not hardcoded.
