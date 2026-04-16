# Contract -- Phase 4.4.6.4: Rollback Plan

## Target
Checklist item 4.4.6.4: "Rollback plan: if live Sharpe < 0.5 in first 2 weeks -> stop signals, investigate"

## Success Criteria

1. `docs/ROLLBACK_PLAN.md` exists and documents:
   - SC1: Trigger condition (live Sharpe < 0.5, trailing 14-day window)
   - SC2: Stop-signals command with exact syntax
   - SC3: Re-approval gate (fresh 4.4.6.1 sign-off from Peder before restart)
   - SC4: Investigation checklist (what to check before restarting)
   - SC5: Re-run recipe for rehearsal
2. `scheduler.py` exports a `pause_signals()` function that shuts down the scheduler
3. Drill at `scripts/go_live_drills/rollback_plan_test.py`:
   - SC6: ROLLBACK_PLAN.md exists
   - SC7: Doc mentions Sharpe < 0.5 threshold
   - SC8: Doc mentions 14-day window
   - SC9: Doc mentions pause_signals or stop-signals command
   - SC10: Doc mentions Peder re-approval
   - SC11: `scheduler.py` has `pause_signals` function
   - SC12: `pause_signals` accesses the `_scheduler` global
   - SC13: Drill exits 0

## Rollback Plan
Revert the commit if the drill fails or the doc is insufficient.
