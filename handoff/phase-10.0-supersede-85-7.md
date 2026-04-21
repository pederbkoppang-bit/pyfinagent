# phase-8.5.7 "Overnight orchestration cron" -- SUPERSEDE LOG

**Date:** 2026-04-20
**Superseded by:** phase-10.1 Sprint calendar config (`backend/autoresearch/sprint_calendar.yaml`)

---

## What 8.5.7 was

Nightly APScheduler cron running ~100 autoresearch experiments per night,
budget-gated via `backend/autoresearch/budget.py`. See phase-8.5.7 scaffold
in `backend/autoresearch/cron.py` + `scripts/harness/autoresearch_cron_test.py`.

## Why it is retired

Phase-10 Recursive Evolution Loop changes the cadence model:

- **Before (phase-8.5.7):** unbounded nightly cron, up to 100 experiments.
- **After (phase-10):** 2 weekly slots -- Thursday batch trigger + Friday promotion gate -- governed by `backend/autoresearch/sprint_calendar.yaml`. Plus monthly Champion/Challenger Sortino gate (HITL, phase-10.6).

The phase-10 cadence is more disciplined:
- Explicit weekly ledger (`weekly_ledger.tsv`, phase-10.2).
- DSR + PBO gate on Friday (reuses phase-8.5.5 gate).
- Rollback kill-switch on challenger drawdown (phase-10.7).
- Monthly Sortino gate requires owner approval (HITL).

## Masterplan reconciliation

- phase-8.5.7 remains `status: done` in masterplan (historical record of the scaffold).
- This doc is the cross-reference from phase-8.5 to phase-10's new cadence.
- The APScheduler job registered in phase-8.5.7 may be de-registered by a future housekeeping patch; until then its idempotency key prevents double-runs.

## References

- `backend/autoresearch/cron.py` (phase-8.5.7 scaffold, retained)
- `backend/autoresearch/sprint_calendar.yaml` (phase-10.1, authoritative from 2026-04-20)
- `handoff/harness_log.md` phase-10.x cycle blocks

---

**Authored:** 2026-04-20 as part of phase-10.0 harness cycle.
