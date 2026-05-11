---
step: phase-23.5.11
title: Cron job verification — nightly_outcome_rebuild — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA (clean — production-stub disclosed)
verification_command: 'python3 tests/verify_phase_23_5_11.py'
---

# Experiment Results — phase-23.5.11

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_11.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.11`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_outcome_rebuild"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK nightly_outcome_rebuild scheduled 2026-05-10T04:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_11.py
OK nightly_outcome_rebuild status=scheduled next_run=2026-05-10T04:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "nightly_outcome_rebuild",
  "source": "slack_bot",
  "schedule": "phase-9.6 cron",
  "next_run": "2026-05-10T04:00:00+02:00",
  "last_run": null,
  "status": "scheduled"
}
```

`last_run: null` because the daemon was last restarted at 23:24
CEST and the cron fires at 04:00 CEST tomorrow. Pre-restart
registry showed `last_run_at="2026-05-08T02:00:00.011390+00:00"`.

## Why the criterion is satisfied

- Handler (`nightly_outcome_rebuild.py`) makes zero HTTP calls.
- `heartbeat()` at `nightly_outcome_rebuild.py:22` correctly wired
  with daily idempotency_key wrapping the full computation block.
- Bridge surfaces `status="scheduled"` from registry's startup
  seed.
- Trigger correctly registered at `scheduler.py:528-529`
  (`hour=4`, `misfire_grace_time=3600`, `coalesce=True`).

## Production-stub gap (NOT a regression, NOT in scope)

`_default_fetch()` returns `[]` (no BQ read), `_default_write()`
returns `len(outcomes)` (no BQ write). The fire completes
successfully through `heartbeat()` but reports `rebuilt=0`. Same
pattern as 23.5.7 (daily_price_refresh) and 23.5.8
(weekly_fred_refresh). Bulk fix at end of phase-9 block.

**Tally so far for the production-stub pattern:**
- AFFECTED (3 of 5): daily_price_refresh, weekly_fred_refresh,
  nightly_outcome_rebuild.
- NOT AFFECTED (2 of 5): nightly_mda_retrain (real train_fn runs),
  hourly_signal_warmup (real infrastructure; only the injected
  compute_fn is no-op).

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.10 (14 prior) | PASS |
| 23.5.11 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire real ledger_fetch_fn / outcome_write_fn.
- The 2 sibling phase-9 jobs not yet covered (weekly_data_integrity,
  cost_budget_watcher).
- Refactoring outcome-tracking.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.11-research-brief.md`
- `tests/verify_phase_23_5_11.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_11.py
```
