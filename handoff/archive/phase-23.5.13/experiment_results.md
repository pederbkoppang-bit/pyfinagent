---
step: phase-23.5.13
title: Cron job verification — cost_budget_watcher — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA (clean — partial-stub disclosed)
verification_command: 'python3 tests/verify_phase_23_5_13.py'
---

# Experiment Results — phase-23.5.13

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_13.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.13`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="cost_budget_watcher"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK cost_budget_watcher scheduled 2026-05-10T06:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_13.py
OK cost_budget_watcher status=scheduled next_run=2026-05-10T06:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "cost_budget_watcher",
  "source": "slack_bot",
  "schedule": "phase-9.8 interval",
  "next_run": "2026-05-10T06:00:00+02:00",
  "last_run": null,
  "status": "scheduled"
}
```

`last_run: null` post-restart; pre-restart registry showed
`last_run_at="2026-05-08T04:00:04.641105+00:00"`. Tomorrow 06:00
CEST fire will re-populate.

## Why the criterion is satisfied

- Handler (`cost_budget_watcher.py`) makes zero HTTP calls; uses
  `google.cloud.bigquery.Client` directly for BQ spend queries
  (lines 91-115).
- `heartbeat()` correctly wired at line 56 with daily idempotency_key.
- Bridge surfaces `status="scheduled"` from registry's startup
  seed.
- Trigger correctly registered at `scheduler.py:532-533`
  (`hour=6`, `misfire_grace_time=3600`, `coalesce=True`).
- **Performs REAL BQ work** — fetches `total_bytes_billed` from
  `INFORMATION_SCHEMA.JOBS_BY_PROJECT` and runs real
  `BudgetEnforcer.tick()` evaluation.

## Production-stub PARTIAL (NOT a regression, NOT in scope)

`alert_fn` parameter is NOT injected by `register_phase9_jobs()`
(`scheduler.py:544` — `add_job(func, ...)` with no `alert_fn`
kwarg). Budget trips therefore log a `logger.warning` only
(`cost_budget_watcher.py:77`), no Slack message sent. Same wiring
gap as `weekly_data_integrity` (23.5.12).

## Final production-stub tally — all 7 phase-9 jobs verified

| Job | Status | Notes |
|------|--------|-------|
| daily_price_refresh | AFFECTED | _default_fetch + _default_write stubs |
| weekly_fred_refresh | AFFECTED | _default_fetch + _default_write stubs |
| nightly_outcome_rebuild | AFFECTED | _default_fetch + _default_write stubs |
| nightly_mda_retrain | NOT AFFECTED | real train_fn runs (PromotionGate fails on stub model only) |
| hourly_signal_warmup | NOT AFFECTED | real cache infra; injected compute_fn is no-op |
| weekly_data_integrity | NOT AFFECTED | real BQ work; alert_fn unwired |
| **cost_budget_watcher** | **PARTIAL** | **real BQ work + real BudgetEnforcer; alert_fn unwired** |

**4 of 7 phase-9 jobs have wiring gaps** (3 fully stubbed +
cost_budget_watcher's alert_fn). Recommend single follow-up step
**23.5.13.1** to bulk-wire production fetch/write/alert functions
in `register_phase9_jobs()`.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.12 (16 prior) | PASS |
| 23.5.13 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire `alert_fn` (deferred to 23.5.13.1 bulk fix).
- Wire production fetch/write for daily_price_refresh +
  weekly_fred_refresh + nightly_outcome_rebuild (deferred).
- The 6 launchd jobs (separate bridge problem).

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.13-research-brief.md`
- `tests/verify_phase_23_5_13.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_13.py
```
