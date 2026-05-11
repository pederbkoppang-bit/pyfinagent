---
step: phase-23.5.12
title: Cron job verification — weekly_data_integrity — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA (clean — TRUE liveness, real BQ work)
verification_command: 'python3 tests/verify_phase_23_5_12.py'
---

# Experiment Results — phase-23.5.12

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_12.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.12`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_data_integrity"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK weekly_data_integrity scheduled 2026-05-11T05:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_12.py
OK weekly_data_integrity status=scheduled next_run=2026-05-11T05:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "weekly_data_integrity",
  "source": "slack_bot",
  "schedule": "phase-9.7 cron",
  "next_run": "2026-05-11T05:00:00+02:00",
  "last_run": null,
  "status": "scheduled"
}
```

`last_run: null` because the daemon was last restarted at 23:24
CEST and `weekly_data_integrity` is a weekly cron — pre-restart
the registry showed `status="never_run"` (it had never fired).
Tomorrow's 05:00 CEST fire will be the first observed run.

## Why the criterion is satisfied (TRUE liveness, real work)

- Handler (`weekly_data_integrity.py`) makes zero HTTP calls.
- `heartbeat()` correctly wired at line 48 with weekly idempotency_key.
- Bridge surfaces `status="scheduled"` from registry's startup
  seed.
- Trigger correctly registered at `scheduler.py:531`.
- **Performs REAL BQ work** — queries `__TABLES__` for row counts
  (line 85-87), computes drift vs JSON snapshot (line 52),
  optionally calls `alert_fn` (not wired — see adjacent finding),
  saves updated snapshot (line 59-62).

## Adjacent finding (NOT a regression, NOT in scope)

`alert_fn` parameter is NOT wired by `register_phase9_jobs()`.
The job DETECTS row-count drift but cannot SLACK-ALERT because the
optional `alert_fn` is never injected. Same wiring-gap pattern as
the production-stub jobs (where `fetch_fn`/`write_fn` aren't
injected). Bulk fix at end of phase-9 block.

## Updated production-stub tally

After 6 of 7 phase-9 jobs verified:
- **AFFECTED (3 of 6):** daily_price_refresh, weekly_fred_refresh,
  nightly_outcome_rebuild — `_default_fetch`/`_default_write` stubs
  produce empty/no-op work.
- **NOT AFFECTED (3 of 6):** nightly_mda_retrain (real train_fn),
  hourly_signal_warmup (real cache infra; injected compute_fn is
  no-op), **weekly_data_integrity** (real BQ work; only `alert_fn`
  unwired).

Last to verify: cost_budget_watcher (23.5.13).

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.11 (15 prior) | PASS |
| 23.5.12 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire `alert_fn` so drift triggers Slack alerts.
- The 1 sibling phase-9 job remaining.
- Refactor drift detection or threshold.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.12-research-brief.md`
- `tests/verify_phase_23_5_12.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_12.py
```
