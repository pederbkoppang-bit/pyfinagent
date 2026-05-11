---
step: phase-23.5.2
title: Cron job verification — ticket_queue_process_batch — experiment results
date: 2026-05-08
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_2.py'
---

# Experiment Results — phase-23.5.2

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_2.py` — replayable verifier mirroring
   the 23.5.1 pattern (urllib + asserts).

## Verification command — verbatim from `.claude/masterplan.json::23.5.2`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="ticket_queue_process_batch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result (run 2026-05-08)

```
$ python3 -c '...verbatim immutable command...'
OK ticket_queue_process_batch scheduled 2026-05-08T18:38:51.633914+02:00
EXIT=0
```

```
$ python3 tests/verify_phase_23_5_2.py
OK ticket_queue_process_batch status=scheduled next_run=2026-05-08T18:39:11.633914+02:00
EXIT=0
```

(The `next_run` value advances by ~5s between successive probes —
expected for an IntervalTrigger with `seconds=5`.)

## Live `/api/jobs/all` entry

```json
{
  "id": "ticket_queue_process_batch",
  "source": "main_apscheduler",
  "schedule": "interval[0:00:05]",
  "next_run": "2026-05-08T18:38:51.633914+02:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Ticket queue batch processor"
}
```

## Why the criterion is structurally satisfied (stronger than 23.5.1)

For an `IntervalTrigger` registered without `end_date` (per
`backend/main.py:197-231`):

1. **`status="manifest"` impossible** — `cron_dashboard_api.py:174`
   derives `"scheduled" if nrt is not None else "paused"`.
   `"manifest"` is reserved exclusively for `_static_to_dict()` (out-
   of-process entries). Same as 23.5.1.

2. **`next_run is not None` is tautological** — APScheduler 3.x
   `IntervalTrigger.get_next_fire_time()` only returns `None` when
   `end_date` is exceeded (cited:
   https://apscheduler.readthedocs.io/en/3.x/modules/triggers/interval.html).
   With no `end_date` configured, the trigger always has a future
   fire time. Verified live (next_run advances by 5s between
   probes).

3. **The real discriminating gate is `j is not None`** — the job
   must be registered with the scheduler. The verification command
   checks this first via `j = next(...) ; assert j is not None`.

All three conditions are satisfied in the live response.

## `last_run: null` is by design (NOT a verifier failure)

Same architectural gap as 23.5.1: `cron_dashboard_api.py:173`
comment `# APScheduler doesn't expose this; phase-2 if needed`. No
`EVENT_JOB_EXECUTED` listener wired on the main scheduler. Not
required by the masterplan criterion.

## Adjacent finding (researcher's brief, NOT a regression)

APScheduler defaults `coalesce=True` + `misfire_grace_time=1s`
(documented at
https://apscheduler.readthedocs.io/en/3.x/userguide.html). For a
60-second agent call (per the ticket_queue handler design), this
swallows ~12 missed 5-second fires. **By design** — tickets are not
lost; they remain `OPEN` in SQLite (`backend/db/tickets_db.py:403`)
and are retried on the next successful batch.

FD-leak guard from phase-23.1.19 (`contextlib.closing()` at
`backend/services/ticket_queue_processor.py:43`) covers this job's
SQLite connection lifecycle. Regression test:
`tests/db/test_tickets_db_no_fd_leak.py` (100-iter, asserts net FD
delta ≤ 5).

## Verdict against the immutable criterion

> "status != 'manifest' AND next_run is not None"

Actual: `status="scheduled"`, `next_run` populated and advancing.

**The criterion IS met. Verifier exits 0. No regression vs phase-23.3.1
(which audited both paper_trading_daily and ticket_queue_process_batch
on 2026-05-07).**

## Findings to surface to the operator

1. **`ticket_queue_process_batch` is healthy** — registered with
   `AsyncIOScheduler`, fires every 5 seconds, FD-leak-safe.
2. **Same `last_run: null` architectural gap as 23.5.1** — applies
   to all `main_apscheduler` jobs. Documented in the harness log.
3. **`coalesce=True` swallows fires** when handler exceeds interval
   — by design for queue-pull pattern; not a defect.

## What this step does NOT do

- Wire the missing event listener.
- Tune `coalesce` / `misfire_grace_time` / `max_instances`.
- Refactor the ticket-queue handler.
- Investigate the 17 sibling jobs.
- Modify `cron_dashboard_api.py` or `main.py`.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.2 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.2-research-brief.md` — researcher.
- `tests/verify_phase_23_5_2.py` — replayable verifier.

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_2.py
```

Exits 0 on PASS; non-zero with a labelled error on backend-down,
job-missing, status-manifest, or next-run-null.
