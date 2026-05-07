# Phase-23.3.1 audit findings — Main APScheduler

**Cycle date:** 2026-05-07
**Scope:** the 2 main-process APScheduler jobs (paper_trading_daily +
ticket queue process_batch).

## Verdict: PASS WITH FIX

Both jobs are firing correctly as designed. One operator-visibility
issue was found and fixed in this same step (anonymous job id).

## Per-job

### `paper_trading_daily`
- **Registered:** `backend/api/paper_trading.py:911-923` (cron mon-fri
  14:00 ET, `id=_scheduler_job_id`, `replace_existing=True`).
- **Live next_run:** `2026-05-08T14:00:00-04:00` ✓ (tomorrow's 18:00 UTC
  cycle scheduled).
- **Live status:** `scheduled`.
- **Pre-fix description on /cron:** `_scheduled_run` (qualname).
- **Post-fix description on /cron:** `Paper trading daily run` ✓.

### Ticket queue `process_batch`
- **Registered:** `backend/main.py:217` inside the `lifespan` async
  context manager. 5-second interval.
- **Live next_run:** `2026-05-07T21:25:01.201963+02:00` ✓ (≤5s from
  audit time).
- **Live status:** `scheduled`.
- **Pre-fix id:** APScheduler-generated UUID hex
  `2db2dd276ba94305a9aec11a5bb58f6c`. Description was
  `lifespan.<locals>.process_batch`.
- **Post-fix id:** `ticket_queue_process_batch` ✓. Description:
  `Ticket queue batch processor` ✓.

## What was changed

```diff
 # backend/main.py:217
-processor_job = queue_scheduler.add_job(process_batch, 'interval', seconds=5)
+processor_job = queue_scheduler.add_job(
+    process_batch, 'interval', seconds=5,
+    id="ticket_queue_process_batch",
+    name="Ticket queue batch processor",
+    replace_existing=True,
+)

 # backend/api/paper_trading.py:911-923
 _scheduler.add_job(
     _scheduled_run, "cron",
     hour=settings.paper_trading_hour, minute=0,
     day_of_week="mon-fri",
     timezone=ZoneInfo("America/New_York"),
     id=_scheduler_job_id,
+    name="Paper trading daily run",
     replace_existing=True,
 )
```

## Sibling concerns

None unhandled. Researcher (adaf19a0d83c77106) grepped the repo for
`2db2dd276ba94305a9aec11a5bb58f6c` and found zero callsites depending
on the auto-UUID — safe rename.

## Live HTTP verification

```
$ curl http://127.0.0.1:8000/api/jobs/all | jq '.jobs | map(select(.source=="main_apscheduler"))'
[
  {
    "id": "paper_trading_daily",
    "source": "main_apscheduler",
    "schedule": "cron[day_of_week='mon-fri', hour='14', minute='0']",
    "next_run": "2026-05-08T14:00:00-04:00",
    "status": "scheduled",
    "description": "Paper trading daily run"
  },
  {
    "id": "ticket_queue_process_batch",
    "source": "main_apscheduler",
    "schedule": "interval[0:00:05]",
    "next_run": "2026-05-07T21:25:01+02:00",
    "status": "scheduled",
    "description": "Ticket queue batch processor"
  }
]
```

## Q/A

Per the same-session pragmatism documented in phase-23.3.0 honest
disclosures, no separate Q/A subagent was spawned for this small
audit step. The deterministic verifier (`tests/verify_phase_23_3_1.py`,
4 checks including a live HTTP probe) is the canonical gate.
