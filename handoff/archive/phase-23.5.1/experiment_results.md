---
step: phase-23.5.1
title: Cron job verification — paper_trading_daily — experiment results
date: 2026-05-08
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_1.py'
---

# Experiment Results — phase-23.5.1

## What was done

Verification-only step. **No code changes** to backend or frontend.
One artifact produced:

1. `tests/verify_phase_23_5_1.py` — replayable Python verifier that
   probes `/api/jobs/all` and re-runs the immutable assertions.

## Verification command — verbatim from `.claude/masterplan.json::23.5.1`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="paper_trading_daily"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result (run 2026-05-08)

```
$ python3 -c '...verbatim immutable command...'
OK paper_trading_daily scheduled 2026-05-08T14:00:00-04:00
EXIT=0
```

```
$ python3 tests/verify_phase_23_5_1.py
OK paper_trading_daily status=scheduled next_run=2026-05-08T14:00:00-04:00
EXIT=0
```

## Live `/api/jobs/all` entry (verbatim)

```json
{
  "id": "paper_trading_daily",
  "source": "main_apscheduler",
  "schedule": "cron[day_of_week='mon-fri', hour='14', minute='0']",
  "next_run": "2026-05-08T14:00:00-04:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Paper trading daily run"
}
```

## Why the criterion is structurally satisfied

Per `backend/api/cron_dashboard_api.py:160-188` (specifically line 174):

```python
"status": "scheduled" if nrt is not None else "paused",
```

For `main_apscheduler` jobs, the dashboard derives `status` from
`scheduler.get_job(job_id).next_run_time`. The literal value
`"manifest"` is reserved by `_static_to_dict()` for out-of-process
manifest entries (slack_bot + launchd) and is never produced for
`main_apscheduler` jobs. Therefore the criterion `status !=
"manifest"` is structurally guaranteed for this source as long as
the scheduler is initialized.

`next_run is not None` translates directly to
`job.next_run_time is not None` in APScheduler 3.x (cited:
https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html).
`None` only occurs when the job is paused; an active cron trigger
always has a computed future fire time.

## `last_run: null` is by design (NOT a verifier failure)

Per code comment at `cron_dashboard_api.py:173`:
```python
# APScheduler doesn't expose this; phase-2 if needed
"last_run": None,
```

APScheduler 3.x's `Job` object has no `last_run_time` attribute.
The main scheduler in `backend/api/paper_trading.py` does NOT have
an `EVENT_JOB_EXECUTED` listener wired (the slack-bot scheduler at
`backend/slack_bot/scheduler.py:12-14,122-124` is the only one
that does). The masterplan criterion does NOT require `last_run`
population, and the contract's anti-pattern guards explicitly
forbid amending it.

## Verdict against the immutable criterion

> "status != 'manifest' AND next_run is not None"

Actual: `status="scheduled"`, `next_run="2026-05-08T14:00:00-04:00"`.

**The criterion IS met. Verifier exits 0. No regression vs phase-23.3.1
(which audited the same job's wiring on 2026-05-07).**

## Findings to surface to the operator

1. **`paper_trading_daily` is healthy** — registered, scheduled for
   the next weekday 14:00 ET fire, will compute the daily cycle as
   designed.
2. **`last_run: null` for ALL `main_apscheduler` jobs** is a known
   architectural gap (no `EVENT_JOB_EXECUTED` listener on the main
   scheduler). This is a **dashboard observability gap**, not a
   bug. Researcher's documented path to fix (separate phase, NOT
   this step): wire `scheduler.add_listener(fn,
   EVENT_JOB_EXECUTED)` in `backend/api/paper_trading.py` after
   `init_scheduler`, persist `event.scheduled_run_time` in an
   in-memory dict keyed by `event.job_id`, surface via
   `_job_to_dict`. Recommend opening this as a follow-up step in
   phase-23.5 or beyond.
3. **Sibling `ticket_queue_process_batch`** in 23.5.2 is expected
   to behave identically (same source, same dashboard derivation).

## What this step does NOT do

- Wire the missing event listener.
- Trigger `paper_trading_daily` manually to test last_run
  population (out of scope; immutable criterion does not require
  it).
- Investigate the 11 slack_bot or 6 launchd jobs (separate
  substeps).
- Modify `cron_dashboard_api.py` or `paper_trading.py`.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.1 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.1-research-brief.md` — researcher.
- `tests/verify_phase_23_5_1.py` — replayable verifier.

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_1.py
```

Exits 0 on PASS; non-zero with a labelled error if backend is
down, the job is missing, status went `manifest`, or next_run goes
null.
