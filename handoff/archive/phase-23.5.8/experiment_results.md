---
step: phase-23.5.8
title: Cron job verification — weekly_fred_refresh — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — with adjacent production-stub finding)
verification_command: 'python3 tests/verify_phase_23_5_8.py'
---

# Experiment Results — phase-23.5.8

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_8.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.8`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_fred_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK weekly_fred_refresh scheduled 2026-05-10T02:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_8.py
OK weekly_fred_refresh status=scheduled next_run=2026-05-10T02:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "weekly_fred_refresh",
  "source": "slack_bot",
  "schedule": "phase-9.3 cron",
  "next_run": "2026-05-10T02:00:00+02:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Weekly refresh of FRED macro series"
}
```

## Why the criterion is satisfied

- No HTTP calls to backend in the handler (`weekly_fred_refresh.py`
  is pure Python).
- `heartbeat()` has no URL.
- Cross-process push uses `_HEARTBEAT_URL` (127.0.0.1) — no Docker
  alias.
- Bridge surfaces `status="scheduled"` from the registry's startup
  seed.
- CronTrigger computed `next_run` correctly.

## Adjacent finding (NOT in scope, NOT a regression)

Per researcher's audit of `register_phase9_jobs()` at
`scheduler.py:535-548`: the function calls `scheduler.add_job(func, ...)`
where `func = getattr(mod, "run")` — **without partial-applying
`fetch_fn` or `write_fn`**. So the scheduler fires `run()` with zero
kwargs, which means:

- `_default_fetch` STUB is active in production (returns
  `{s: [] for s in series}` — empty dict).
- `_default_write` STUB is active in production (returns
  `len(rows)` only).

Result: the job runs cleanly through `heartbeat()`, posts
`status="ok"`, **but does NOT actually fetch from FRED nor write to
BQ.** Same coverage gap as 23.5.7's daily_price_refresh.

This pattern affects **all 7 phase-9 jobs**. The fix is wiring
production fetch/write functions in `register_phase9_jobs()`. Out
of scope here; recommend a single follow-up step (e.g.,
`phase-23.5.13.1`) at the end of the phase-9 block.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.7.1 | PASS |
| 23.5.8 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire production fetch_fn/write_fn (deferred — single step at end of phase-9 block).
- Tune the weekly fire time.
- Touch FRED API logic.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.8-research-brief.md`
- `tests/verify_phase_23_5_8.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_8.py
```
