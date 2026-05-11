---
step: phase-23.5.10
title: Cron job verification — hourly_signal_warmup — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA (clean — TRUE liveness signal)
verification_command: 'python3 tests/verify_phase_23_5_10.py'
---

# Experiment Results — phase-23.5.10

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_10.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.10`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="hourly_signal_warmup"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK hourly_signal_warmup ok 2026-05-10T01:05:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_10.py
OK hourly_signal_warmup status=ok next_run=2026-05-10T01:05:00+02:00
EXIT=0
```

`status="ok"` reflects a real fire (the job ran at HH:05 and the
heartbeat listener recorded `status="ok"` to the registry). The
hourly cadence means the registry status stays warm continuously.

## Live `/api/jobs/all` entry

```json
{
  "id": "hourly_signal_warmup",
  "source": "slack_bot",
  "schedule": "phase-9.5 interval",
  "next_run": "2026-05-10T01:05:00+02:00",
  "last_run": "2026-05-09T22:05:00...",
  "status": "ok",
  "description": "Hourly cache warmup for enrichment signals"
}
```

## Why the criterion is satisfied (TRUE liveness)

- Handler (`hourly_signal_warmup.py`) makes zero HTTP calls; pure
  in-process loop over watchlist + injectable compute_fn +
  injectable dict cache.
- `heartbeat()` correctly wired with hourly idempotency_key.
- Bridge surfaces real `status="ok"` from registry (not a startup
  seed — this job has actually fired since the 23:24 CEST daemon
  restart).
- Live log evidence (researcher cited): `2026-05-10 00:05:00 status:
  started` → `status: ok, duration_s: 0.00021...`.

## Trigger correction (cosmetic)

Schedule label says "phase-9.5 interval" but the actual trigger is
`cron(minute=5)` (every hour at HH:05) per `scheduler.py:526-527`.
Wall-clock aligned, NOT a strict interval. Same cosmetic class as
the "phase-9.X cron" labels in `_SLACK_BOT_JOBS`. Deferred to the
schedule-label cosmetic fix step.

## Production-stub gap (NOT a regression, NOT in scope)

Default `compute_signal_fn = lambda t: {"score": 0.0}` so the cache
fills with placeholders. The infrastructure is real (heartbeat,
idempotency, watchlist load, cache write), but the SIGNAL is a
no-op stub. Same pattern as 23.5.7 / 23.5.8. Not affecting the
immutable criterion (which tests scheduling); deferred to bulk
fix at end of phase-9 block.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.9 (13 prior) | PASS |
| 23.5.10 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Wire production compute_signal_fn.
- Fix the schedule-label cosmetic.
- Touch the 3 sibling phase-9 jobs.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.10-research-brief.md`
- `tests/verify_phase_23_5_10.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_10.py
```
