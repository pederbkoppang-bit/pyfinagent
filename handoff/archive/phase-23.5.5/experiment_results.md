---
step: phase-23.5.5
title: Cron job verification — watchdog_health_check — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — with strong in-the-wild evidence)
verification_command: 'python3 tests/verify_phase_23_5_5.py'
---

# Experiment Results — phase-23.5.5

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_5.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.5`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="watchdog_health_check"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK watchdog_health_check ok 2026-05-09T22:50:21.067885+02:00
EXIT=0

$ python tests/verify_phase_23_5_5.py
OK watchdog_health_check status=ok next_run=2026-05-09T22:50:21.067885+02:00
EXIT=0
```

`status="ok"` (was `"scheduled"` post-restart, advanced to
`"ok"` after the first successful fire).

## Live `/api/jobs/all` entry

```json
{
  "id": "watchdog_health_check",
  "source": "slack_bot",
  "schedule": "interval watchdog_interval_minutes",
  "next_run": "2026-05-09T22:50:21.067885+02:00",
  "last_run": "2026-05-09T20:35:21+00:00",
  "status": "ok",
  "description": "Slack-bot self-watchdog (alerts on backend unreachability)"
}
```

## In-the-wild evidence (load-bearing for the spam-fix verification)

Per researcher's audit of `handoff/logs/slack_bot.log`:

- **Daemon restart:** 2026-05-09 10:20:21 CEST (Scheduler started).
- **First watchdog fire:** 10:35:21 CEST (+15 min after restart —
  APScheduler IntervalTrigger wait-one-interval default per
  jdhao Nov 2024).
- **Total fires logged:** 49 consecutive at 15-min intervals
  through 22:35 CEST. No gaps.
- **Slack posts (the user-visible signal):** **ZERO.**
  - No `Watchdog unhealthy transition`
  - No `Watchdog steady-unhealthy`
  - No `Watchdog recovery`
  - The state machine correctly classified every fire as
    `None→True` (first) then steady `True→True` (rest).

**The phase-23.5.2.6 spam fix is operating exactly as designed.**
The 15-minute Slack spam the operator complained about is gone.

## Why the criterion is satisfied

- `status="ok"` (not `"manifest"`) — set by the heartbeat listener
  after at least one successful fire reached the backend's
  `/api/jobs/heartbeat` endpoint.
- `next_run="2026-05-09T22:50:21.067885+02:00"` — the trigger
  recomputes after every fire; advancing per the 15-min interval.

The criterion is a TRUE liveness signal:
- A frozen scheduler cannot produce an advancing `next_run`.
- A dead listener cannot record `status="ok"`.
- A wrong-hostname watchdog (the pre-23.5.2.6 state) would have
  posted spam — the operator would already know.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 paper_trading_daily | PASS |
| 23.5.2 ticket_queue_process_batch | PASS |
| 23.5.2.5 heartbeat bridge | PASS |
| 23.5.2.6 watchdog spam fix | PASS (4/4) |
| 23.5.3 morning_digest liveness | PASS |
| 23.5.3.1 digest Docker-alias fix | PASS (4/4) |
| 23.5.4 evening_digest | PASS |
| 23.5.5 watchdog liveness | PASS (this step) |

## What this step does NOT do

- Tune the watchdog interval.
- Add meta-monitoring (watchdog of the watchdog).
- Touch the watchdog handler (already fixed in 23.5.2.6).
- Investigate the 12 sibling jobs.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.5-research-brief.md`
- `tests/verify_phase_23_5_5.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_5.py
```
