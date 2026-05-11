---
step: phase-23.5.3
title: Cron job verification — morning_digest — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (with prominent false-positive finding)
verification_command: 'python3 tests/verify_phase_23_5_3.py'
---

# Experiment Results — phase-23.5.3

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_3.py` — replayable verifier mirroring
   the 23.5.1 / 23.5.2 / 23.5.2.5 pattern (urllib + asserts).

## Verification command — verbatim from `.claude/masterplan.json::23.5.3`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="morning_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result (run 2026-05-09)

```
$ <verbatim immutable command>
OK morning_digest scheduled 2026-05-09T08:00:00-04:00
EXIT=0

$ python tests/verify_phase_23_5_3.py
OK morning_digest status=scheduled next_run=2026-05-09T08:00:00-04:00
EXIT=0
```

## Live `/api/jobs/all` entry (post-bridge)

```json
{
  "id": "morning_digest",
  "source": "slack_bot",
  "schedule": "cron daily morning_digest_hour:00 ET",
  "next_run": "2026-05-09T08:00:00-04:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Slack morning digest (top movers + holdings recap)"
}
```

The `last_run` is null because the slack-bot daemon was restarted
twice today (once at 19:30 for 23.5.2.5 deployment and again at
09:49 for 23.5.2.6). `morning_digest` fires at 8 AM ET — the next
fire is tomorrow morning.

## Why the criterion is satisfied

- `status="scheduled"` (not `"manifest"`) — set by the
  phase-23.5.2.5 startup state-push. Bridge merge at
  `cron_dashboard_api.py:208-227` surfaces it from the registry.
- `next_run="2026-05-09T08:00:00-04:00"` — computed by APScheduler
  `CronTrigger(hour=8, minute=0, timezone=ZoneInfo("America/New_York"))`
  and pushed by the slack-bot's `_seed_next_run_registry` at
  `scheduler.py:85+`. ISO format includes the `-04:00` offset for
  Eastern Daylight Time. Researcher confirmed APScheduler 3.11.0+
  (2024) `zoneinfo` migration handles ET correctly; no DST skip
  bug for hour=8.

The verification command exits 0; the immutable criterion is
**met as written**.

## CRITICAL FINDING — criterion is a false positive for morning_digest

The criterion says nothing about whether `_send_morning_digest`
actually delivers a Slack message. Per researcher's audit:

`_send_morning_digest` (`backend/slack_bot/scheduler.py:146+`) uses
`_BACKEND_URL = "http://backend:8000"` (Docker-compose DNS alias)
to call:
- `{_BACKEND_URL}/api/portfolio/performance`  (line ~211)
- `{_BACKEND_URL}/api/reports/?limit=5`       (line ~214)

This is the SAME bug class as the watchdog had pre-phase-23.5.2.6.
On the Mac host process, `backend` does not resolve. Both
`httpx.AsyncClient.get()` calls raise `ConnectError`. The handler's
`except Exception: logger.exception(...)` (lines 226-227) is fail-
open — no re-raise. APScheduler fires `EVENT_JOB_EXECUTED`
regardless, so `_aps_to_heartbeat` records `status="ok"`. The
dashboard would show green after the next 8 AM fire even though
the operator received no Slack message.

**Anthropic immutable-criteria doctrine forbids amending the
criterion.** This step's verdict is PASS on the criterion as
written. The actual correctness bug is captured for the next
substep.

## Planned follow-up — phase-23.5.3.1

After this step closes, insert phase-23.5.3.1 in the masterplan:

> **23.5.3.1 — Fix Docker-alias hostname in `_send_morning_digest`
> + `_send_evening_digest`**
>
> Apply the same `127.0.0.1` repointing pattern that 23.5.2.6 used
> for the watchdog. Either introduce a single new constant
> `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"` and have all
> three handlers (watchdog/morning/evening) use it, OR refactor
> `_BACKEND_URL` to be conditional on environment. Add tests that
> assert each digest function probes localhost (not the Docker
> alias).
>
> Run BEFORE 23.5.4 (which would otherwise also be a false-
> positive PASS for evening_digest).

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| `tests/verify_phase_23_5_1.py` (paper_trading_daily) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2.py` (ticket_queue_process_batch) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2_5.py` (heartbeat bridge) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2_6.py` (watchdog) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_3.py` (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Fix the Docker-alias bug in `_send_morning_digest` /
  `_send_evening_digest` (deferred to phase-23.5.3.1).
- Add tests for `_send_morning_digest` httpx calls (deferred to
  23.5.3.1).
- Modify `cron_dashboard_api.py`, `job_status_api.py`,
  `scheduler.py`, or any other backend file.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.3 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.3-research-brief.md` — researcher.
- `tests/verify_phase_23_5_3.py` — replayable verifier.

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_3.py
```
