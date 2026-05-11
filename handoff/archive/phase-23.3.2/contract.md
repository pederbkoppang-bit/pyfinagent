---
step: phase-23.3.2
title: Slack-bot core jobs audit -- wire heartbeat-push so /api/jobs/status reflects real fires
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_2.py'
research_brief: handoff/current/phase-23.3.2-external-research.md (also see phase-23.3.2-internal-codebase-audit.md)
---

# Contract — phase-23.3.2

## Hypothesis

Slack-bot process is running (PID 16385, since 2026-04-08). All 4
core jobs (morning_digest, evening_digest, watchdog_health_check,
prompt_leak_redteam) are registered at
`backend/slack_bot/scheduler.py:35-79`. But `/api/jobs/status`
returns all 7 phase-9 jobs as `never_run` despite a month of uptime,
because the slack-bot APScheduler events are NOT POSTed to the
`POST /api/jobs/heartbeat` endpoint that already exists.

Researcher (a258c450e44cd773d) confirmed the gap:
- `backend/slack_bot/job_runtime.py:83`'s `sink` defaults to
  `logger.info` rather than HTTP push.
- `backend/api/job_status_api.py:135-143`'s `POST /api/jobs/heartbeat`
  is fully functional, ready to receive events.
- `_JOB_NAMES` at `job_status_api.py:52-60` covers only the 7
  phase-9 ids; the 4 core ids are not pre-seeded.

Fix shape: add an APScheduler event listener in slack_bot/scheduler.py
that POSTs to the heartbeat endpoint on EVENT_JOB_EXECUTED /
EVENT_JOB_ERROR / EVENT_JOB_MISSED, AND extend `_JOB_NAMES` to include
the 4 core ids. Result: after the next fire of any of the 4 + 7 = 11
slack-bot-process jobs, `/api/jobs/status` shows real `last_run_at`,
`status`, and `last_error`. The /cron Jobs tab can then derive
liveness from real heartbeats instead of static manifest pills.

## Research-gate summary

Researcher (a258c450e44cd773d) returned `gate_passed: true`:
- 6 sources read in full (APScheduler FAQ + User Guide, Fowler
  Heartbeat pattern, microservices.io health-check, daily-devops
  Green-Dashboard-Dead-Application, Healthchecks.io)
- 17 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 -- no breaking changes
- 7 internal files inspected with file:line anchors
- Concrete recommendation: literal code snippet for 2-change fix

Key findings cited:
- APScheduler FAQ: shared jobstore across processes is unsafe;
  HTTP/RPC is the correct pattern for cross-process introspection.
- Fowler: heartbeat = periodic message proving liveness; canonical
  primitive.
- daily-devops: "Green Dashboard, Dead Application" anti-pattern --
  manifest-only without liveness signal.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `backend/slack_bot/scheduler.py` registers an APScheduler event
   listener (added after `_scheduler.start()` at line 81) that POSTs
   `{job, status, finished_at, error}` to
   `http://localhost:8000/api/jobs/heartbeat` on EVENT_JOB_EXECUTED,
   EVENT_JOB_ERROR, and EVENT_JOB_MISSED. The listener is fail-open
   (any HTTP/exception is swallowed; the scheduler must not break).
2. `backend/api/job_status_api.py::_JOB_NAMES` is extended to
   include `morning_digest`, `evening_digest`, `watchdog_health_check`,
   `prompt_leak_redteam`. The pre-seeded registry now has 11 rows
   (was 7).
3. After the slack-bot process is restarted (operator action),
   `/api/jobs/status` returns 11 rows. (We can't test post-fire
   live without waiting for a real cron tick; instead, this phase
   provides a unit test that simulates a fire via the listener and
   asserts a row updates from `never_run` to `ok`.)
4. Regression test `tests/services/test_slack_bot_heartbeat_push.py`
   - mocks `httpx.post` and triggers the listener with a fake
     EVENT_JOB_EXECUTED; asserts the POST was made with the right
     URL + payload shape.
   - asserts the listener swallows httpx.RequestError (fail-open).
   - asserts that calling `record_heartbeat({...})` for a core job
     id (e.g., morning_digest) updates `_registry` from
     `never_run` to `ok`.
5. `python tests/verify_phase_23_3_2.py` exits 0.
6. `python -c "import ast; ast.parse(...)"` passes for both modified
   files.
7. The audit deliverable
   `handoff/current/phase-23.3.2-audit-findings.md` documents the
   gap, the fix, and the operator-restart step required to activate
   the heartbeat-push for the running slack-bot process.

## Plan steps

1. Edit `backend/slack_bot/scheduler.py` to import APScheduler events,
   define `_aps_to_heartbeat(event)`, and `add_listener` after
   `_scheduler.start()`. Use `httpx.Client` (sync; the listener is
   called from APScheduler's executor, not asyncio).
2. Edit `backend/api/job_status_api.py:_JOB_NAMES` to add the 4 core
   ids with phase comments.
3. Add `tests/services/test_slack_bot_heartbeat_push.py` (3 tests).
4. Add `tests/verify_phase_23_3_2.py` (deterministic checks).
5. Write `handoff/current/phase-23.3.2-audit-findings.md`.
6. Append `harness_log.md` AFTER PASS.

## Operator-restart caveat

**This phase ships the wiring but does NOT restart the slack-bot
process.** The slack-bot daemon at PID 16385 has been running since
2026-04-08 with the OLD code; it cannot pick up the new event
listener until restarted (`pkill -f "slack_bot.app" && python -m
backend.slack_bot.app &` or equivalent). The audit-findings doc
states this explicitly. The verifier therefore tests the wiring via
unit-test simulation, not a live end-to-end fire.

## Out of scope

- Restarting the slack-bot process (operator job; this phase ships
  the code).
- Changing the manifest pill rendering on /cron (UI choice; can be
  derived later from `/api/jobs/status` once heartbeats flow).
- Adding `/healthz` or `/live` endpoint to the slack-bot process
  itself (it has no inbound HTTP server; would need to add Bolt
  health route or a separate uvicorn — bigger change, separate phase).
- The `SLACK_CHANNEL_ID` silent-skip guard at scheduler.py:28-30
  noted by researcher as a "main hidden failure mode" -- a P2
  follow-up to surface a 503 or red status on /cron when the guard
  fires, deferred from this phase.

## Backwards compatibility

- Listener is purely additive. Pre-existing scheduler behavior
  unchanged.
- `_JOB_NAMES` extension is additive (extra rows appear; existing
  consumers indexing by name still work).
- Fail-open httpx.post: if the main backend is down or unreachable,
  the slack-bot scheduler keeps firing; only the heartbeat is lost.
- No new dependencies (httpx is already in `slack_bot/scheduler.py`
  imports).

## References

- Researcher: `handoff/current/phase-23.3.2-{external-research,internal-codebase-audit}.md`
- `backend/slack_bot/scheduler.py:81` (insert listener here)
- `backend/api/job_status_api.py:52-60` (extend _JOB_NAMES)
- APScheduler events: EVENT_JOB_EXECUTED / ERROR / MISSED
- Fowler heartbeat pattern; daily-devops Green-Dashboard anti-pattern
