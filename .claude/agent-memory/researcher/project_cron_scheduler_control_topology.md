---
name: cron-scheduler-control-topology
description: Backend APScheduler instances ARE reachable from the API layer via cron_dashboard_api's registry; in-process job ids + triple double-fire guard for the daily cycle (phase-49.2)
metadata:
  type: project
---

The backend's APScheduler instances are reachable from the API/router layer
WITHOUT new plumbing — a registry already exists. Researched phase-49.2.

**Why:** phase-49.2 needed operator cron-control endpoints (pause/resume/
trigger-now). The gating question was whether in-process control is even
possible from the API layer; it is.

**How to apply:** When designing any scheduler-control or job-introspection
feature, reuse this topology instead of re-tracing:

- `backend/api/cron_dashboard_api.py` owns `_RUNNING_SCHEDULERS` dict +
  `register_scheduler(name, scheduler)` + `get_registered_schedulers()`.
  `backend/main.py` lifespan registers TWO live `AsyncIOScheduler`s:
  `"main"` (the paper scheduler) and `"queue"` (ticket-queue 5s job). Same
  process / same event loop as every `/api/*` route. So a router can call
  `get_registered_schedulers()["main"].pause_job(id)` etc. in-process.
- Second handle for the paper job only: `paper_trading.py` module global
  `_scheduler` (set in `init_scheduler`), already used for `get_job`/
  `remove_job`/`next_run_time`.
- IN-PROCESS controllable job ids (allowlist): `paper_trading_daily` (main),
  `ticket_queue_process_batch` (queue). EVERYTHING ELSE in
  `cron_dashboard_api.py` `_SLACK_BOT_JOBS` (11 jobs) + `_LAUNCHD_JOBS` is
  CROSS-PROCESS / OS-level — a static manifest only; the backend cannot
  control them. Reject those ids with 404.
- APScheduler **3.11.2** (NOT 4.x). 3.x job-level `pause_job`/`resume_job`/
  `get_job`/`get_jobs`/`modify_job`/`reschedule_job`/`remove_job` all verified
  present. 4.0 is a redesign (Schedule.paused field) — pin to 3.x.

**Triple double-fire guard for the daily cycle** (so a manual trigger can't
double-execute trades): (1) `/run-now` (paper_trading.py:~1016) checks
`get_loop_status()["running"]` -> HTTP 409; (2) `run_daily_cycle`
(autonomous_loop.py:152) `_running` flag short-circuits; (3) file-based
`cycle_lock` fcntl flock (cycle_lock.py:117) is cross-process. Trigger-now
should REUSE `/run-now`, not `modify_job(next_run_time=now)` (racy + perturbs
schedule). APScheduler 3.x has NO first-class run-now (idiom = transient
date-job); Quartz's `triggerJob` is the cross-domain contrast.

Confirmation-gate + audit pattern to mirror: `kill_switch.py` audit JSONL +
phase-49.1 `backend/services/risk_overrides.py` + `RiskLimitRequest`/
`KillSwitchActionRequest` (`{confirmation, reason}`). See
[[project_research_gate_discipline]].
