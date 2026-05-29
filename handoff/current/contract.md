# Contract -- phase-49.2: Operator cron-control endpoints

**Step id:** 49.2 | **Priority:** P2 (P7 operator control surface -- "cron enable+trigger") | **depends_on:** 49.1
**Date:** 2026-05-29 | **harness_required:** true | **$0 LLM** (backend; restart once to load new routes)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher gate: **7 sources read in full, recency scan, 30 URLs, 6 internal files, gate_passed=true**). Decisive:
- **Q1 -- scheduler reachable in-process (no new plumbing):** `cron_dashboard_api.py:49` already has `_RUNNING_SCHEDULERS` + `register_scheduler()` + `get_registered_schedulers()`. `main.py:264` registers `"main"` (paper `AsyncIOScheduler`), `main.py:317` registers `"queue"` (ticket-queue). Both run in the SAME process/event loop as every `/api` route -> the router can call `get_registered_schedulers()["main"].pause_job(id)/resume_job(id)/get_job(id)` directly.
- **Q2 -- 2 controllable jobs:** `paper_trading_daily` (on `"main"`, registered `paper_trading.py:1302`, id `paper_trading.py:38`) + `ticket_queue_process_batch` (on `"queue"`, `main.py:309`). The 11 `_SLACK_BOT_JOBS` + 6 `_LAUNCHD_JOBS` (`cron_dashboard_api.py:79-121`) are a static cross-process manifest -> reject with 404.
- **Q3 -- existing surface read-only:** `GET /api/jobs/all` (`cron_dashboard_api.py:402`) lists `{id, source, schedule, next_run, last_run, status, description}` and already renders paused jobs as `status="paused"` (line 191). New control endpoints belong in `cron_dashboard_api.py` (it owns the registry).
- **Q4 -- double-fire TRIPLE-guarded:** `/run-now` (`paper_trading.py:1024`) running-check -> 409; `run_daily_cycle` (`autonomous_loop.py:152`) `_running` short-circuit; cross-process `cycle_lock` fcntl flock (`cycle_lock.py:117`). Trigger-now MUST reuse the `/run-now` path, NOT `modify_job(next_run_time=now)` (racy).
- **Q5 -- APScheduler 3.11.2** (not 4.x): `pause_job`/`resume_job`/`get_job`/`get_jobs` present. NOTE: `/stop` uses `remove_job` (deletes the job); cron-control uses `pause_job`/`resume_job` (preserves the job + trigger) -- distinct, must not fight.
- **External:** APScheduler 3.x docs; Trigger.dev (repeat trigger != new run -> reuse the in-flight guard); K8s suspend-not-delete (reversible); Hasura (audit every invocation).

## Hypothesis
Adding `POST /api/jobs/{id}/pause|resume|trigger` to `cron_dashboard_api.py` -- acting on the already-registered in-process schedulers, allowlisted to the 2 backend-owned jobs, confirmation-gated + audited, with `trigger` delegating to `/run-now`'s triple-guard -- delivers the P7 "cron enable+trigger" control with no double-fire risk and no new scheduler plumbing.

## Success criteria (IMMUTABLE -- copied verbatim from masterplan step 49.2)
1. POST /api/jobs/{id}/pause, /api/jobs/{id}/resume, /api/jobs/{id}/trigger exist on the cron dashboard router; each is confirmation-gated (mirroring the kill-switch/risk-limits pattern) and appends an audit row to handoff/cron_control_audit.jsonl
2. pause/resume act on the IN-PROCESS registered scheduler via get_registered_schedulers() (NOT a newly created scheduler); they are allowlisted to the 2 backend-owned jobs (paper_trading_daily, ticket_queue_process_batch); unknown or cross-process (slack_bot/launchd) job ids are rejected with HTTP 404
3. GET /api/jobs/all reflects a paused job's state (status='paused' and/or a paused/controllable flag) after a pause call, and back to running after resume
4. trigger for paper_trading_daily reuses the /run-now guarded path (the _running / cycle_lock guard) and therefore returns HTTP 409 (NOT a double-run) when a cycle is already running; it does NOT use modify_job(next_run_time=now)
5. a LIVE curl round-trip is captured verbatim in live_check_49.2.md: pause paper_trading_daily -> GET /api/jobs/all shows it paused -> resume -> GET shows it running again, plus a 404 on a cross-process job id; every action appears in handoff/cron_control_audit.jsonl

**live_check:** REQUIRED -- live curl pause/resume round-trip + a 404 on a cross-process id, in live_check_49.2.md + the audit rows.

## Plan steps
1. **`backend/services/cron_control.py`** (NEW, small): `_AUDIT_PATH = handoff/cron_control_audit.jsonl`; `CONTROLLABLE = {"paper_trading_daily": "main", "ticket_queue_process_batch": "queue"}`; `_append_audit(action, job_id, **)`; helpers `pause(job_id)`, `resume(job_id)`, `is_paused(job_id)` that resolve the scheduler via `cron_dashboard_api.get_registered_schedulers()` and call `pause_job`/`resume_job` (guard: unknown id -> raise -> 404; APScheduler `JobLookupError` -> 404). Mirror the audit pattern of risk_overrides.py/kill_switch.py.
2. **`backend/api/cron_dashboard_api.py`**: add `CronControlRequest(BaseModel){confirmation, reason}` + 3 routes. pause/resume call cron_control + invalidate `paper:*` cache. trigger: for `paper_trading_daily` reuse the `/run-now` guarded path (import + call the same guarded helper, or replicate its `get_loop_status()["running"]` -> 409 check then `asyncio.create_task(_run_cycle_background(...))`); for `ticket_queue_process_batch` -> 400 "trigger not supported (pause/resume only)" in this step (note as follow-on) OR a guarded job.func call if trivial. Extend GET /jobs/all rows with `controllable: bool` (additive, non-breaking).
3. **Verify**: ast.parse; route registration; restart backend; LIVE curl pause -> GET(paused) -> resume -> GET(running); 404 on a `_SLACK_BOT_JOBS` id; capture into live_check_49.2.md; confirm cron_control_audit.jsonl rows. (Resume the paper job at the end so the daily schedule is left intact.)
4. **EVALUATE**: fresh qa (no self-eval). Then harness_log.md (LAST), then flip masterplan 49.2 -> done.

## Safety notes
- pause/resume are reversible (preserve the job + trigger), unlike /stop's remove_job. The test MUST resume paper_trading_daily at the end so the daily cycle keeps firing (don't leave the money loop paused).
- trigger reuses the existing triple-guard -> cannot double-fire trades. No new trade-execution path.
- Cross-process jobs (slack_bot/launchd) are explicitly NOT controllable here (404) -- a future step can add a flag the slack_bot scheduler polls.

## References
- handoff/current/research_brief.md (gate-passing brief)
- backend/api/cron_dashboard_api.py:49 (_RUNNING_SCHEDULERS), :402 (GET /jobs/all), :191 (paused status), :79-121 (cross-process manifest)
- backend/main.py:264,317 (register_scheduler main/queue)
- backend/api/paper_trading.py:38 (job id), :947-1026 (/run-now guard), :1302 (job registration)
- backend/services/autonomous_loop.py:152 (_running), backend/services/cycle_lock.py:117 (flock)
- backend/services/kill_switch.py + risk_overrides.py (audit-JSONL pattern)
- APScheduler 3.x docs; Trigger.dev idempotency; K8s suspend; Hasura audit
