# Experiment results -- phase-49.2: Operator cron-control endpoints

**Date:** 2026-05-29 | **Result: built + live-verified** | $0 LLM | backend restarted to load new routes.

## What was built
Operator pause/resume/trigger control for the BACKEND's in-process APScheduler jobs (P7 "cron enable+trigger"), confirmation-gated + audited, with trigger reusing /run-now's triple-guard so it can never double-fire a trading cycle.

## Files changed/added
1. **`backend/services/cron_control.py`** (NEW, ~140 lines) -- mirrors the kill_switch/risk_overrides audit pattern:
   - `CONTROLLABLE = {"paper_trading_daily":"main", "ticket_queue_process_batch":"queue"}` (the 2 backend-owned in-process jobs).
   - `pause/resume` resolve the scheduler via `cron_dashboard_api.get_registered_schedulers()` (lazy import -> no cycle) and call APScheduler 3.x `pause_job`/`resume_job`; `JobLookupError` + unknown/cross-process id -> `CronControlError` (-> 404).
   - `status(job_id)` (paused/next_run), `is_controllable`, `record_trigger` (audit), append-only JSONL at `handoff/cron_control_audit.jsonl`.
2. **`backend/api/cron_dashboard_api.py`** -- `CronControlRequest{confirmation, reason}` + 3 routes: `POST /api/jobs/{job_id}/pause` (PAUSE_JOB), `/resume` (RESUME_JOB), `/trigger` (TRIGGER_JOB). pause/resume invalidate `paper:*` cache. trigger: paper_trading_daily -> `await run_now()` (reuses the 409 running-check + _running + cycle_lock guard); ticket_queue -> 400 (pause/resume only this step). Also added an additive `controllable: bool` key to `_job_to_dict` so `GET /jobs/all` rows flag the 2 controllable jobs.

## Verification command output (masterplan immutable command)
```
$ python -c "import ast; ast.parse(open('backend/api/cron_dashboard_api.py').read())"   -> OK
$ python -c "import backend.api.cron_dashboard_api as c; paths=...; assert pause&resume&trigger present"
cron-control routes: ['/api/jobs/{job_id}/pause', '/api/jobs/{job_id}/resume', '/api/jobs/{job_id}/trigger']
$ test -f handoff/current/live_check_49.2.md   -> exists
```
Both modules ast.parse clean; cron_control.CONTROLLABLE correct (is_controllable: paper_trading_daily=True, morning_digest=False).

## Live verification (full evidence in live_check_49.2.md)
GET(scheduled,controllable) -> PAUSE(paused,next_run=null) -> GET(paused) -> RESUME(scheduled,next_run restored) -> GET(scheduled). Cross-process morning_digest -> HTTP 404. Wrong confirmation -> HTTP 400. ticket_queue trigger -> HTTP 400. Audit JSONL: pause+resume rows. **paper_trading_daily left RESUMED (next_run 2026-06-01T14:00) -- the money loop is intact.**

## Success criteria mapping (all 5 met)
1. pause/resume/trigger endpoints exist + confirmation-gated + audited to cron_control_audit.jsonl -- YES.
2. pause/resume act on the in-process registered scheduler (get_registered_schedulers), allowlisted to the 2 jobs; cross-process/unknown -> 404 -- YES (morning_digest 404).
3. GET /jobs/all reflects paused state after pause + scheduled after resume -- YES (live).
4. trigger for paper_trading_daily reuses /run-now's guard (409 when running), NOT modify_job -- YES (code: `await run_now()`; the running-check is run_now line 1 at paper_trading.py:1024-1026).
5. live curl round-trip + 404 captured in live_check_49.2.md; actions audited -- YES.

## Scope honesty
- Did NOT fire a real paper_trading_daily trigger (would incur LLM spend + real trades -- operator-gated). Criterion #4's guard is verified by code reuse of the already-validated /run-now path, not by firing a cycle.
- ticket_queue_process_batch trigger is intentionally OUT OF SCOPE (returns 400); pause/resume support it. Cross-process slack_bot/launchd jobs are intentionally NOT controllable (404); a future step can add a flag the slack_bot scheduler polls.
- pause/resume are reversible (preserve the job + trigger), distinct from /stop's remove_job. Test left the paper job RESUMED.
