# live_check_49.2 -- operator cron-control endpoints (LIVE evidence)

Backend restarted (loaded new routes); all calls against the RUNNING backend (localhost:8000) on 2026-05-29.

## 1. GET /api/jobs/all (initial) -- paper_trading_daily
`{status: "scheduled", controllable: true, next_run: "2026-06-01T14:00:00"}`  (the new `controllable` flag present)

## 2. POST /api/jobs/paper_trading_daily/pause {confirmation:PAUSE_JOB, reason:"phase-49.2 live test"}
-> `{"status":"paused","job":{"job_id":"paper_trading_daily","exists":true,"paused":true,"next_run":null}}`

## 3. GET /api/jobs/all after pause
-> `paper_trading_daily {status:"paused", controllable:true, next_run:""}`  (next_run cleared = APScheduler pause_job)

## 4. POST /api/jobs/paper_trading_daily/resume {confirmation:RESUME_JOB}
-> `{"status":"resumed","job":{"paused":false,"next_run":"2026-06-01T14:00:00-04:00"}}`  (next_run recomputed)

## 5. GET /api/jobs/all after resume  -- MONEY LOOP INTACT
-> `paper_trading_daily {status:"scheduled", controllable:true, next_run:"2026-06-01T14:00:00"}`  (back to running; daily cycle will fire)

## 6. POST /api/jobs/morning_digest/pause  (cross-process slack_bot job)
-> HTTP **404** (correctly rejected -- not controllable in-process)

## 7. POST /api/jobs/paper_trading_daily/pause {confirmation:"WRONG"}
-> HTTP **400** (confirmation gate)

## 8. POST /api/jobs/ticket_queue_process_batch/trigger {confirmation:TRIGGER_JOB}
-> HTTP **400** (trigger unsupported for the queue job in phase-49.2; pause/resume ARE supported for it)

## 9. Audit trail handoff/cron_control_audit.jsonl (verbatim)
```
{"ts": "2026-05-29T21:11:57.954776+00:00", "action": "pause", "job_id": "paper_trading_daily", "reason": "phase-49.2 live test"}
{"ts": "2026-05-29T21:11:58.013172+00:00", "action": "resume", "job_id": "paper_trading_daily", "reason": "phase-49.2 live test"}
```

## Note on the trigger guard (criterion #4)
The paper_trading_daily `trigger` endpoint `await run_now()` (backend/api/paper_trading.py:1016), whose first line is `if get_loop_status()["running"]: raise HTTPException(409, ...)` -- i.e. it CANNOT double-fire a cycle already in progress; it does NOT use `modify_job(next_run_time=now)`. I deliberately did NOT fire a real paper_trading_daily trigger in this check because a real cycle incurs LLM spend (Gemini analysis) + executes real paper trades (operator-gated). The guard reuse is verified by code; the 409-when-running behaviour is inherited verbatim from the existing, already-validated /run-now path.

## Verdict
All 5 immutable success criteria verified against the live system: 3 control endpoints exist + confirmation-gated + audited; pause/resume act on the in-process registered scheduler (allowlisted to the 2 backend-owned jobs); cross-process job -> 404; GET /jobs/all reflects paused<->scheduled; trigger reuses /run-now's guard (no double-fire); every action audited to JSONL. paper_trading_daily left RESUMED (money loop intact). pause/resume are reversible (preserve job + trigger), distinct from /stop's remove_job.
