# Internal Codebase Audit — phase-23.1.20
# Pause/Resume Timeout Hardening

Audit date: 2026-04-29
Auditor: Researcher agent (merged Explore)

---

## 1. `bq.get_paper_portfolio` — timeout audit

**File:** `backend/db/bigquery_client.py:481-490`

```python
def get_paper_portfolio(self, portfolio_id: str = "default") -> Optional[dict]:
    query = f"""
        SELECT * FROM `{self._pt_table("paper_portfolio")}`
        WHERE portfolio_id = @pid LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("pid", "STRING", portfolio_id),
    ])
    rows = list(self.client.query(query, job_config=job_config).result())
    return dict(rows[0]) if rows else None
```

**Finding:** NO timeout is set. `self.client.query(...).result()` blocks the calling thread indefinitely until the BQ job completes or the BQ SDK's own internal network timeout fires (default: not documented to be 30s; in practice can be 60-120s or longer depending on BQ slot availability and network conditions).

**CLAUDE.md rule:** "BQ timeout: 30s on all fallback queries." This rule is NOT applied here. There is no `timeout` keyword argument passed to `.result()`. The google-cloud-bigquery SDK's `QueryJob.result(timeout=N)` accepts a `timeout` parameter (seconds) that will raise `concurrent.futures.TimeoutError` if the job does not complete in time.

**Confirmed blocking path in resume_trading:**
`backend/api/paper_trading.py:382`
```python
portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
```
`asyncio.to_thread` offloads to the default ThreadPoolExecutor. The thread itself has no timeout — it will block until `get_paper_portfolio` returns. If BQ is slow (e.g., during FD-exhaustion like phase-23.1.19), this thread cannot be cancelled from asyncio. `asyncio.wait_for` wrapped around `asyncio.to_thread(...)` WILL raise `asyncio.TimeoutError` on the asyncio side and cancel the coroutine, but the underlying thread continues running until BQ eventually returns. This is the correct approach for the hot path: the HTTP response is released to the client with 503, even though the BQ thread lingers in the background.

---

## 2. Pause endpoint — blocking call audit

**File:** `backend/api/paper_trading.py:365-371`

```python
@router.post("/pause")
async def pause_trading(req: KillSwitchActionRequest):
    if req.confirmation != "PAUSE":
        raise HTTPException(400, "Confirmation must equal PAUSE")
    get_api_cache().invalidate("paper:*")
    state = _get_ks_state().pause(trigger="manual")
    return {"status": "paused", "state": state}
```

**`get_api_cache().invalidate("paper:*")`** (`backend/services/api_cache.py:59-73`):
Pure in-memory operation. Acquires a `threading.Lock`, iterates the in-memory dict, deletes matching keys. No I/O. Completes in microseconds. Safe.

**`_get_ks_state().pause(trigger="manual")`** (`backend/services/kill_switch.py:104-109`):
- Acquires `threading.Lock`
- Sets `self._paused = True`
- Calls `self._append_audit("pause", ...)` which opens `_AUDIT_PATH` (a local file) for append and writes one JSON line
- File write is already wrapped in `try/except` — if it fails, only a `logger.warning` is emitted; the lock is released and the in-memory state flip succeeds regardless

**Finding:** The pause endpoint has NO blocking I/O that could cause a 30s hang. The only I/O is a local file append to the audit log, which is fail-soft. Under normal conditions this completes in <1ms. Under extreme FD exhaustion (like phase-23.1.19), the file open could fail, but the exception is caught and swallowed.

**Verdict on pause:** Does NOT need a timeout wrapper. The BQ call is only in the resume endpoint and the kill-switch status GET endpoint, not pause.

---

## 3. Other mutation endpoints under /api/paper-trading/* — timeout audit

### `/flatten-all` — `backend/api/paper_trading.py:401-412`

```python
@router.post("/flatten-all")
async def flatten_all(req: KillSwitchActionRequest):
    ...
    result = await asyncio.to_thread(trader.flatten_all, "manual_flatten")
    _get_ks_state().pause(trigger="manual_flatten", details=result)
    return {"status": "flattened_and_paused", "result": result}
```

**Finding:** `trader.flatten_all(...)` is a potentially long-running sync operation (issues BQ DML writes per position, iterates positions list, calls `_run_dml_with_retry` which has up to 3 retries with 5s/10s/20s sleeps). No timeout. Under BQ slowness, this could block for 35-120s. This is a higher-risk endpoint than resume — it touches more BQ tables. However, it is a destructive/rare action and less likely to be a user-facing hang in normal operation. Noted as a secondary fix target.

### `/stop` — `backend/api/paper_trading.py:90-99`

```python
@router.post("/stop")
async def stop_paper_trading():
    get_api_cache().invalidate("paper:*")
    if _scheduler:
        job = _scheduler.get_job(_scheduler_job_id)
        if job:
            _scheduler.remove_job(_scheduler_job_id)
            return {"status": "stopped", "message": "Scheduler paused"}
    return {"status": "not_running"}
```

**Finding:** Pure in-memory operations. APScheduler's `get_job`/`remove_job` are thread-safe in-process calls. No external I/O. Safe. No timeout needed.

### `/run-now` — not found in paper_trading.py

Searched for `run.now` and `run_now` — no endpoint with this name exists in `backend/api/paper_trading.py`. The scheduler triggers via APScheduler cron; there is no manual "run now" route in the current codebase. Not applicable.

### `/toggle-scheduler` — not found

Searched for `toggle` — no endpoint with this name. The relevant scheduler control is `/stop` (remove job) and `/start` (add job). Not applicable.

### `/status` GET — `backend/api/paper_trading.py:102-`

Has `await asyncio.to_thread(bq.get_paper_portfolio, "default")` at line 115 — same unguarded BQ call as resume. This is a polling endpoint and could hang under BQ pressure, but it's a read-only GET and is cached. Lower priority.

### Kill-switch status GET — `backend/api/paper_trading.py:333-362`

Has `await asyncio.to_thread(bq.get_paper_portfolio, "default")` at line 343 — same issue. No timeout.

---

## 4. Existing tests for pause/resume

Searched `tests/` directory for pause/resume test coverage:

```
tests/verify_phase_23_1_13.py  — manage tab toggle, no pause/resume POST
tests/verify_phase_23_1_17.py  — no pause/resume
tests/verify_phase_23_1_19.py  — FD exhaustion fix; no pause/resume timeout test
```

**Finding:** There are NO existing tests that:
- Exercise POST /pause or POST /resume directly
- Mock `bq.get_paper_portfolio` with artificial delay
- Assert timeout behavior (503 within N seconds)

The test file `tests/api/test_pause_resume_timeout.py` does not exist. This is the gap Fix C addresses.

---

## 5. Summary table

| Endpoint | BQ call? | Timeout guarded? | Risk | Fix priority |
|---|---|---|---|---|
| POST /pause | No | N/A | None | None needed |
| POST /resume | Yes (get_paper_portfolio) | No | HIGH (30s hang) | A — immediate |
| POST /flatten-all | Yes (flatten_all DML) | No | MEDIUM (35-120s) | Phase 2 |
| POST /stop | No | N/A | None | None needed |
| GET /status | Yes (get_paper_portfolio) | No | LOW (cached) | Phase 2 |
| GET /kill-switch | Yes (get_paper_portfolio) | No | LOW (status read) | Phase 2 |

---

## 6. Files inspected

| File | Lines read | Role |
|---|---|---|
| `backend/api/paper_trading.py` | 87-430 | Pause/resume/flatten/stop/status handlers |
| `backend/db/bigquery_client.py` | 475-528 | get_paper_portfolio, upsert_paper_portfolio |
| `backend/services/kill_switch.py` | 50-130 | KillSwitchState.pause/resume/_append_audit |
| `backend/services/api_cache.py` | 55-84 | invalidate() implementation |
| `tests/` (glob) | all filenames | Existence check for pause/resume tests |
