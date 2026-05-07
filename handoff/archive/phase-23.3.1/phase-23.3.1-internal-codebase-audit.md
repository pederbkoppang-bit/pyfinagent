# Phase-23.3.1 Internal Codebase Audit
## Topic: APScheduler ticket-queue job — id/name labelling

### Files inspected

| File | Lines read | Role |
|------|-----------|------|
| `backend/main.py` | 191-224 | Queue scheduler init + add_job call |
| `backend/api/paper_trading.py` | 911-923 | Canonical add_job shape (_add_scheduler_job) |
| `backend/api/cron_dashboard_api.py` | 1-80, 128-177 | _job_to_dict, description field logic |

---

### Finding 1 — The bare add_job call (main.py:217)

```python
# backend/main.py:217
processor_job = queue_scheduler.add_job(process_batch, 'interval', seconds=5)
```

No `id=` and no `name=` are passed. APScheduler 3.11.2 (installed version) generates a UUID4 string as the job's `id` and sets `job.name` to the callable's `__qualname__`. Because `process_batch` is defined inside the `lifespan` async context manager, its `__qualname__` is `lifespan.<locals>.process_batch`. That is the value that appears in the `/cron` Jobs tab as "description".

The auto-generated `id` is a hex UUID, matching the live-observed value `2db2dd276ba94305a9aec11a5bb58f6c`. This UUID is ephemeral: it is regenerated on every backend restart.

---

### Finding 2 — No code references the UUID id elsewhere

Grep result for `2db2dd276ba94305a9aec11a5bb58f6c` across the entire repo: **zero hits**. No code path attempts to look up or cancel this job by the auto-generated id. The `processor_job` variable is only used in the shutdown path:

```python
# backend/main.py (shutdown block, approximate)
if 'queue_scheduler' in locals():
    queue_scheduler.shutdown(wait=False)
```

The shutdown operates on the scheduler object, not the job id. Therefore renaming the id is safe: no downstream code will break.

---

### Finding 3 — Canonical shape from paper_trading.py:911-923

```python
# backend/api/paper_trading.py:911-923
_scheduler_job_id = "paper_trading_daily"   # module-level constant

def _add_scheduler_job(settings):
    if not _scheduler:
        return
    _scheduler.add_job(
        _scheduled_run,
        "cron",
        hour=settings.paper_trading_hour,
        minute=0,
        day_of_week="mon-fri",
        timezone=ZoneInfo("America/New_York"),
        id=_scheduler_job_id,        # <-- explicit string id
        replace_existing=True,       # <-- restart-safe
    )
```

Observations:
- `id` is a module-level constant string, not an inline literal.
- `replace_existing=True` is always passed. This is the mandatory pattern for any job that might survive a restart: without it, re-registration on restart raises `ConflictingIdError` when a persistent jobstore is in use (or silently creates a duplicate with in-memory stores in some versions).
- `name=` is NOT passed here — the `paper_trading_daily` job shows its `id` as description in the dashboard (because `_job_to_dict` falls back to `job.id` when `job.name` is None/empty). Adding `name=` to both jobs would make the description column consistently human-readable.

---

### Finding 4 — _job_to_dict description logic (cron_dashboard_api.py:143)

```python
# backend/api/cron_dashboard_api.py:143
"description": getattr(job, "name", None) or getattr(job, "id", "?"),
```

Priority: `job.name` first, then `job.id` as fallback. When no `name=` is supplied to `add_job`, APScheduler sets `job.name` to the callable's `__qualname__`. So:

- `paper_trading_daily` job: `job.name` = `_scheduled_run` (module-level function, clean qualname) — but it still falls through to `job.id = "paper_trading_daily"` only if `job.name` evaluates falsy. Actually `_scheduled_run` is truthy, so description currently shows `_scheduled_run`, not `paper_trading_daily`. **Both jobs benefit from an explicit `name=`.**
- `process_batch` job: `job.name` = `lifespan.<locals>.process_batch` (closure qualname, noisy), `job.id` = UUID.

**The `or` short-circuit means: set an explicit `name=` on both jobs and the description column renders cleanly regardless of what `id` is.**

---

### Finding 5 — replace_existing semantics for an in-memory scheduler

`queue_scheduler` is an `AsyncIOScheduler` with no persistent jobstore configured (default = `MemoryJobStore`). On process restart the in-memory store is blank, so `replace_existing` is moot for preventing `ConflictingIdError`. However:

1. **Consistency/convention**: Passing `replace_existing=True` aligns with the established project pattern and is safe to add. It is a no-op cost for an in-memory store.
2. **Future-proofing**: If the store is ever migrated to SQLAlchemy or Redis, the flag is already present.
3. **Operational risk if omitted**: If a future refactor moves the job registration out of the lifespan try-block into a path that can be called multiple times (e.g., a health-check triggered re-init), a missing `replace_existing=True` with a string `id` would surface a `ConflictingIdError`.

---

### Recommendation block

Exact kwargs to add at `backend/main.py:217`:

```python
processor_job = queue_scheduler.add_job(
    process_batch,
    'interval',
    seconds=5,
    id="ticket_queue_process_batch",
    name="Ticket queue batch processor",
    replace_existing=True,
)
```

Sibling concern — paper_trading_daily also lacks `name=`:
```python
# backend/api/paper_trading.py:914 -- inside _add_scheduler_job
_scheduler.add_job(
    _scheduled_run,
    "cron",
    hour=settings.paper_trading_hour,
    minute=0,
    day_of_week="mon-fri",
    timezone=ZoneInfo("America/New_York"),
    id=_scheduler_job_id,
    name="Paper trading daily run",   # <-- add this
    replace_existing=True,
)
```

This makes the `/cron` Jobs tab description column show human-readable labels for both main-process jobs, matching the static manifest style used for slack_bot jobs (cron_dashboard_api.py:64-89).

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/main.py` | 197-223 | Queue scheduler init; bare add_job at line 217 | NEEDS id/name/replace_existing |
| `backend/api/paper_trading.py` | 911-923 | Canonical add_job shape; has id + replace_existing | NEEDS name= |
| `backend/api/cron_dashboard_api.py` | 128-144 | _job_to_dict; description = job.name or job.id | No change needed |
| `backend/api/cron_dashboard_api.py` | 38-51 | Scheduler registry (_RUNNING_SCHEDULERS) | No change needed |

No code references the auto-generated UUID `2db2dd276ba94305a9aec11a5bb58f6c` anywhere. Safe to add explicit id.
