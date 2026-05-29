# Research Brief — phase-49.2: Operator cron-control endpoints

Tier: moderate. Status: IN PROGRESS (incremental write).
Date: 2026-05-29

## Scope
Backend control endpoints (likely under the existing cron dashboard
router) to: list jobs with next_run_time + paused state; pause a job;
resume a job; trigger a job to run now — for the BACKEND's own
in-process APScheduler jobs. Consistent with the existing
confirmation-gated/audited control pattern (kill-switch/risk-limits).

---

## INTERNAL CODE TRACE (5 gating questions) — DEFINITIVE

### Q1 — Is the APScheduler instance reachable from the API/router layer? **YES — a registry already exists.**

DECISIVE: the cron router OWNS a scheduler registry. No new plumbing needed.
- `backend/api/cron_dashboard_api.py:49` `_RUNNING_SCHEDULERS: dict[str, Any] = {}`
  + `register_scheduler(name, scheduler)` (line 52) + `get_registered_schedulers()`
  (line 61). Comment (lines 42-48) states the design intent verbatim: lifespan
  populates this so the router can call `.get_jobs()` "without importing main.py
  (which would create a circular dependency at import time)."
- `backend/main.py:24-26` imports `register_scheduler as _register_cron_scheduler`.
- `backend/main.py:264` `_register_cron_scheduler("main", scheduler)` (the paper
  scheduler) and `main.py:317` `_register_cron_scheduler("queue", queue_scheduler)`
  (the ticket-queue scheduler).
- Both are live `AsyncIOScheduler` instances in the SAME process as every
  `/api/*` route (`main.py:260,295`).
=> The new control endpoints can call `get_registered_schedulers()["main"]` (or
   iterate to find the job by id) and invoke `scheduler.pause_job(id)` /
   `resume_job(id)` / `get_job(id)` / `modify_job(id, next_run_time=...)`
   IN-PROCESS. **In-process control IS possible.** No control-flag-polling
   workaround is needed for the backend (main + queue) schedulers.

SECOND independent handle (paper job only): `backend/api/paper_trading.py:37`
holds `_scheduler = None` module global, set in `init_scheduler` (line 1292
`_scheduler = scheduler`). paper_trading.py already calls
`_scheduler.get_job(_scheduler_job_id)` (line 90/109/137/143),
`_scheduler.remove_job(...)` (line 111), and reads `job.next_run_time` (line 144).
So the paper job has TWO reachable handles; the cron registry is the GENERAL one
that also covers the queue scheduler.

### Q2 — In-process backend jobs vs cross-process slack_bot jobs — CONFIRMED split.

**IN-PROCESS (controllable from the backend API — IN SCOPE):**
| Job id | Scheduler (registry key) | Registered at | Trigger |
|--------|--------------------------|---------------|---------|
| `paper_trading_daily` | `main` | `paper_trading.py:1302` (`_add_scheduler_job`, id=`_scheduler_job_id` line 38) | cron mon-fri @ `paper_trading_hour`:00 America/New_York |
| `ticket_queue_process_batch` | `queue` | `main.py:309` | interval 5s |

These two are the ONLY jobs on backend-owned schedulers. Both reachable via
`get_registered_schedulers()` AND introspected today by `GET /api/jobs/all`
(`cron_dashboard_api.py:407-410`, `source="main_apscheduler"`).

**CROSS-PROCESS (OUT OF SCOPE — cannot control in-process; note as follow-on):**
- slack_bot scheduler (`backend/slack_bot/scheduler.py`) runs in a SEPARATE
  process (`python -m backend.slack_bot.app`). The 11 jobs in
  `_SLACK_BOT_JOBS` (`cron_dashboard_api.py:79-102`: morning_digest,
  evening_digest, watchdog_health_check, prompt_leak_redteam,
  daily_price_refresh, weekly_fred_refresh, nightly_mda_retrain,
  hourly_signal_warmup, nightly_outcome_rebuild, weekly_data_integrity,
  cost_budget_watcher) are a STATIC MANIFEST — the backend cannot call
  `.get_jobs()` on that scheduler (comment lines 65-71). The phase-23.2.23
  brief already decided cross-process IPC is overkill for a single-dev local
  app. **Do NOT try to pause/trigger these from the backend API in this step.**
- launchd jobs (`_LAUNCHD_JOBS`, lines 108-121) are OS-level; `launchctl`
  controls them, not APScheduler. OUT OF SCOPE.

CONCLUSION: phase-49.2 controls exactly TWO job ids: `paper_trading_daily` and
`ticket_queue_process_batch`. Reject any other id with HTTP 404. Pausing the
5-second queue processor is low-value/risky (it drains the ticket queue) — a
defensible MVP scopes control to `paper_trading_daily` only and treats the
queue job as read-only-listed; design supports both, gate by an allowlist.

### Q3 — What does the existing cron surface already expose? **Read-only today.**

`backend/api/cron_dashboard_api.py` (`APIRouter(prefix="/api", tags=["cron"])`,
line 40):
- `GET /api/jobs/all` (line 402) — unified inventory: live APScheduler rows
  (`main_apscheduler`) + slack_bot manifest + launchd manifest. Per-row shape
  (`_job_to_dict`, lines 177-193): `{id, source, schedule, next_run, last_run,
  status, description}`. `status` = `"scheduled"` if `next_run_time` set else
  `"paused"` (line 191). **This is the listing shape to extend** — a paused
  APScheduler job already renders `status="paused"` here.
- `GET /api/logs/tail` (line 468) — allowlisted log tail. Read-only.
- `backend/api/job_status_api.py` — `GET /status` + `POST /heartbeat`
  (heartbeat registry merged into /jobs/all at line 422). Read-only re: jobs.

There is NO existing pause/resume/trigger endpoint on the cron router. The only
job-mutators today live on the paper-trading router: `/stop` (line 104,
`_scheduler.remove_job`) and `/start` (line 78, re-`_add_scheduler_job`) and
`/run-now` (line 1016).

NEW control endpoints SHOULD live in `cron_dashboard_api.py` (same router,
`prefix="/api"`, tags=["cron"]) because that is the generic cross-scheduler
surface that already owns the registry. Proposed paths under `/api/jobs/...`.

### Q4 — Idempotency / double-fire safety — **TRIPLE-GUARDED. Reuse /run-now's path.**

The daily cycle cannot double-execute trades even if triggered mid-run:
1. **API guard** (`/run-now`, paper_trading.py:1024-1026): reads
   `get_loop_status()["running"]`; raises HTTP 409 "A trading cycle is already
   in progress" if a cycle is live. (`get_loop_status` is
   `autonomous_loop.py:2132`, returns `_running`.)
2. **In-process re-entrancy flag** (`run_daily_cycle`, autonomous_loop.py:152-154):
   `if _running: return {"status":"skipped","reason":"already_running"}` — fires
   BEFORE any work, even if the scheduler calls `_scheduled_run` directly.
3. **File-based cross-process lock** (`run_daily_cycle`, autonomous_loop.py:167-171):
   `cycle_lock.acquire()` → `fcntl.flock(LOCK_EX|LOCK_NB)` on
   `handoff/.autonomous_loop.lock` (cycle_lock.py:101-154). Raises
   `CycleLockError` → `{"status":"skipped","reason":"already_running_file_lock"}`
   if any process (backend OR a stray run) holds it. Kernel auto-releases on
   process death; stale locks cleaned by TTL/dead-pid (cycle_lock.py:83-98).

`/run-now` itself (paper_trading.py:1016-1035) is the canonical manual-trigger:
guard → `get_api_cache().invalidate("paper:*")` → `asyncio.create_task(
_run_cycle_background(settings))` (fire-and-forget on the event loop) → returns
`{"status":"started"}`. dry_run=true short-circuits (no LLM/BQ/trades).

DESIGN DECISION: the cron "trigger-now" for `paper_trading_daily` MUST reuse the
`/run-now` code path (call the same `_run_cycle_background` + 409-guard), NOT
`scheduler.modify_job(next_run_time=now)`. Rationale: (a) `/run-now` already
carries the cache-invalidate + 409 guard; (b) `modify_job(next_run_time=now)`
mutates the cron schedule and only fires on the scheduler's next wakeup — racy
and it leaves the schedule perturbed. APScheduler's own recommended "run now"
is to add a transient `date`-trigger job, but here the in-process function call
via the existing guarded path is simpler and already battle-tested. For the
queue job (if triggered), a direct awaited call to its batch fn is fine (it has
its own try/except, main.py:300-303) but low value — recommend NOT exposing
trigger for the queue job.

### Q5 — APScheduler version + control API — **3.11.2; full control surface present.**

`pip show apscheduler` → **3.11.2** (3.x line, NOT 4.x). Verified live on the
project venv that `AsyncIOScheduler` exposes: `pause_job(id)`, `resume_job(id)`,
`get_job(id)`, `get_jobs()`, `modify_job(id, **changes)`, `reschedule_job(id,
trigger=...)`, `remove_job(id)`, `add_job(...)`. The `Job` instance also exposes
`.modify()`, `.reschedule()`, `.pause()`, `.resume()`, `.next_run_time`.

3.x semantics (relevant to the design):
- `scheduler.pause_job(id)` sets the job's `next_run_time = None` → it stops
  firing but stays registered. `GET /api/jobs/all` already renders such a job
  as `status="paused"` (cron_dashboard_api.py:191). Reversible via `resume_job`.
- `scheduler.resume_job(id)` recomputes `next_run_time` from the trigger.
- These are 3.x APIs. In APScheduler **4.x** the scheduler API was redesigned
  (data-store/async-first; `pause`/`resume` moved/renamed) — but we are on 3.x,
  so the 3.x calls above are correct. Pin docs to 3.x.

NOTE: the existing `/stop` uses `remove_job` (deletes the job entirely; `/start`
re-adds it). For phase-49.2 prefer `pause_job`/`resume_job` (preserves the job +
its trigger; cleaner "next_run_time" story) over remove/re-add, AND keep them
consistent so `/stop`+`/start` and pause/resume don't fight. Document that
`/stop` (remove) and cron-pause (pause_job) are different mechanisms on the same
job — recommend cron endpoints use pause/resume and leave /stop as-is, OR note
the interaction (a removed job can't be resumed; it must be re-added via /start).

### RECOMMENDED DESIGN (internal)

Add to `backend/api/cron_dashboard_api.py` (same router; it owns the registry):

- **`GET /api/jobs/all`** — EXTEND each `main_apscheduler` row with a
  `controllable: true` flag + `paused: bool` (already derivable from
  `next_run is None`). Cross-process rows get `controllable: false`. No breaking
  change to the existing shape (additive keys only).
- **`POST /api/jobs/{job_id}/pause`** — confirmation-gated. Look up the job
  across `get_registered_schedulers()`; if found and in the in-process allowlist
  (`{"paper_trading_daily","ticket_queue_process_batch"}`), call
  `scheduler.pause_job(job_id)`; else 404 (unknown/cross-process). Audit-append.
- **`POST /api/jobs/{job_id}/resume`** — confirmation-gated; `scheduler.resume_job(job_id)`.
- **`POST /api/jobs/{job_id}/trigger`** — confirmation-gated.
  - For `paper_trading_daily`: DELEGATE to the existing `/run-now` path —
    import `run_now` semantics: check `get_loop_status()["running"]` → 409;
    else `asyncio.create_task(_run_cycle_background(get_settings()))` +
    `get_api_cache().invalidate("paper:*")`. (Reuses the triple guard.)
  - For other jobs: out of MVP scope (or a direct guarded call).
- **`GET /api/jobs/{job_id}` (optional)** — single-job detail incl.
  `next_run_time`, `paused`, trigger string.

Confirmation gate + audit (mirror phase-49.1 `risk_overrides` exactly):
- Request model mirroring `RiskLimitRequest` (paper_trading.py:52-59):
  `{confirmation: str, reason: str}`. Require `confirmation == "PAUSE_JOB"` /
  `"RESUME_JOB"` / `"TRIGGER_JOB"` (matches the `KillSwitchActionRequest`
  pattern, paper_trading.py:47-49, 523/532/599).
- Audit: append-only JSONL like `kill_switch_audit.jsonl` /
  `risk_overrides_audit.jsonl`. Suggest `handoff/cron_control_audit.jsonl` with
  `{ts, job_id, action, actor, reason, prev_next_run, new_next_run}`.
- `get_api_cache().invalidate("paper:*")` on any state change (consistent with
  every paper mutator) so `/status` + `/jobs/all` reflect the new state.

Safety constraints baked in:
- Allowlist of controllable in-process job ids (reject everything else 404).
- Never expose pause/trigger for cross-process (slack_bot/launchd) jobs.
- trigger-now NEVER bypasses the `_running` + `cycle_lock` guards (reuse /run-now).
- Pausing `paper_trading_daily` does NOT disable the kill-switch breach checks
  (those run inside the cycle; a paused scheduler just means no NEW cycles —
  acceptable and is the operator's explicit intent).

---

## EXTERNAL RESEARCH

### Search-query variants run (3-variant discipline)
1. **Current-frontier (2026):** "operator manual trigger scheduled job idempotency
   double-fire safety production runbook 2026"; "FastAPI APScheduler runtime job
   management endpoint pause resume add remove 2025 2026".
2. **Last-2-year (2025):** "APScheduler max_instances coalesce misfire_grace_time
   prevent overlapping job runs 2025"; "scheduler job control API pause resume
   trigger next_run_time production service 2025"; "APScheduler 4.0 release
   scheduler API changes pause resume 2025".
3. **Year-less canonical:** "APScheduler pause resume job runtime reschedule
   trigger now 3.x documentation"; "cron job pause enable disable trigger endpoint
   admin API design pattern audit". (Surfaced the canonical APScheduler 3.x docs,
   Quartz API, Kubernetes suspend, Hasura — the founding prior-art.)

### Read in full (>=5 required — counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-29 | official docs (primary, version-matched 3.x) | WebFetch (full) | "When a job is paused, its next run time is cleared and no further run times will be calculated for it until the job is resumed." Methods: `pause_job`/`resume_job`/`reschedule_job`/`modify_job`/`get_jobs`/`remove_job`. "To run a job immediately, omit `trigger` argument when adding the job" — there is NO dedicated run-now method; transient date-job is the idiom. `max_instances` default 1 ("only one instance of each job is allowed to be run at the same time"); `coalesce` rolls missed runs into one; `misfire_grace_time` gates whether a missed run still fires. |
| trigger.dev/docs/idempotency | 2026-05-29 | official docs (job-runner vendor) | WebFetch (full) | Idempotency key: "the second request will not create a new task run. Instead, the original run's handle is returned." Scopes: run / attempt / global ("runs only once ever"). `idempotencyKeyTTL` (default 30 days). "When a run with an idempotency key fails, the key is automatically cleared." |
| kubernetes (nickjanetakis.com/.../suspending-kubernetes-cron-jobs) | 2026-05-29 | authoritative blog (ops pattern) | WebFetch (full) | `suspend: true` "won't get triggered, this is often more user friendly than temporarily deleting the entire cron job and adding it back later." Suspended job "remains registered... but ceases execution" — "maintaining clean audit trails... without the overhead of deletion and recreation." Maintenance-mode use case. |
| quartz-scheduler.org/api/2.3.0/.../Scheduler.html | 2026-05-29 | official docs (cross-domain: mature Java scheduler) | WebFetch (full) | `triggerJob` = "Trigger the identified JobDetail (execute it now)" — a first-class run-now. `pauseJob` = "Pause the JobDetail... by pausing all of its current Triggers." `resumeJob`: "If any of the Job's Triggers missed one or more fire-times, then the Trigger's misfire instruction will be applied." Confirms pause/resume/trigger-now is the canonical scheduler-control triad. |
| hasura.io/blog/introducing-scheduled-triggers... | 2026-05-29 | authoritative blog (API-driven cron) | WebFetch (full) | "Hasura also stores the full logs of every invocation for later retrieval." Delivery is "atleast-once guaranteed"; on restart, queued events "(re)triggered based on a customisable tolerance configuration" (default 6h) — analogous to APScheduler `misfire_grace_time`+`coalesce`. Audit-every-invocation is the norm. |
| ahaw021.medium.com/scheduled-jobs-with-fastapi-and-apscheduler | 2026-05-29 | practitioner blog (exact stack: FastAPI+APScheduler) | WebFetch (full) | `AsyncIOScheduler` "integrates with the existing event loop — it starts within the FastAPI/uvicorn context and operates on the same async runtime"; scheduler held as a **global variable** so "route handlers... add, delete, and list scheduled jobs." CRITICAL CAVEAT it flags: "Running multiple uvicorn worker processes would create separate scheduler instances per worker, potentially causing duplicate job executions." |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not full |
|-----|------|--------------|
| apscheduler.readthedocs.io/en/3.x/modules/job.html | official | Job-API reference; content captured via search + userguide full-read |
| apscheduler.readthedocs.io/en/master/api.html (4.0) | official | Recency-scan evidence (4.0 redesign); we are on 3.x so out of direct scope |
| github.com/agronholm/apscheduler issues/465 + discussions/592,613 | official (maintainer) | 4.0 progress + overlap-prevention Q&A; search snippet sufficient |
| cron-job.org/rest-api.html | vendor docs | PATCH `{job:{enabled:true}}` enable/disable pattern; snippet sufficient |
| developers.cloudflare.com/workers/.../cron-triggers | official | Stores 100 most-recent invocations (audit-retention norm); snippet |
| vercel.com/docs/cron-jobs/manage-cron-jobs | official | Managed-cron control surface; snippet |
| buildmvpfast.com/.../idempotent-ai-agent-retry-safe-patterns-2026 | blog (2026) | HTTP 403 on WebFetch; idempotency-key/preconditions/approvals captured via search snippet |
| ibm.com/docs/en/runbook-automation (triggers) | official | "manual or semi-automated... operator must complete parameter values"; snippet |
| rajansahu713.medium.com/...fastapi-apscheduler (Mar 2024) | practitioner | Truncated on WebFetch; intro only |
| github.com/amisadmin/fastapi-scheduler + pypi fastapi_scheduler | OSS | Confirms FastAPI+APScheduler dashboard/control wrapper exists as a category |
| sentry.io/answers/schedule-tasks-with-fastapi | vendor | Redundant with the two FastAPI full-reads |

URLs collected total: 30+ across the 6 search passes.

### Recency scan (last 2 years, 2024-2026) — PERFORMED
Findings that COMPLEMENT / clarify the canonical sources:
- **APScheduler 4.0 (in progress, 2024-2026)** — a ground-up async-first redesign;
  `Schedule` objects carry a `paused` field and the scheduler API is restructured
  vs 3.x. RELEVANCE: confirms we must pin the design to **3.x semantics**
  (`pause_job`/`resume_job` job-level calls), since the project runs 3.11.2 and the
  4.x API differs. No reason to adopt 4.x for this step.
  (Source: apscheduler.readthedocs.io/en/master/versionhistory.html + issue #465.)
- **Trigger.dev idempotency docs (current, 2025-2026)** — modern managed job-runner
  formalizes the idempotency-key pattern (run/attempt/global scope + TTL) that maps
  directly onto our `cycle_lock` + `_running`-flag double-fire guard. Complements
  the older "add preconditions + idempotency key" runbook guidance.
- **FastAPI+APScheduler practitioner pattern (Mar 2024, Hawke)** — current confirmation
  that `AsyncIOScheduler`-in-FastAPI-process + global-scheduler-shared-with-routes is
  THE idiom, AND that the multi-uvicorn-worker duplicate-execution failure mode is
  real. Directly validates the internal Q1 finding (registry-shared scheduler) and
  the existing `cycle_lock` cross-process guard.
No 2024-2026 source CONTRADICTS the design. Consensus is unchanged and strengthened:
expose pause/resume/trigger-now via API on the in-process scheduler, audit every
invocation, and guard trigger-now against double-fire. No finding supersedes the
existing `cycle_lock`/`_running` re-entrancy guards already in the repo.

### Consensus vs debate
CONSENSUS (cross-domain — APScheduler + Quartz + Kubernetes + Hasura + Trigger.dev):
(1) the standard scheduler-control triad is **pause / resume / trigger-now**, plus a
list-with-next-run read; (2) **pause/suspend is preferred over delete** because it
keeps the job registered, preserves the trigger, and is reversible (K8s "more user
friendly than... deleting", APScheduler "clears next_run_time", Quartz "pauses all
its Triggers"); (3) **every invocation/control action should be audit-logged** (Hasura
"full logs of every invocation", Cloudflare 100-most-recent); (4) **manual trigger-now
must be idempotent / double-fire-safe** (Trigger.dev idempotency key; runbook
"add preconditions... idempotency"); (5) **misfire/coalesce/max_instances** are the
knobs that prevent overlap on the scheduled path (APScheduler default max_instances=1).
DEBATE: whether "run now" is a first-class scheduler method. Quartz has `triggerJob`
(yes); APScheduler 3.x does NOT (idiom = add a transient trigger-less/date job OR call
the target function directly). For pyfinagent the cleaner answer sidesteps the debate:
the paper job ALREADY has a guarded run-now path (`/run-now`) — reuse it rather than
poking the scheduler.

### Pitfalls (from literature)
- **Multi-worker duplicate execution (Hawke; project phase-38.13.1)** — if the backend
  ever runs >1 uvicorn worker, each worker starts its own `AsyncIOScheduler` →
  duplicate fires. MITIGATION already present: the `cycle_lock` fcntl flock
  (cycle_lock.py) is cross-process and prevents double-execution of the daily cycle
  even under multi-worker. The control endpoints, however, would only pause/resume the
  scheduler in the worker that served the request — so on a true multi-worker deploy
  pause is per-worker. NOTE this as a known limitation; the local single-worker Mac
  deployment ([[project_local_only_deployment]]) is unaffected today.
- **trigger-now double-fire (Trigger.dev; runbook)** — a manual trigger while a run is
  in-flight must not double-execute. MITIGATION: reuse `/run-now`'s 409 guard
  (`get_loop_status()["running"]`) + `run_daily_cycle`'s `_running` flag +
  `cycle_lock`. Do NOT `modify_job(next_run_time=now)` (racy, perturbs schedule).
- **Delete-instead-of-pause loses the trigger (K8s)** — the existing `/stop` uses
  `remove_job` (deletes the job; only `/start` re-adds it). A `resume` on a removed job
  fails. MITIGATION: cron control endpoints use `pause_job`/`resume_job` (preserve the
  job); document the `/stop` vs cron-pause distinction so they don't fight.
- **Un-audited control action (Hasura, Cloudflare)** — every pause/resume/trigger must
  append a timestamped audit row. MITIGATION: append-only JSONL mirroring
  `kill_switch_audit.jsonl` / `risk_overrides_audit.jsonl`.
- **Controlling cross-process jobs you can't reach** — attempting to pause a slack_bot/
  launchd job from the backend API silently no-ops or errors. MITIGATION: allowlist of
  in-process job ids; reject all others with HTTP 404.

### Application to pyfinagent (external -> internal anchors)
1. Pause/resume/trigger-now triad (APScheduler userguide; Quartz triggerJob; K8s
   suspend) -> NEW endpoints on `backend/api/cron_dashboard_api.py` (router already
   owns `get_registered_schedulers()`, line 61): `POST /api/jobs/{id}/pause` ->
   `scheduler.pause_job(id)`, `POST .../resume` -> `resume_job(id)`,
   `POST .../trigger` -> reuse `/run-now`. List via the existing `GET /api/jobs/all`
   (line 402), extended with `paused`/`controllable` keys.
2. Prefer pause over delete (K8s "more user friendly"; APScheduler "clears
   next_run_time") -> use `pause_job`/`resume_job`, NOT the `remove_job` mechanism the
   current `/stop` (paper_trading.py:111) uses. `GET /api/jobs/all` already renders a
   paused job as `status="paused"` (cron_dashboard_api.py:191) — no UI change needed.
3. Trigger-now must be double-fire-safe (Trigger.dev idempotency; runbook
   preconditions) -> reuse `/run-now` (paper_trading.py:1016) which already chains the
   409 guard + `_running` flag (autonomous_loop.py:152) + `cycle_lock`
   (autonomous_loop.py:167 / cycle_lock.py:117). Do NOT reschedule next_run_time.
4. Audit every control action (Hasura full-logs; Cloudflare retention) -> append-only
   JSONL `handoff/cron_control_audit.jsonl` mirroring
   `backend/services/kill_switch.py` audit + phase-49.1
   `backend/services/risk_overrides.py`. Invalidate `paper:*` cache on write
   (consistent with every paper mutator, e.g. paper_trading.py:525).
5. Confirmation gate on control actions (Quartz/Hasura governance; Knight Capital
   four-eyes from phase-49.1) -> request model mirroring `RiskLimitRequest`
   (paper_trading.py:52-59) / `KillSwitchActionRequest` (paper_trading.py:47-49):
   `{confirmation, reason}`, require `confirmation in {"PAUSE_JOB","RESUME_JOB",
   "TRIGGER_JOB"}` (matches paper_trading.py:523/532/599).
6. Pin to APScheduler 3.x API (recency scan: 4.0 redesign) -> 3.11.2 confirmed
   installed; `pause_job`/`resume_job`/`get_job`/`modify_job` verified present on the
   project venv. Do not adopt 4.x.
7. Allowlist in-process job ids only (cross-process pitfall) -> controllable =
   `{"paper_trading_daily","ticket_queue_process_batch"}`; reject slack_bot/launchd
   ids (manifest-only, cron_dashboard_api.py:79-121) with HTTP 404.

### 2-3 external-source-backed safety best practices (deliverable item c)
1. **Pause via flag, not delete; keep it reversible.** Kubernetes `suspend:true`
   ("more user friendly than... deleting the entire cron job"); APScheduler "clears
   next run time... until the job is resumed." -> use `pause_job`/`resume_job`,
   never remove/re-add for the cron-control surface.
2. **Make trigger-now idempotent against an in-flight run.** Trigger.dev: a repeat
   trigger with the same key "will not create a new task run." -> the cron trigger
   reuses `/run-now`'s `_running`/`cycle_lock` guards rather than firing blind.
3. **Audit every control action + every invocation.** Hasura "stores the full logs of
   every invocation for later retrieval"; Cloudflare retains the 100 most recent. ->
   append-only JSONL audit, mirroring the kill-switch/risk-overrides pattern, on every
   pause/resume/trigger.

---

## JSON ENVELOPE

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
