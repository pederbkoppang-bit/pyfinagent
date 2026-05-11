---
step: phase-23.5.2.5
title: Bridge slack-bot heartbeat registry into /api/jobs/all + next_run push
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); slack=[j for j in r["jobs"] if j["source"]=="slack_bot"]; assert len(slack)==11, f"want 11 slack_bot, got {len(slack)}"; non_manifest=[j for j in slack if j["status"]!="manifest"]; assert len(non_manifest)>=6, f"expect >=6 jobs surfaced from registry: {[(j[chr(34)+chr(105)+chr(100)+chr(34)], j[chr(34)+chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115)+chr(34)]) for j in slack]}"; with_nr=[j for j in slack if j["next_run"]]; assert len(with_nr)==11, f"all 11 slack_bot jobs must have next_run populated"; print("OK", len(slack), "slack_bot;", len(non_manifest), "non-manifest;", len(with_nr), "with next_run")'''
research_brief: handoff/current/phase-23.5.2.5-research-brief.md
---

# Contract — phase-23.5.2.5

## Hypothesis

Three coordinated edits make `/api/jobs/all` reflect real slack-bot
job state without requiring any new IPC infrastructure:

1. **`backend/slack_bot/scheduler.py`** — extend `_aps_to_heartbeat`
   to include `next_run_time` (looked up via
   `_scheduler.get_job(event.job_id).next_run_time`) AND add a
   startup state-push loop after `_scheduler.start()` that POSTs
   `{job, status: "scheduled", next_run_time}` for every registered
   job, so the registry is seeded BEFORE any job fires.
2. **`backend/api/job_status_api.py`** — add `next_run_time` field
   to the row schema, persist it in `record_heartbeat()`, and
   export `get_registry_snapshot()` for cross-module reads under
   the existing lock.
3. **`backend/api/cron_dashboard_api.py`** — replace the two-line
   slack_bot static loop at lines 208-209 with a registry-merge
   block that consults `job_status_api.get_registry_snapshot()` per
   manifest entry. Fallback when an entry has no registry row:
   `status="never_run"` (matches `JobStatus.status` default at
   `job_status_api.py:74`), NOT `"manifest"` (which is a pyfinagent-
   local term that doesn't appear in any mainstream orchestrator
   per the brief's Prefect / Airflow / Dagster citations).

After the edits + slack-bot daemon restart, all 11 slack_bot jobs
have `status != "manifest"` and `next_run` populated. The
masterplan criterion at 23.5.3-23.5.13 becomes structurally
satisfiable.

## Research-gate summary

`researcher` agent `a0129c09825c9af61` ran tier=moderate and
returned `gate_passed: true` with:
- 9 external sources fetched in full via WebFetch (≥5 floor cleared)
- 5 snippet-only + 9 read-in-full = 14 URLs (≥10 floor)
- Recency scan 2024-2026 performed (digon.io 2025 confirms single-
  process observability is the dominant pattern; cross-process
  state push remains the only viable approach for pyfinagent's
  topology)
- Three-query discipline followed
- 6 internal files inspected

Brief: `handoff/current/phase-23.5.2.5-research-brief.md`.

**Three explicit recommendations from the brief:**

1. Heartbeat augmentation + startup state-push (NOT periodic poll,
   NOT RPC pull, NOT Redis Pub/Sub).
2. Merge inline in `get_all_jobs()` (NOT inside `_static_to_dict`,
   which stays unchanged for launchd).
3. Fallback status `"never_run"` (NOT `"manifest"`) for jobs
   without a registry row.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied from `.claude/masterplan.json::23.5.2.5.verification`:

```
python3 -c 'import json,urllib.request as u;
  r=json.load(u.urlopen("http://localhost:8000/api/jobs/all"));
  slack=[j for j in r["jobs"] if j["source"]=="slack_bot"];
  assert len(slack)==11, f"want 11 slack_bot, got {len(slack)}";
  non_manifest=[j for j in slack if j["status"]!="manifest"];
  assert len(non_manifest)>=6, f"expect >=6 jobs surfaced from registry";
  with_nr=[j for j in slack if j["next_run"]];
  assert len(with_nr)==11, f"all 11 slack_bot jobs must have next_run populated";
  print("OK", len(slack), "slack_bot;", len(non_manifest), "non-manifest;", len(with_nr), "with next_run")'
```

Decoded into deterministic checks:

1. `/api/jobs/all` returns exactly 11 entries with `source="slack_bot"`.
2. At least 6 of them have `status != "manifest"` (the 6 with
   real heartbeats: morning_digest, daily_price_refresh,
   nightly_mda_retrain, hourly_signal_warmup, nightly_outcome_rebuild,
   cost_budget_watcher). Jobs without heartbeats but seeded by the
   startup state-push will have `status="scheduled"` (also `!=
   "manifest"`), so realistically all 11 will pass this check —
   `>=6` is a conservative floor.
3. All 11 have `next_run is not None` (seeded by startup state-
   push).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief, `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Edit `backend/api/job_status_api.py`:
      - Add `next_run_time: str | None = None` to `JobStatus` dataclass at line 74.
      - In `record_heartbeat()`, persist `next_run_time` from event payload (line 104-109).
      - Add `get_registry_snapshot() -> dict[str, dict]` exported helper that returns a copy of `_REGISTRY` under `_lock`.
      - Add to `__all__`.
   b. Edit `backend/slack_bot/scheduler.py`:
      - Extend `_aps_to_heartbeat` to include `next_run_time`:
        ```python
        job_obj = _scheduler.get_job(event.job_id)
        payload["next_run_time"] = (
            job_obj.next_run_time.isoformat()
            if job_obj and job_obj.next_run_time else None
        )
        ```
      - After `_scheduler.start()` at line 127, add a startup loop:
        ```python
        # phase-23.5.2.5: seed the registry with next_run_time for
        # every registered job so /api/jobs/all has data before any
        # job fires.
        for j in _scheduler.get_jobs():
            try:
                payload = {
                    "job": j.id,
                    "status": "scheduled",
                    "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
                }
                requests.post(_HEARTBEAT_URL, json=payload, timeout=2)
            except Exception as e:
                logger.warning("startup state-push fail-open for %s: %r", j.id, e)
        ```
   c. Edit `backend/api/cron_dashboard_api.py`:
      - Add `from backend.api import job_status_api` to imports.
      - Replace lines 208-209 with:
        ```python
        snapshot = job_status_api.get_registry_snapshot()
        for entry in _SLACK_BOT_JOBS:
            row = snapshot.get(entry["id"], {})
            jobs.append({
                "id": entry["id"],
                "source": "slack_bot",
                "schedule": entry.get("schedule", "?"),
                "next_run": row.get("next_run_time"),
                "last_run": row.get("last_run_at"),
                "status": row.get("status", "never_run"),
                "description": entry.get("description", entry["id"]),
            })
        ```
   d. Restart slack-bot daemon: `pkill -f "slack_bot.app" && nohup .venv/bin/python -m backend.slack_bot.app > handoff/logs/slack_bot.log 2>&1 &`
   e. Backend (uvicorn --reload) auto-picks up cron_dashboard_api.py + job_status_api.py changes. Verify reload happened by tailing backend.log.
   f. Run the immutable verification command verbatim. Capture output.
   g. Add a new test `tests/api/test_cron_dashboard_slack_bot_merge.py` that mocks the registry and asserts the merged output (per brief's test-impact section).
   h. Write `tests/verify_phase_23_5_2_5.py` — replayable verifier.
   i. Write `experiment_results.md`.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness audit
   FIRST, then deterministic re-verification + circular-import
   sanity check + slack-bot log tail confirming startup state-push
   ran.
5. **LOG phase:** append `harness_log.md` AFTER Q/A returns
   PASS/CONDITIONAL. Flip 23.5.2.5 status only after the log
   append.

## Anti-patterns guarded (≥2)

1. **Heavyweight queue/broker** (Redis/RabbitMQ) — pyfinagent is
   local-only single-Mac; in-memory + HTTP push is correct
   (researcher cited this explicitly).
2. **Persistent registry** (BQ/SQLite) — out of scope; the in-
   memory registry's "reset on restart" is acceptable for a local
   deployment per `project_local_only_deployment.md` memory.
3. **Editing `_static_to_dict`** to handle the merge — would
   couple slack_bot and launchd code paths; brief recommends
   keeping it untouched and merging inline at the call site.
4. **Using `"manifest"` as the fallback** — brief notes this is a
   pyfinagent-local term not present in any mainstream orchestrator
   (Prefect / Airflow / Dagster). `"never_run"` is the documented
   term and matches `JobStatus.status` default.
5. **Polling instead of push** — would add latency and unnecessary
   load; the heartbeat path is already in place and just needs the
   one-time startup seed.
6. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Wiring `EVENT_JOB_EXECUTED` listener on the MAIN scheduler
  (`backend/main.py` / `backend/api/paper_trading.py`). That's a
  separate enhancement for the `main_apscheduler` jobs (which
  already pass via `_job_to_dict`).
- Launchd live introspection (`launchctl print` parsing).
  Different bridge, separate substep.
- Persistent registry (file/BQ-backed). In-memory is fine.
- The `next_run` push for `paper_trading_daily` and
  `ticket_queue_process_batch` — those already populate via
  `_job_to_dict` (in-process scheduler).

## Backwards compatibility

- `JobStatus` dataclass gains an optional `next_run_time` field
  (default None) — non-breaking per data-expert.io schema-evolution
  guidance.
- `_aps_to_heartbeat` payload gains an optional `next_run_time`
  field — `record_heartbeat()` reads it via `event.get("next_run_time")`,
  so old slack-bot processes (without the new code) still
  function (their heartbeats just won't include `next_run_time`,
  and registry rows for those jobs surface `next_run_time=None`).
- `_static_to_dict` unchanged for launchd entries — no regression
  on the 6 launchd jobs.
- New `get_registry_snapshot()` helper is purely additive.
- Existing `tests/api/test_cron_dashboard.py:73-81` does NOT assert
  `status == "manifest"`, only key presence — verified by
  researcher.

## Risk

- **Circular import risk**: `cron_dashboard_api.py` imports
  `job_status_api`. `job_status_api.py` does NOT import from
  `cron_dashboard_api.py` (verified by grep). Safe.
- **Registry data race**: `get_registry_snapshot()` must hold
  `_lock` while copying. Failure to do so could expose mid-write
  state. Mitigation: `with _lock: return dict(_REGISTRY)`.
- **Backend reload race**: changes to `cron_dashboard_api.py` will
  trigger uvicorn reload; if the reload happens mid-request the
  client may see a transient 5xx. Mitigation: low risk on a local
  dev server; documented in experiment_results.
- **Slack-bot daemon misses next_run on first fire**: if a job
  fires DURING the startup state-push loop (before the seed POST
  completes), the registry briefly has stale state. Mitigation:
  the next heartbeat (post-execution) overwrites correctly.
- **No regressions to phase-23.5.1 / 23.5.2 verifiers**: their
  jobs are `main_apscheduler` source, which uses `_job_to_dict`
  not `_static_to_dict`. Unaffected by this change.

## References

- Research brief:
  `handoff/current/phase-23.5.2.5-research-brief.md` (researcher
  `a0129c09825c9af61`, 9 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.2.5.verification`.
- Files to edit:
  - `backend/api/job_status_api.py` (lines 60-110, ~109 for the
    new helper).
  - `backend/slack_bot/scheduler.py` (lines 34-60 for listener
    extension, after line 127 for startup state-push).
  - `backend/api/cron_dashboard_api.py` (imports + lines 208-209).
- APScheduler events module:
  https://apscheduler.readthedocs.io/en/3.x/modules/events.html
- APScheduler Job docs:
  https://apscheduler.readthedocs.io/en/3.x/modules/job.html
- Prefect states (no "manifest" state):
  https://docs.prefect.io/v3/concepts/states
- Schema-evolution guide:
  https://www.dataexpert.io/blog/backward-compatibility-schema-evolution-guide
