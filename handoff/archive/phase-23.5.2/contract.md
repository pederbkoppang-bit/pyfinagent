---
step: phase-23.5.2
title: Cron job verification — ticket_queue_process_batch (main_apscheduler)
cycle_date: 2026-05-08
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="ticket_queue_process_batch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.2-research-brief.md
---

# Contract — phase-23.5.2

## Hypothesis

The `ticket_queue_process_batch` APScheduler job (registered at
`backend/main.py:197-231` via `AsyncIOScheduler` with `'interval',
seconds=5` and no `end_date`) is alive and healthy:
`status="scheduled"` and `next_run` populated in `/api/jobs/all`.
The criterion is **structurally guaranteed even more strongly than
for 23.5.1**:

- `status="manifest"` is impossible for any `main_apscheduler` job
  (`cron_dashboard_api.py:174` derives `"scheduled" if nrt is not
  None else "paused"`; `"manifest"` is reserved for
  `_static_to_dict()`).
- `next_run is not None` is **tautological** for an `IntervalTrigger`
  without `end_date` — APScheduler 3.x's
  `IntervalTrigger.get_next_fire_time()` only returns `None` when
  `end_date` is exceeded. With no `end_date` configured, the job
  always has a future fire time.

The real discriminating gate for an interval job is `j is not None`
(the job is registered and the scheduler is alive). The
verification command checks all three: existence, status,
non-null next_run.

`last_run: null` is **by design** for the same reason as 23.5.1 —
no `EVENT_JOB_EXECUTED` listener wired on the main scheduler
(`cron_dashboard_api.py:173` comment). Not required by the
criterion.

## Research-gate summary

`researcher` agent `a258e82e537f932f1` ran tier=simple and returned
`gate_passed: true` with:
- 6 external sources fetched in full via WebFetch (≥5 floor cleared
  with 4 NEW interval-specific reads beyond the 23.5.1 reuse set)
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor)
- Recency scan 2024-2026 performed (jdhao.github.io 2024 immediate-
  start finding noted; no 2025-2026 supersession)
- Three-query discipline followed
- 7 internal files inspected (`backend/main.py:197-231`,
  `backend/services/ticket_queue_processor.py`,
  `backend/api/cron_dashboard_api.py`, `backend/db/tickets_db.py`,
  `tests/db/test_tickets_db_no_fd_leak.py`,
  `tests/test_queue_processor.py`,
  `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md`)

Brief: `handoff/current/phase-23.5.2-research-brief.md`.

**Researcher's recommendation:** the immutable criterion is
**complete and sufficient** for IntervalTrigger jobs (in fact
stronger than for CronTrigger — the `next_run is not None` part
is tautological so the real gate is the `j is not None` existence
check, which the verification command also performs).

**Adjacent finding (not a regression, not in scope):** APScheduler
defaults `coalesce=True` + `misfire_grace_time=1s`, which means a
60-second agent call (per the ticket_queue handler design) swallows
~12 missed 5-second fires. This is by design — tickets are not lost
(they remain OPEN in SQLite and are retried on the next successful
batch). FD-leak guard from phase-23.1.19 (`contextlib.closing()` at
`ticket_queue_processor.py:43`) covers this job's connection
lifecycle.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.2.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="ticket_queue_process_batch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded into deterministic checks:

1. The verification command exits **0** and prints
   `OK ticket_queue_process_batch <status> <next_run_iso>`.
2. The fetched JSON has `status != "manifest"` (currently
   `"scheduled"`).
3. The fetched JSON has `next_run is not None` (currently
   `"2026-05-08T18:06:36.633914+02:00"`).
4. Backend on port 8000 is reachable (urllib won't raise).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief, `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim from masterplan;
      capture stdout + exit code.
   b. Write `tests/verify_phase_23_5_2.py` — replayable verifier
      mirroring the 23.5.1 pattern (urllib + assertions + clear
      exit code semantics).
   c. Write `handoff/current/experiment_results.md` with verbatim
      output + cited file:line for the source-of-truth claims.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness-
   compliance audit FIRST, then deterministic re-run, then LLM
   judgment.
5. **LOG phase:** append `handoff/harness_log.md` AFTER Q/A returns
   PASS / CONDITIONAL. Flip 23.5.2 status only after the log
   append.

## Anti-patterns guarded (≥2)

1. **Treating `last_run: null` as a failure** — same architectural
   gap as 23.5.1; not required by the masterplan criterion.
2. **Amending the criterion** in light of the IntervalTrigger
   tautology insight — the criterion stands as-is. The fact that
   `next_run is not None` is tautologically true for this trigger
   class is interesting but does NOT mean the criterion is too
   loose; it means the criterion is satisfied by construction
   plus the existence check. Forbidden by Anthropic immutable-
   criteria doctrine to retrofit.
3. **Reporting the `coalesce=True` fire-swallowing** as a defect —
   it's by design and surfaced as an adjacent finding, not a
   regression.
4. **Out-of-scope code change** to alter the trigger or scheduler
   config. Verification step only.
5. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Wiring the `EVENT_JOB_EXECUTED` listener on the main scheduler
  (architectural enhancement, deferred follow-up).
- Tuning `coalesce` / `misfire_grace_time` / `max_instances`.
- Refactoring `ticket_queue_processor.py`.
- The 17 sibling jobs in phase-23.5.

## Backwards compatibility

Pure additive: 1 new verifier file + rolling handoff files. No
code changes.

## Risk

- Same as 23.5.1: backend must be reachable; scheduler must be
  initialized.
- Additional consideration unique to this job: if the
  ticket_queue handler hangs (e.g., a stuck agent call past the
  60s timeout), `coalesce=True` will collapse missed fires.
  Visible only via stale `last_run` if it were populated — out
  of scope for this verification.

## References

- Research brief: `handoff/current/phase-23.5.2-research-brief.md`
  (researcher `a258e82e537f932f1`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.2.verification`.
- Job registration: `backend/main.py:197-231`.
- Handler: `backend/services/ticket_queue_processor.py`.
- Dashboard derivation: `backend/api/cron_dashboard_api.py:160-188`
  (line 174).
- Phase-23.1.19 FD-leak guard:
  `tests/db/test_tickets_db_no_fd_leak.py` +
  `ticket_queue_processor.py:43`.
- Phase-23.3.1 archive (prior audit confirming healthy state):
  `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md`.
- Phase-23.5.1 archive (sibling step, structural argument):
  `handoff/archive/phase-23.5.1/contract.md` +
  `handoff/archive/phase-23.5.1/research_brief.md`.
- APScheduler 3.x IntervalTrigger:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/interval.html
- APScheduler User Guide:
  https://apscheduler.readthedocs.io/en/3.x/userguide.html
