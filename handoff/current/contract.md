---
step: phase-23.3.1
title: Main APScheduler audit — paper_trading_daily + queue process_batch (job naming)
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_1.py'
research_brief: handoff/current/phase-23.3.1-external-research.md (also see phase-23.3.1-internal-codebase-audit.md)
---

# Contract — phase-23.3.1

## Hypothesis

Live `/api/jobs/all` returns 2 main-process APScheduler jobs:
- `paper_trading_daily` — id ✓, next_run ✓, but no `name=` so the
  `description` column on /cron renders "_scheduled_run" (the qualname).
- Anonymous queue process_batch — id is the auto-generated UUID
  `2db2dd276ba94305a9aec11a5bb58f6c`; description is
  `lifespan.<locals>.process_batch`.

Researcher confirmed both `add_job` calls are missing `name=` and the
queue call is missing `id=` and `replace_existing=True`. APScheduler
docs explicitly say "MUST define an explicit ID and use
replace_existing=True or you will get a new copy of the job every
time your application restarts." Audit verdict: jobs ARE working as
designed (next_run set, fired correctly), but operator-visibility on
the /cron page is degraded by missing labels.

Two-line fix.

## Research-gate summary

Researcher (adaf19a0d83c77106) returned `gate_passed: true`:
- 7 sources read in full (APScheduler 3.x base scheduler + Job module
  + User Guide + 4.x API + migration guide + agronholm Issue #487 +
  flask-apscheduler Issue #7)
- 17 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 (no new findings supersede 3.x guidance)
- 3 internal files inspected with file:line anchors
- Concrete recommendation: literal kwargs to add to both call sites

## Immutable success criteria (verbatim)

1. `backend/main.py` — the queue scheduler `add_job(process_batch, ...)`
   call now passes `id="ticket_queue_process_batch"`,
   `name="Ticket queue batch processor"`, and
   `replace_existing=True`.
2. `backend/api/paper_trading.py` — the existing
   `_scheduler.add_job(_scheduled_run, ...)` call now passes
   `name="Paper trading daily run"` (existing `id=_scheduler_job_id`
   and `replace_existing=True` unchanged).
3. Live `/api/jobs/all` after backend restart shows BOTH jobs with
   human-readable `id` and `description`. Specifically:
   - `paper_trading_daily` has `description: "Paper trading daily run"`.
   - The queue job has `id: "ticket_queue_process_batch"` and
     `description: "Ticket queue batch processor"`. No UUID hex
     string.
4. `python tests/verify_phase_23_3_1.py` exits 0.
5. `python -c "import ast; ast.parse(open(P).read())"` passes for
   both modified files.
6. The audit deliverable `handoff/current/phase-23.3.1-audit-findings.md`
   exists and documents: (a) both jobs verified working as designed
   (next_run, status), (b) the labels-fix applied, (c) any sibling
   concerns flagged but deferred (none expected).

## Plan steps

1. Edit `backend/main.py:217` to add the three kwargs.
2. Edit `backend/api/paper_trading.py:914` to add `name=`.
3. Restart backend; capture verbatim `/api/jobs/all` output for the
   experiment_results + audit-findings.
4. Write `tests/verify_phase_23_3_1.py` — AST + grep + live HTTP probe
   that asserts the response contains the new id and description.
5. Write `handoff/current/phase-23.3.1-audit-findings.md` summarising
   the audit verdict + delta.
6. Append `harness_log.md` AFTER Q/A PASS.

## Out of scope

- Migration to APScheduler 4.x (`add_schedule` API) — separate effort.
- Renaming any other anonymous APScheduler jobs across the codebase
  beyond the 2 called out above (researcher confirmed no other
  callsites in the main process).
- Slack-bot jobs (separate process, covered by phase-23.3.2/3).

## Backwards compatibility

- Renaming the queue job's id from auto-UUID to
  `"ticket_queue_process_batch"` is safe — researcher's grep found
  zero callsites depending on the auto-UUID.
- Adding `name=` is purely cosmetic; no behavior change.
- `replace_existing=True` is idempotent for the in-memory job store
  (queue_scheduler default) but follows project convention and is
  future-proof for a possible JobStore migration.

## References

- Researcher: `handoff/current/phase-23.3.1-{external-research,internal-codebase-audit}.md`
- `backend/main.py:217` (queue add_job)
- `backend/api/paper_trading.py:914` (paper_trading_daily add_job)
- `backend/api/cron_dashboard_api.py:143` (description = job.name OR job.id)
- APScheduler 3.x User Guide: explicit-id-+-replace_existing rule
