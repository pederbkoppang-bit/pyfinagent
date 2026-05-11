---
step: phase-23.5.11
title: Cron job verification — nightly_outcome_rebuild (slack_bot, phase-9.6)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_outcome_rebuild"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.11-research-brief.md
---

# Contract — phase-23.5.11

## Hypothesis

`nightly_outcome_rebuild` (registered at `scheduler.py:528-529` —
`hour=4`, `misfire_grace_time=3600`, `coalesce=True`) appears in
`/api/jobs/all` with `status != "manifest"` and `next_run`
populated. Bridge surfaces `status="scheduled"` from the registry
startup-seed.

Per researcher:
- **No Docker-alias bug** — handler makes zero HTTP calls.
- **`heartbeat()` correctly wired** at `nightly_outcome_rebuild.py:22`
  with daily idempotency_key wrapping the full computation block.
- **Production-stub affected** — `_default_fetch()` returns `[]`
  (no BQ read), `_default_write()` returns `len(outcomes)` (no BQ
  write). Same pattern as 23.5.7 / 23.5.8. The fire completes
  successfully and reports `rebuilt=0`. Bulk fix at end of phase-9
  block.

Live state: `next_run="2026-05-10T04:00:00+02:00"` (= 22:00 ET
today; tomorrow 04:00 CEST).

## Research-gate summary

`researcher` agent `a4e2ebadbc42cdd01` (re-spawn after `ae6c85d5bb9acfae4`
stopped mid-task) ran tier=simple and returned `gate_passed: true`:
- 5 sources fetched in full (towards-data-engineering idempotent
  pipelines, Prefect idempotency, OneUptime batch retry Jan 2026,
  Airbyte idempotency, BetterStack APScheduler).
- 10 snippet-only + 5 read-in-full = 15 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 5 internal files inspected.

Brief: `handoff/current/phase-23.5.11-research-brief.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_outcome_rebuild"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. Self-evaluation by Main — Q/A is mandatory.
2. Wiring production fetch/write here — out of scope; bulk fix at
   end of phase-9 block.
3. Conflating production-stub with criterion failure.

## Out of scope

- Wiring real ledger_fetch_fn / outcome_write_fn.
- The 2 sibling phase-9 jobs not yet covered.
- Refactoring outcome-tracking logic.

## References

- Research brief: `handoff/current/phase-23.5.11-research-brief.md`.
- Job source: `backend/slack_bot/jobs/nightly_outcome_rebuild.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:528-529`.
