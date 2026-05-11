---
step: phase-23.5.13
title: Cron job verification — cost_budget_watcher (slack_bot, phase-9.8)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="cost_budget_watcher"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.13-research-brief.md
---

# Contract — phase-23.5.13

## Hypothesis

`cost_budget_watcher` (registered at `scheduler.py:532-533`,
`hour=6`, `misfire_grace_time=3600`, `coalesce=True`) appears in
`/api/jobs/all` with `status != "manifest"` and `next_run`
populated. Bridge surfaces `status="ok"` from the registry's
recent fire (`last_run_at="2026-05-08T04:00:04+00:00"` pre-restart;
post-restart will flip to `"scheduled"` until tomorrow's 06:00 CEST
fire).

Per researcher:
- **No Docker-alias bug** — handler makes zero HTTP calls; BQ
  fetch uses `google.cloud.bigquery.Client` directly
  (`cost_budget_watcher.py:91-115`).
- **`heartbeat()` correctly wired** at `cost_budget_watcher.py:56`
  with `IdempotencyKey.daily(JOB_NAME, day=day)`.
- **Production-stub PARTIAL** — performs REAL work (BQ spend
  fetch + real `BudgetEnforcer` evaluation), but `alert_fn` is
  NOT injected by `register_phase9_jobs()` so budget trips only
  log a warning, not a Slack message. Same `alert_fn` wiring gap
  as `weekly_data_integrity` (23.5.12).

Live state: `next_run="2026-05-10T06:00:00+02:00"`.

## NOTE — autonomous harness collision (one residual)

The autonomous mas-harness was paused via `launchctl bootout` in
the prior cycle (23.5.12), but one already-in-flight cycle finished
at 22:51 UTC and re-clobbered `handoff/current/contract.md` with
its "Sprint Contract -- Cycle 1" optimization content. Restored.
No further mas-harness fires expected this session.

## Adjacent finding (NOT in scope)

`alert_fn` not wired by `register_phase9_jobs()` — same pattern as
weekly_data_integrity. Bulk fix at end of phase-9 block (this is
the LAST step of the block, so the fix step should come right
after this one).

## Research-gate summary

`researcher` agent `a42146fafc9b645ff` ran tier=simple and
returned `gate_passed: true`:
- 6 external sources fetched in full (LiteLLM Budget Manager,
  Robust Perception idempotent cron, Pascal Landau BQ cost
  monitoring, OneUptime LLMOps cost-management Jan 2026,
  OneUptime idempotent receiver Jan 2026, GCP BigQuery best-
  practices cost docs).
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 7 internal files inspected.

Brief: `handoff/current/phase-23.5.13-research-brief.md`.

**Final production-stub tally (all 7 phase-9 jobs verified):**
- AFFECTED (3): daily_price_refresh, weekly_fred_refresh,
  nightly_outcome_rebuild — `_default_fetch`/`_default_write`
  stubs produce empty/no-op work.
- NOT AFFECTED — REAL WORK (3): nightly_mda_retrain,
  hourly_signal_warmup, weekly_data_integrity.
- PARTIAL (1): cost_budget_watcher — real BQ work + real
  BudgetEnforcer; only `alert_fn` unwired.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="cost_budget_watcher"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. Self-evaluation by Main — Q/A is mandatory.
2. Wiring alert_fn here — out of scope; bulk fix at end of phase-9
   block (this IS the last step of that block; recommend opening
   23.5.13.1 after this closes).
3. Treating the partial-stub status as a 23.5.13 verdict defect —
   it's a wiring gap, criterion is met.

## Out of scope

- Wiring `alert_fn` (recommend follow-up step 23.5.13.1).
- Wiring production fetch/write across the 3 stub-affected jobs
  (recommend the same 23.5.13.1 to bulk-fix all 4 phase-9 wiring
  gaps).
- The 6 launchd jobs (separate bridge problem; substeps 23.5.14-
  23.5.19).

## References

- Research brief: `handoff/current/phase-23.5.13-research-brief.md`.
- Job source: `backend/slack_bot/jobs/cost_budget_watcher.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:532-533, 544`.
- `BudgetEnforcer`: `backend/autoresearch/budget.py`.
