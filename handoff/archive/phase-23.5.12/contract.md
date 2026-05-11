---
step: phase-23.5.12
title: Cron job verification — weekly_data_integrity (slack_bot, phase-9.7)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_data_integrity"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.12-research-brief.md
---

# Contract — phase-23.5.12

## Hypothesis

`weekly_data_integrity` (registered at `scheduler.py:531`) appears
in `/api/jobs/all` with `status != "manifest"` and `next_run`
populated. Bridge surfaces `status="scheduled"` from registry
startup-seed.

Per researcher:
- **No Docker-alias bug** — handler makes zero HTTP calls.
- **`heartbeat()` correctly wired** at `weekly_data_integrity.py:48`
  with `IdempotencyKey.weekly()`.
- **NOT production-stub affected** — performs REAL BQ work: queries
  `__TABLES__` for row counts, computes drift vs JSON snapshot,
  saves updated snapshot.

Live state: `next_run="2026-05-11T05:00:00+02:00"`.

## NOTE — autonomous harness collision

The autonomous harness (launchd `com.pyfinagent.mas-harness`,
interval 1800s) clobbered `handoff/current/contract.md` with its
own "Sprint Contract -- Cycle 1" optimization content while this
per-step cycle was running. Contract has been restored. The
collision is an orthogonal observability gap, surfaced for
follow-up: per-step contracts and autonomous harness contracts
should not share the same file slot.

## Adjacent finding — alert_fn not wired (NOT in scope)

`alert_fn` parameter is NOT wired by `register_phase9_jobs()`.
The job DETECTS drift but cannot SLACK-ALERT. Bulk fix at end of
phase-9 block.

## Research-gate summary

`researcher` agent `a8f924609d1a8e6b1` ran tier=simple and
returned `gate_passed: true`. 7 sources read in full (Atlan, GCP,
Sparvi, Anomalo, BetterStack, APScheduler 3.x, Integrate.io). 16
URLs. Recency scan 2024-2026. 7 internal files. Brief:
`handoff/current/phase-23.5.12-research-brief.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_data_integrity"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract (re-written after harness collision).
3. (DONE — GENERATE) verifier + experiment_results.
4. **EVALUATE phase:** re-spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Out of scope

- Wiring `alert_fn`.
- Coordinating per-step contract slot with autonomous harness.
- The 1 remaining phase-9 job.

## References

- Research brief: `handoff/current/phase-23.5.12-research-brief.md`.
- Job source: `backend/slack_bot/jobs/weekly_data_integrity.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:531`.
