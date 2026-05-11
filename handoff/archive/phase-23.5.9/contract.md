---
step: phase-23.5.9
title: Cron job verification — nightly_mda_retrain (slack_bot, phase-9.4)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_mda_retrain"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.9-research-brief.md
---

# Contract — phase-23.5.9

## Hypothesis

The `nightly_mda_retrain` job appears in `/api/jobs/all` with
`status != "manifest"` and `next_run` populated. Per researcher:

- **No Docker-alias bug** — handler makes zero HTTP calls and
  imports only local modules.
- **`heartbeat()` correctly wired** — uses idempotency_key with
  daily skip on re-entry.
- **Criterion is a TRUE liveness signal** — and notably this job is
  NOT affected by the production-stub pattern: `train_fn` runs
  with default `dsr=0.80` which fails the PromotionGate threshold
  (0.95), but the job still completes the heartbeat path and
  posts `status="ok"`. Historical evidence: registry has
  `last_run_at="2026-05-08T01:00:00.011634+00:00"`.

Live state: `next_run="2026-05-10T03:00:00+02:00"`,
`status="scheduled"` (post the recent daemon restart at 23:24
CEST; will flip to `ok` after tomorrow's 03:00 ET fire).

## Research-gate summary

`researcher` agent `affc655717154ac0e` ran tier=simple and
returned `gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): scikit-learn
  permutation_importance, Robust Perception idempotent cron,
  Google Cloud MLOps, Comet retrain importance, Temporal
  idempotency, lakeFS MLOps pipeline, Neova drift monitoring 2026.
- 9 snippet-only + 6 read-in-full = 15 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 6 internal files inspected.

Brief: `handoff/current/phase-23.5.9-research-brief.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="nightly_mda_retrain"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Self-evaluation by Main** — Q/A is mandatory.
2. **Conflating production-stub gap with this step** — researcher
   noted this job is NOT affected (real train_fn runs; only
   promotion gate fails on stub data). Criterion is true liveness.

## Out of scope

- The 4 sibling phase-9 jobs not yet covered.
- Refactoring MDA logic / promotion gate threshold.

## References

- Research brief:
  `handoff/current/phase-23.5.9-research-brief.md`.
- Job source: `backend/slack_bot/jobs/nightly_mda_retrain.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:524-525`.
