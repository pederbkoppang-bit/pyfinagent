---
step: phase-23.5.10
title: Cron job verification — hourly_signal_warmup (slack_bot, phase-9.5)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="hourly_signal_warmup"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.10-research-brief.md
---

# Contract — phase-23.5.10

## Hypothesis

The `hourly_signal_warmup` job (registered at
`scheduler.py:526-527` with `cron, minute=5` — fires hourly at
HH:05) appears in `/api/jobs/all` with `status != "manifest"` and
`next_run` populated.

Per researcher:
- **No Docker-alias bug** — handler makes zero HTTP calls; pure
  in-process (settings read + injectable compute_fn + injectable
  dict cache).
- **`heartbeat()` correctly wired** with hourly idempotency_key
  (`hourly_signal_warmup:2026-05-09T22` shape).
- **Criterion is TRUE liveness** — confirmed by live logs showing
  `status: started` → `status: ok` cycles every hour. Stub
  `compute_signal_fn` returns `{"score": 0.0}` so the cache fills
  with placeholders, but the infrastructure (heartbeat, idempotency,
  watchlist load, cache write) is real work.

Live state: `status="ok"`, `next_run` populated and advancing
hourly.

**Trigger correction (vs masterplan label):** the schedule label
in `_SLACK_BOT_JOBS` says "phase-9.5 interval" but the actual
trigger is `cron(minute=5)` (every hour at HH:05) — wall-clock
aligned, not strict interval. Researcher confirmed at
`scheduler.py:526-527`. Adding to the schedule-label cosmetic
finding for the deferred fix step.

## Research-gate summary

`researcher` agent `aea5e5105c0b0835c` ran tier=simple and
returned `gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): OneUptime cache-
  warming Jan 2026, Aerospike cache-warming, Pinak Datta Mar 2026
  idempotency, APScheduler CronTrigger, BetterStack APScheduler,
  ThinkingLoop 2025 scheduler strategies.
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 5 internal files inspected.

Brief: `handoff/current/phase-23.5.10-research-brief.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="hourly_signal_warmup"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. Self-evaluation by Main — Q/A is mandatory.
2. Wiring real compute_signal_fn here — out of scope; bulk fix
   at end of phase-9 block.
3. Fixing the schedule label cosmetic — out of scope.

## Out of scope

- Wiring production compute_signal_fn.
- The 3 sibling phase-9 jobs not yet covered.
- Schedule-label cosmetic fix.

## References

- Research brief: `handoff/current/phase-23.5.10-research-brief.md`.
- Job source: `backend/slack_bot/jobs/hourly_signal_warmup.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:526-527`.
