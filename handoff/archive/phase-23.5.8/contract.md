---
step: phase-23.5.8
title: Cron job verification — weekly_fred_refresh (slack_bot, phase-9.3)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_fred_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.8-research-brief.md
---

# Contract — phase-23.5.8

## Hypothesis

The `weekly_fred_refresh` job (registered via
`register_phase9_jobs()`) appears in `/api/jobs/all` with
`status != "manifest"` and `next_run` populated. The criterion is
met — bridge surfaces `status="scheduled"` from the startup-seed.

Per researcher: **No Docker-alias bug.**
`backend/slack_bot/jobs/weekly_fred_refresh.py` makes zero HTTP
calls to backend; `heartbeat()` has no URL; cross-process push
goes via `_aps_to_heartbeat() → _HEARTBEAT_URL` (127.0.0.1).
External FRED API calls (when production fetch is wired) hit
`api.stlouisfed.org` — not a Docker hostname.

## Adjacent finding (NOT in scope, surface for follow-up)

Researcher noted: `register_phase9_jobs()` at `scheduler.py:535-548`
calls `scheduler.add_job(func, ...)` where `func = getattr(mod, "run")`
**WITHOUT partial application of `fetch_fn` or `write_fn`**. So the
scheduler fires `run()` with zero kwargs, which means:

- `_default_fetch` STUB is called (returns `{s: [] for s in series}` —
  empty dict).
- `_default_write` STUB is called (returns `len(rows)` only).

The job runs cleanly through `heartbeat()`, posts `status="ok"`,
**but does NOT actually fetch from FRED nor write to BQ.**

This is the SAME production-stub limitation researcher noted for
`daily_price_refresh` in 23.5.7. It's a known gap — the comment at
`weekly_fred_refresh.py:37` literally says "injected in tests;
production wraps fredapi" — but the production wrapping was never
done in `register_phase9_jobs()`.

**This affects all 7 phase-9 jobs** (each has `_default_fetch`/
`_default_write` stubs). Recommend a single follow-up step (e.g.,
`phase-23.5.13.1`) to wire production fetch/write in
`register_phase9_jobs()`. Out of scope here; flag for after the
phase-9 block closes.

## Research-gate summary

`researcher` agent `a2c0ac6bdbc1f7775` ran tier=simple and returned
`gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): APScheduler 3.x
  CronTrigger + User Guide, fredapi PyPI, mortada/fredapi GitHub,
  fedfred Medium, idempotency batch-processing Medium.
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 6 internal files inspected.

Brief: `handoff/current/phase-23.5.8-research-brief.md`.

**Three answers from researcher:**
1. **No Docker-alias bug** — handler / heartbeat / FRED API are all
   localhost-or-external; no internal Docker hostname.
2. **`heartbeat()` correctly wired** — uses ISO-week idempotency key,
   skips on re-entry within the same week, emits started/ok/failed.
3. **Criterion is partial liveness** — passes scheduler-registration
   + bridge-pushed `next_run`, but does NOT verify FRED data flow
   (production stubs active). Same gap as 23.5.7.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.8.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="weekly_fred_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. Verification command exits 0.
2. `status != "manifest"`.
3. `next_run is not None`.

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent with explicit Write
   skeleton.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Treating production-stub gap as a 23.5.8 defect** —
   architectural coverage gap, not criterion failure. Same framing
   as 23.5.7.
2. **Wiring production fetch_fn/write_fn here** — out of scope; bulk
   fix at end of phase-9 block.
3. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Wiring real FRED fetch in `register_phase9_jobs()` (deferred).
- Tuning the weekly fire time.
- The 5 sibling phase-9 jobs.

## Backwards compatibility

Pure additive: 1 new verifier + rolling handoff files.

## Risk

- Backend availability requirement.
- Bridge regression.

## References

- Research brief:
  `handoff/current/phase-23.5.8-research-brief.md` (researcher
  `a2c0ac6bdbc1f7775`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.8.verification`.
- Job source: `backend/slack_bot/jobs/weekly_fred_refresh.py`.
- `register_phase9_jobs`: `backend/slack_bot/scheduler.py:535-548`.
- `heartbeat()`: `backend/slack_bot/job_runtime.py:66-114`.
- APScheduler CronTrigger:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
- fredapi: https://pypi.org/project/fredapi/
