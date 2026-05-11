---
step: phase-23.5.7
title: Cron job verification — daily_price_refresh (slack_bot, phase-9.2)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="daily_price_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.7-research-brief.md
---

# Contract — phase-23.5.7

## Hypothesis

The `daily_price_refresh` job (registered via
`register_phase9_jobs()` in `backend/slack_bot/scheduler.py`) appears
in `/api/jobs/all` with `status != "manifest"` and `next_run`
populated. The criterion is met by the bridge's startup-seed.

Per researcher: **No Docker-alias bug.** The phase-9 jobs use a
DIFFERENT execution path than the 4 core slack-bot jobs:
- `backend/slack_bot/jobs/daily_price_refresh.py` is a pure Python
  function (no HTTP calls).
- `heartbeat()` context manager from `backend/slack_bot/job_runtime.py`
  has no URL — it logs to a sink.
- Cross-process delivery is via `_aps_to_heartbeat()` at
  `scheduler.py:55-93` using `_HEARTBEAT_URL =
  "http://127.0.0.1:8000/api/jobs/heartbeat"` (correctly localhost-
  pinned, not Docker-aliased).

The bug class fixed in 23.5.2.6 + 23.5.3.1 does not apply to any
phase-9 job. Researcher confirmed by reading the function body
verbatim.

## Research-gate summary

`researcher` agent `a796ac63282c1bd52` ran tier=simple and
returned `gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): APScheduler 3.x,
  BetterStack APScheduler scaling, StartDataEngineering idempotency,
  Pinak Datta Mar 2026 idempotency, aetperf Nov 2025 yfinance ETL,
  DEV.to APScheduler best-practices.
- 11 snippet-only + 6 read-in-full = 17 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 6 internal files inspected.

Brief: `handoff/current/phase-23.5.7-research-brief.md`.

**Researcher's three answers:**
1. **No Docker-alias bug** in `daily_price_refresh.py` (zero HTTP
   calls).
2. **`heartbeat()` correctly wired** — has no URL itself; cross-
   process delivery uses `_HEARTBEAT_URL` (127.0.0.1).
3. **Criterion satisfied** — partial liveness (criterion tests
   scheduling, not actual fetch). Production path uses
   `_default_fetch` / `_default_write` STUBS (`jobs/daily_price_refresh.py:44-50`)
   that don't call yfinance or BQ; a successful fire writes no
   real data. This is a known coverage gap, NOT a verification
   defect for the criterion as written.

## CRITICAL adjacent finding (NOT in scope for this step but MUST be surfaced)

Researcher's tail of `handoff/logs/slack_bot.log:334-341` shows:

> `format_evening_digest` raised `KeyError: slice(None, 10, None)` at
> `formatters.py:376` at **23:00:01 CEST** (= 17:00:01 ET, evening
> digest fire time).

This is a **runtime bug** in the digest path that phase-23.5.4
(closed PASS earlier today) did NOT catch — because 23.5.4's
criterion only tests scheduling, not actual delivery. The
post-23.5.3.1 evening_digest fire reached `format_evening_digest`,
which then crashed on a slice access against probably-malformed
trades data.

**This is an HONEST limitation of the criterion** (it tests
scheduling, not end-to-end delivery), NOT a 23.5.4 verdict
defect. The right follow-up: a separate step to fix
`format_evening_digest` (likely after the slack_bot block closes,
or immediately if the operator prefers).

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.7.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="daily_price_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. Verification command exits 0.
2. `status != "manifest"`.
3. `next_run is not None`.

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.
   AFTER closing this step, propose to operator a new follow-up
   step for the `format_evening_digest` KeyError fix.

## Anti-patterns guarded

1. **Spinning the production-stub limitation as a 23.5.7 defect** —
   it's an architectural coverage gap, not a criterion failure.
2. **Fixing the format_evening_digest KeyError here** — out of
   scope; surface for follow-up.
3. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Fixing the `format_evening_digest` KeyError (separate step).
- Wiring real yfinance / BQ in the daily_price_refresh production
  path (separate hardening task).
- The 6 sibling phase-9 jobs.

## Backwards compatibility

Pure additive: 1 new verifier + rolling handoff files.

## Risk

- Backend availability requirement.
- Bridge regression.

## References

- Research brief:
  `handoff/current/phase-23.5.7-research-brief.md` (researcher
  `a796ac63282c1bd52`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.7.verification`.
- Job source: `backend/slack_bot/jobs/daily_price_refresh.py`.
- `heartbeat()`: `backend/slack_bot/job_runtime.py:66-114`.
- Cross-process push: `backend/slack_bot/scheduler.py:55-93`
  (`_aps_to_heartbeat` using `_HEARTBEAT_URL`).
- `format_evening_digest` runtime bug:
  `handoff/logs/slack_bot.log:334-341`,
  `backend/slack_bot/formatters.py:376`.
