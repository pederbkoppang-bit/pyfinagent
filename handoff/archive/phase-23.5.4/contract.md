---
step: phase-23.5.4
title: Cron job verification — evening_digest (slack_bot)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="evening_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.4-research-brief.md
---

# Contract — phase-23.5.4

## Hypothesis

The `evening_digest` job (registered at
`backend/slack_bot/scheduler.py:85-94` with `CronTrigger(hour=17,
minute=0, timezone=ZoneInfo("America/New_York"))`) appears in
`/api/jobs/all` with `status != "manifest"` and `next_run`
populated. The phase-23.5.2.5 bridge surfaces real registry data
into the dashboard; the phase-23.5.3.1 Docker-alias fix repointed
both digest httpx calls to localhost.

Live state confirms: `status="scheduled"`,
`next_run="2026-05-09T17:00:00-04:00"`. The criterion is met.

**Unlike 23.5.3, this is NOT a false positive.** The four-link
chain `httpx call → format_evening_digest → chat.postMessage →
status="ok"` is now end-to-end functional:
- httpx hits `127.0.0.1:8000` (confirmed at scheduler.py:247, 250 post-23.5.3.1).
- `/api/paper-trading/trades?limit=10` returns trades data.
- `format_evening_digest` produces 5-6 blocks (well under the
  Slack 50-block limit; section text under 3000 chars).
- `chat.postMessage` accepts the payload.
- `EVENT_JOB_EXECUTED` fires on genuine completion → heartbeat
  listener records `status="ok"` for a real reason.

## Research-gate summary

`researcher` agent `add29c9ad499c973e` ran tier=simple and
returned `gate_passed: true` with:
- 7 external sources fetched in full via WebFetch (≥5 floor),
  with ≥3 NEW reads beyond prior briefs: Slack Block Kit reference,
  chat.postMessage docs, DST handling for cron jobs (InventiveHQ),
  idempotent jobs design (DEV Community), Slack bolt-python
  idempotency (#564), Redis distributed-locking for Slack bots.
- 7 snippet-only + 7 read-in-full = 14 URLs (≥10 floor).
- Recency scan 2024-2026 performed (no findings supersede; 50-block
  / 3000-char limits unchanged from 2024).
- Three-query discipline followed.
- 6 internal files inspected.

Brief: `handoff/current/phase-23.5.4-research-brief.md`.

**Researcher's clean recommendation:** the criterion is a TRUE
liveness signal — no false-positive caveat applies. The 23.5.3
disclosure does NOT carry over.

**Adjacent finding (NOT a regression, NOT in scope):**
`chat.postMessage` has no native idempotency key. Theoretical
double-send risk if daemon restarts within the same second the
job fires. Known architectural limitation of single-instance local
deployment; not introduced by any recent phase. Researcher cited
Redis SET NX dedup pattern as the canonical mitigation — out of
scope for verification.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.4.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="evening_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. Verification command exits 0 and prints `OK evening_digest <status> <next_run_iso>`.
2. `status != "manifest"` (currently `"scheduled"`).
3. `next_run is not None` (currently `"2026-05-09T17:00:00-04:00"`).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief; gate passed.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim from masterplan.
   b. Write `tests/verify_phase_23_5_4.py` (replayable).
   c. Write `experiment_results.md` (clean PASS, no false-positive disclosure).
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness
   audit FIRST, then deterministic re-run, then LLM judgment.
5. **LOG phase:** append `harness_log.md` AFTER Q/A returns
   PASS/CONDITIONAL. Flip 23.5.4 status only after the log
   append.

## Anti-patterns guarded (≥2)

1. **Spinning the adjacent idempotency finding as a CONDITIONAL** —
   it's a separate architectural concern, NOT a verification defect.
   Document and move on.
2. **Adding Redis-based dedup as part of this step** — out of
   scope; would change scope from verification to implementation.
3. **Self-evaluation by Main** — Q/A is mandatory.
4. **Verdict-shopping** — first Q/A run; not applicable.

## Out of scope

- Idempotency for `chat.postMessage` (Redis SET NX or other dedup).
- The 14 sibling jobs in phase-23.5.
- Adding metrics/telemetry.
- Refactoring `_send_evening_digest` or `format_evening_digest`.

## Backwards compatibility

Pure additive: 1 new verifier + rolling handoff files. No code
changes.

## Risk

- Backend availability requirement (verifier needs port 8000).
- Bridge regression (23.5.2.5 must still be live; confirmed by
  prior verifiers).
- DST is NOT a risk for 17:00 ET (researcher confirmed via issue
  #606 — that bug only affects 1-3 AM spring-forward window).

## References

- Research brief:
  `handoff/current/phase-23.5.4-research-brief.md` (researcher
  `add29c9ad499c973e`, 7 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.4.verification`.
- Job registration: `backend/slack_bot/scheduler.py:85-94`.
- Handler (post-23.5.3.1 fix):
  `backend/slack_bot/scheduler.py:240-260` (uses
  `_LOCAL_BACKEND_URL`).
- Bridge: `backend/api/cron_dashboard_api.py:208-227`.
- Slack Block Kit: https://docs.slack.dev/reference/block-kit/blocks/
- chat.postMessage: https://docs.slack.dev/reference/methods/chat.postMessage/
- APScheduler CronTrigger:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
