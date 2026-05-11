---
step: phase-23.5.3
title: Cron job verification — morning_digest (slack_bot)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="morning_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.3-research-brief.md
---

# Contract — phase-23.5.3

## Hypothesis

The `morning_digest` job (registered at
`backend/slack_bot/scheduler.py:73-82` with
`CronTrigger(hour=settings.morning_digest_hour, minute=0,
timezone=ZoneInfo("America/New_York"))`) appears in `/api/jobs/all`
with `status != "manifest"` and `next_run` populated. After the
phase-23.5.2.5 bridge, `cron_dashboard_api.py:208-227` merges
`job_status_api.get_registry_snapshot()` into the slack_bot
manifest; the slack-bot scheduler's startup state-push
(`_seed_next_run_registry`) seeds `next_run_time` before any fire.

Live state confirms: `status="scheduled"`,
`next_run="2026-05-09T08:00:00-04:00"`. **The criterion is met.**

## CRITICAL FINDING — criterion is a false positive for this job

**Documented per researcher's brief, NOT used to amend the criterion.**

`_send_morning_digest` (`scheduler.py:146+`) uses
`_BACKEND_URL = "http://backend:8000"` (Docker alias) at lines
~211 and ~214 to call `{_BACKEND_URL}/api/portfolio/performance`
and `{_BACKEND_URL}/api/reports/?limit=5`. This is the SAME bug
class that produced the watchdog spam pre-23.5.2.6 — `backend`
DNS doesn't resolve on the Mac host process, so the httpx calls
raise `ConnectError`.

`_send_morning_digest`'s exception handler at lines 226-227 is
fail-open (`except Exception: logger.exception(...)`) — no
re-raise. APScheduler's `EVENT_JOB_EXECUTED` fires regardless,
so `_aps_to_heartbeat` records `status="ok"` and the dashboard
flips to `status="ok"` after the next 8 AM ET fire. The
`/api/jobs/all` view will say "everything is fine" while the
operator gets ZERO morning digests in Slack.

**Anthropic immutable-criteria doctrine forbids me from amending
the criterion in light of this finding.** The criterion as
written passes; the verdict on this step is PASS. The deeper
correctness bug must be addressed in a separate substep. After
this step closes, I will insert **phase-23.5.3.1** ("Fix Docker-
alias hostname in `_send_morning_digest` + `_send_evening_digest`")
to apply the same `127.0.0.1` repointing pattern that 23.5.2.6
used for the watchdog.

The 23.5.4 step (evening_digest verification) will have the same
false-positive nature, so I'll run 23.5.3.1 BEFORE 23.5.4.

## Research-gate summary

`researcher` agent `aeaed5c5677739e04` ran tier=simple and
returned `gate_passed: true` with:
- 7 external sources fetched in full via WebFetch (≥5 floor)
- 5 snippet-only + 7 read-in-full = 17 URLs (≥10 floor; 17 reported)
- Recency scan 2024-2026 performed (APScheduler 3.11.0 zoneinfo
  migration as the canonical recent change; DST skip bug #606
  unaffects hour=8)
- Three-query discipline followed
- 7 internal files inspected

Brief: `handoff/current/phase-23.5.3-research-brief.md`.

**Key external findings:**
1. `next_run_time.isoformat()` returns ET-offset (e.g., `-04:00`),
   correctly handled by Python's `zoneinfo`.
2. APScheduler 3.11.0 (2024) made `zoneinfo` the canonical
   timezone backend; pyfinagent's `from zoneinfo import ZoneInfo`
   at scheduler.py:8 is correct.
3. DST spring-forward bug (#606) does NOT affect hour=8 (only the
   2-3 AM window).
4. Fall-back day fires ONCE, not twice.

**Key internal findings:**
1. `_send_morning_digest` uses `_BACKEND_URL` Docker alias — the
   false-positive vector documented above.
2. `morning_digest_hour` default is 8 (ET), from `settings.py:199`.
3. `morning_digest` is in `_JOB_NAMES` at `job_status_api.py:63`
   (pre-seeded; bridge surfaces from registry).
4. No tests for `_send_morning_digest` httpx calls — coverage gap.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.3.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="morning_digest"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded into deterministic checks:

1. The verification command exits 0 and prints
   `OK morning_digest <status> <next_run_iso>`.
2. `status != "manifest"` (currently `"scheduled"`).
3. `next_run is not None` (currently
   `"2026-05-09T08:00:00-04:00"`).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief, gate passed.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim from the masterplan.
   b. Write `tests/verify_phase_23_5_3.py` (replayable verifier).
   c. Write `experiment_results.md`.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness
   audit FIRST, then deterministic re-run, then LLM judgment.
   Q/A is told upfront about the false-positive finding so it
   doesn't penalize Main for an "incomplete" verification — the
   criterion was met as written.
5. **LOG phase:** append `harness_log.md` AFTER Q/A returns
   PASS/CONDITIONAL with prominent callout of the false-positive
   finding + the planned 23.5.3.1 follow-up. Flip 23.5.3 status
   only after the log append.
6. **(After 23.5.3 closes)** Insert phase-23.5.3.1 in masterplan
   to fix the Docker-alias bug in digests, then run it through
   full harness BEFORE 23.5.4.

## Anti-patterns guarded (≥2)

1. **Amending the criterion** in light of the false-positive
   finding — Anthropic doctrine: criteria are immutable; if a
   criterion is too loose, that's a finding for the next step,
   not a retrofit on this one.
2. **Spinning the false-positive as a CONDITIONAL** — the
   criterion passed cleanly. The right verdict on this step is
   PASS. The follow-up is a SEPARATE step (23.5.3.1).
3. **Fixing `_send_morning_digest` Docker-alias bug here** — out
   of scope. That's 23.5.3.1's job.
4. **Self-evaluation by Main** — Q/A is mandatory.
5. **Verdict-shopping** — first Q/A run; not applicable, but
   noted.

## Out of scope

- Fixing `_send_morning_digest` Docker-alias bug (deferred to
  23.5.3.1).
- Fixing `_send_evening_digest` (also bugged; deferred to
  23.5.3.1 as a sibling).
- Adding tests for `_send_morning_digest` httpx mock — falls in
  23.5.3.1.
- The 16 sibling jobs in phase-23.5.
- Wiring listener on MAIN scheduler.

## Backwards compatibility

Pure additive: 1 new verifier file + rolling handoff files. No
code changes.

## Risk

- **Backend availability**: verification depends on backend port
  8000. If down, urllib raises and verifier exits non-zero — a
  legitimate failure mode.
- **Bridge regression**: 23.5.2.5's bridge must still be live for
  this verification to pass. If the bridge code regressed, the
  verifier will fail with `status="manifest"`. Mitigation: prior
  step closed PASS today; sibling verifiers re-confirmed green
  in 23.5.2.6's experiment_results.

## References

- Research brief: `handoff/current/phase-23.5.3-research-brief.md`
  (researcher `aeaed5c5677739e04`, 7 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.3.verification`.
- Job registration: `backend/slack_bot/scheduler.py:73-82`.
- Handler with the Docker-alias bug: `_send_morning_digest` at
  `backend/slack_bot/scheduler.py:146+` (lines ~211, ~214).
- 23.5.2.5 bridge: `backend/api/cron_dashboard_api.py:208-227`.
- 23.5.2.6 watchdog fix (template for 23.5.3.1):
  `handoff/archive/phase-23.5.2.6/`.
- APScheduler CronTrigger:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
