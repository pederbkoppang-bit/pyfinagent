---
step: phase-23.5.5
title: Cron job verification — watchdog_health_check (slack_bot)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="watchdog_health_check"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.5-research-brief.md
---

# Contract — phase-23.5.5

## Hypothesis

The `watchdog_health_check` job (registered at
`backend/slack_bot/scheduler.py:97-104` with
`IntervalTrigger(minutes=settings.watchdog_interval_minutes)`,
default 15) appears in `/api/jobs/all` with `status != "manifest"`
and `next_run` populated. This is a TRUE liveness signal — and
the researcher confirmed in-the-wild evidence that the
phase-23.5.2.6 spam-fix is operating correctly.

Live state: `status="ok"`,
`next_run="2026-05-09T22:50:21.067885+02:00"` — the listener has
already recorded successful fires since the 10:20 CEST daemon
restart.

## Research-gate summary

`researcher` agent `a52d055a9652e938f` ran tier=simple and
returned `gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): APScheduler
  IntervalTrigger / User Guide, jdhao 2024 immediate-start, ediri
  Dead-Man's Switch, OneUptime heartbeat Feb 2026, Google SRE
  monitoring distributed systems.
- 8 snippet-only + 6 read-in-full = 14 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 5 internal files inspected.

Brief: `handoff/current/phase-23.5.5-research-brief.md`.

**Researcher's clean recommendation:** criterion is a TRUE
liveness signal. `status="ok"` with advancing `next_run` confirms
(a) APScheduler registered + started the job, (b) at least one
fire executed and the heartbeat reached the backend, (c) the
scheduler is alive and computing future fire times.

**In-the-wild evidence (load-bearing):**
- Daemon restart at 10:20:21 CEST (confirmed in
  `handoff/logs/slack_bot.log`).
- First watchdog fire at 10:35:21 CEST (+15 min — APScheduler
  IntervalTrigger wait-one-interval default per jdhao Nov 2024).
- 49 consecutive fires logged at 15-min intervals through 22:35
  CEST, no gaps.
- **ZERO Slack posts** across all 49 fires. No
  `Watchdog unhealthy transition`, no `Watchdog steady-unhealthy`,
  no `Watchdog recovery` lines. State-machine classified every
  fire as `None→True` (first) then steady `True→True` (rest).

**The phase-23.5.2.6 spam fix is working live as designed.** This
is the empirical proof.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.5.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="watchdog_health_check"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. Verification command exits 0 and prints
   `OK watchdog_health_check <status> <next_run_iso>`.
2. `status != "manifest"` (currently `"ok"` after fires).
3. `next_run is not None` and advances every 15 min.

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief; gate passed;
   in-the-wild proof captured.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim.
   b. Write `tests/verify_phase_23_5_5.py` (replayable).
   c. Write `experiment_results.md` with the 49-fires-zero-posts
      observation prominently logged.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A
   PASS/CONDITIONAL. Flip 23.5.5 status only after the log
   append.

## Anti-patterns guarded

1. **Treating in-the-wild observation as a substitute for the
   immutable criterion** — the 49-fires-zero-posts is supporting
   evidence, but the criterion still has to pass on its own
   terms.
2. **Citing the spam-fix as an architectural change in this
   step** — that was 23.5.2.6's work; this step only verifies.
3. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Tuning `watchdog_interval_minutes`.
- Adding meta-monitoring (the watchdog of the watchdog).
- The 12 sibling jobs.

## Backwards compatibility

Pure additive: 1 new verifier + rolling handoff files.

## Risk

- Backend availability (verifier needs port 8000).
- Bridge regression (23.5.2.5 must still be live).

## References

- Research brief:
  `handoff/current/phase-23.5.5-research-brief.md` (researcher
  `a52d055a9652e938f`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.5.verification`.
- Job registration: `backend/slack_bot/scheduler.py:97-104`.
- Watchdog handler (post-23.5.2.6 fix):
  `backend/slack_bot/scheduler.py:255-322`.
- Phase-23.5.2.6 archive:
  `handoff/archive/phase-23.5.2.6/`.
- APScheduler IntervalTrigger:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/interval.html
- jdhao 2024 immediate-start:
  https://jdhao.github.io/2024/11/02/python_apascheduler_start_job_immediately/
