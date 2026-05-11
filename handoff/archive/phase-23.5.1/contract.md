---
step: phase-23.5.1
title: Cron job verification — paper_trading_daily (main_apscheduler)
cycle_date: 2026-05-08
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="paper_trading_daily"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.1-research-brief.md
---

# Contract — phase-23.5.1

## Hypothesis

The `paper_trading_daily` APScheduler job (registered at
`backend/api/paper_trading.py:911-923`, trigger
`cron[day_of_week='mon-fri', hour='14', minute='0']`) is alive and
healthy: `status="scheduled"` (not `"manifest"`) and `next_run` is
populated in `/api/jobs/all`. The criterion is structurally
guaranteed by the dashboard implementation:
`cron_dashboard_api.py:174` derives `status = "scheduled" if nrt is
not None else "paused"`, so a `main_apscheduler` job CANNOT show
`status="manifest"` (that value is reserved by `_static_to_dict()`
for out-of-process entries — slack_bot manifest entries that the
dashboard hasn't been able to refresh yet).

`last_run: null` is **by design**, not a bug — APScheduler 3.x's
`Job` object has no `last_run_time` field, and the main scheduler
does NOT have an `EVENT_JOB_EXECUTED` listener wired (only the
slack-bot scheduler at `backend/slack_bot/scheduler.py:12-14,122-124`
does). The code comment at `cron_dashboard_api.py:173` explicitly
notes: `# APScheduler doesn't expose this; phase-2 if needed`. The
masterplan criterion does not require `last_run` population for
this step.

## Research-gate summary

`researcher` agent `a60d76678e12b724f` ran tier=simple and returned
`gate_passed: true` with:
- 5 external sources fetched in full via WebFetch (≥5 floor cleared)
- 10 snippet-only + 5 read-in-full = 15 URLs (clears the ≥10 floor)
- Recency scan 2024-2026 performed (no APScheduler 4.x findings
  apply — pyfinagent uses 3.x; no supersession)
- Three-query discipline followed
- 6 internal files inspected (incl. `paper_trading.py`,
  `cron_dashboard_api.py`, `main.py`, slack_bot scheduler,
  `tests/verify_phase_23_3_1.py`, phase-23.3.1 archive)

Brief: `handoff/current/phase-23.5.1-research-brief.md`.

**Researcher's recommendation:** the immutable criterion is
**complete and sufficient**. `status != "manifest" AND next_run is
not None` is a structurally-guaranteed liveness signal for
`main_apscheduler` jobs. No code changes needed; the live API state
already satisfies it.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.1.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="paper_trading_daily"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded into deterministic checks:

1. The verification command exits **0** and prints `OK
   paper_trading_daily <status> <next_run_iso>`.
2. The fetched JSON has `status != "manifest"` (currently
   `"scheduled"`).
3. The fetched JSON has `next_run is not None` (currently
   `"2026-05-08T14:00:00-04:00"`).
4. Backend on port 8000 is reachable (urllib won't raise).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief, `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim from the masterplan;
      capture stdout + exit code.
   b. Write `tests/verify_phase_23_5_1.py` — a 3-line replayable
      verifier wrapping the same JSON parse + asserts so Q/A's
      deterministic leg can re-run.
   c. Write `handoff/current/experiment_results.md` with verbatim
      output + cited file:line for the dashboard derivation.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness-
   compliance audit FIRST, then deterministic re-run, then LLM
   judgment.
5. **LOG phase:** append `handoff/harness_log.md` AFTER Q/A returns
   PASS / CONDITIONAL. Flip `.claude/masterplan.json` 23.5.1 status
   only after the log append (log-last per
   `feedback_log_last.md`).

## Anti-patterns guarded (≥2)

1. **Treating `last_run: null` as a failure.** Per
   `cron_dashboard_api.py:173` comment, this is a known
   architectural gap (no `EVENT_JOB_EXECUTED` listener on the main
   scheduler), and the masterplan criterion does NOT require
   `last_run`. Reporting it as a regression here would be a
   false positive.
2. **Amending the criterion to "must include last_run check"** —
   forbidden by Anthropic immutable-criteria doctrine. If a future
   step wants to require `last_run`, the criterion must be set in
   the masterplan at step-creation time, not retrofitted.
3. **Out-of-scope code change** to wire an event listener on the
   main scheduler. That's a separate enhancement, not a
   verification step.
4. **Self-evaluation by Main** — Q/A is mandatory. Same-session
   pragmatism is forbidden.

## Out of scope

- Wiring an `EVENT_JOB_EXECUTED` listener on the main scheduler
  (the researcher noted the implementation path: cache
  `event.scheduled_run_time` in an in-memory dict, surface via
  `_job_to_dict`. NOT in this step.)
- The 18 sibling jobs in phase-23.5. They each get their own
  substep (23.5.2–23.5.19).
- Slack-bot daemon restart (relevant for 23.5.3-23.5.13).
- Launchd live introspection (relevant for 23.5.14-23.5.19).
- Backend `/health` 404 (separate finding from phase-23.4.0
  reconnaissance).

## Backwards compatibility

Pure additive: 1 new verifier file (`tests/verify_phase_23_5_1.py`)
+ rolling handoff files. No code changes to backend or frontend.

## Risk

- **Backend availability** — the verification depends on backend
  port 8000 being reachable. If backend is down, verification
  fails on the urllib request, not on the criterion. That's a
  legitimate failure mode (verifier surfaces "connection refused"
  rather than a false PASS), and operator action is recovery, not
  retrofit.
- **APScheduler scheduler not yet initialized** at probe time —
  status would still appear as the cached manifest. Mitigation:
  the verification's `assert next_run is not None` would catch
  this; a fresh-restart probe would show null. Live state confirms
  this isn't currently the case.

## References

- Research brief: `handoff/current/phase-23.5.1-research-brief.md`
  (researcher `a60d76678e12b724f`, 5 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.1.verification`.
- Dashboard derivation: `backend/api/cron_dashboard_api.py:160-188`
  (especially line 173 `# APScheduler doesn't expose this; phase-2
  if needed` and line 174 `"scheduled" if nrt is not None else
  "paused"`).
- Job registration: `backend/api/paper_trading.py:911-923`.
- Phase-23.3.1 archive (prior audit confirming same healthy
  state): `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md`.
- APScheduler 3.x events module:
  https://apscheduler.readthedocs.io/en/3.x/modules/events.html
- APScheduler stable Job reference:
  https://apscheduler.readthedocs.io/en/stable/modules/job.html
