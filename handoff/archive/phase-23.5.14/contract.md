---
step: phase-23.5.14
title: Cron job verification — com.pyfinagent.backend-watchdog (launchd)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend-watchdog"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.14-research-brief.md
---

# Contract — phase-23.5.14

## Hypothesis

`com.pyfinagent.backend-watchdog` (launchd, `StartInterval=60`,
`RunAtLoad=true`) appears in `/api/jobs/all` post the
phase-23.5.13.2 bridge with `status != "manifest"`. Live state:
`status="ok"`, `last_run=null`, `next_run=null`.

**The criterion is structurally UNMEETABLE in the current
architecture** — `launchctl print` does not expose next-fire-time
for `StartInterval` jobs. Researcher confirmed via 5 independent
authoritative sources (man page, launchd.info, Apple developer
docs, dabrahams gist, live launchctl probe). The bridge at
`cron_dashboard_api.py:293` documents this explicitly:
`"next_run": None,  # launchctl doesn't expose this`.

**Per Anthropic doctrine: criteria are immutable. The criterion
"next_run is not None" was specified by Main when building the
masterplan AHEAD of the bridge implementation, without knowing
launchd's introspection limits. Silently rewriting is forbidden.**

The honest path:
- Run the verification verbatim — it FAILS on the `next_run null`
  assertion (passes the `status != "manifest"` assertion).
- Verdict: **CONDITIONAL** — the FIRST half of the criterion is
  met (status non-manifest); the SECOND half is structurally
  unmeetable.
- Open a follow-up step to **deliberately amend** the 6 launchd
  criteria as a single coordinated change after all 6 launchd
  substeps close (so the amendment is informed by all 6 cycles).

This applies to all 6 launchd substeps (23.5.14-23.5.19), not just
this one.

## Research-gate summary

`researcher` agent `a58f413f8b255a89d` ran tier=simple and
returned `gate_passed: true`:
- 6 external sources fetched in full (≥5 floor): launchctl(1) man
  page, launchd.info, Apple developer docs, dabrahams gist,
  Alan Siu launchctl 2025, plus one cited from prior briefs.
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 4 internal files inspected.

Brief: `handoff/current/phase-23.5.14-research-brief.md`.

**Four answers from researcher:**
1. **Schedule type:** `StartInterval=60` + `RunAtLoad=true`. No
   `KeepAlive` key. Fires every 60s from last-exit + once at load.
2. **Bridge surfaces status correctly:** YES. Live
   `status="ok"`. `_classify_launchctl_state` correctly maps
   `state="not running"` + `last_exit_code=0` to `"ok"`.
   `launchctl print` shows `runs=6497` (healthy).
3. **`next_run is not None` applies to launchd?** NO. Structural
   gap. Same finding will apply to all 6 launchd substeps.
4. **Recommended verdict:** CONDITIONAL with criterion-mismatch
   disclosure. PASS would silently bury the spec defect; FAIL
   would trigger an unresolvable retry loop.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend-watchdog"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. The verification command runs.
2. Job exists in `/api/jobs/all` ✓ (per researcher).
3. `status != "manifest"` ✓ (`status="ok"` post-bridge).
4. `next_run is not None` ✗ (structurally unmeetable —
   launchctl doesn't expose this).

The verification will exit non-zero on assertion 4. **This is the
expected behavior per the criterion as written. CONDITIONAL is
the doctrinaire correct verdict.**

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Run the verification command verbatim. Capture the
      AssertionError on `next_run null`.
   b. Run a SEPARATE softer-criterion verifier
      `tests/verify_phase_23_5_14.py` that checks only `status !=
      "manifest"` (the half-criterion that IS meetable). This is
      the replayable verifier, NOT a criterion amendment.
   c. Write `experiment_results.md` documenting both: the failed
      hard criterion (verbatim AssertionError) AND the soft check
      (status assertion passes).
4. **EVALUATE phase:** spawn fresh `qa` agent. Expecting
   CONDITIONAL.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.
   Document the structural gap; recommend amendment step.

## Anti-patterns guarded

1. **Silently rewriting the criterion** (Anthropic forbids).
2. **Spinning the structural-unmeetability as a PASS** —
   would bury the spec defect.
3. **Failing into a retry loop** — no implementation change can
   make launchctl return next-fire-time for StartInterval jobs;
   FAIL would be unresolvable.
4. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Amending the masterplan criterion for the 6 launchd substeps
  (deferred — single coordinated amendment after all 6 close).
- Adding plist parsing for next-fire-time (StartCalendarInterval
  jobs would have predictable next-fires; StartInterval jobs
  don't).
- The 5 sibling launchd substeps.

## References

- Research brief:
  `handoff/current/phase-23.5.14-research-brief.md` (researcher
  `a58f413f8b255a89d`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.14.verification`.
- launchd plist: `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist`.
- Bridge implementation: `backend/api/cron_dashboard_api.py:_launchctl_state`.
- Phase-23.5.13.2 archive: `handoff/archive/phase-23.5.13.2/`.
- launchctl(1) man page (mirror):
  https://keith.github.io/xcode-man-pages/launchctl.1.html
- launchd.info: https://www.launchd.info/
