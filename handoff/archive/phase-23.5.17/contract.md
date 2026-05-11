---
step: phase-23.5.17
title: Cron job verification — com.pyfinagent.mas-harness (launchd)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.mas-harness"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'''
research_brief: handoff/current/phase-23.5.17-research-brief.md
---

# Contract — phase-23.5.17

## Hypothesis

`com.pyfinagent.mas-harness` (launchd, `StartInterval=1800`)
appears in `/api/jobs/all` post-bridge with `status="not_loaded"`
because Main bootout'd it earlier this session (around 23.5.12) to
prevent contract.md collisions. The amended criterion (per
23.5.13.3) accepts `not_loaded` as a documented status — criterion
met cleanly without restoring the job.

## Research-gate summary

`researcher` agent `a04b2f6354ddb949b` (tier=simple): 6 sources
read in full (launchd.info, ss64 launchctl, alansiu.net subcommand
basics, Apple ScheduledJobs docs, masklinn cheat sheet, joelsenders
launchctl misuse), 12 URLs, recency scan, 4 internal files.
`gate_passed: true`. Brief:
`handoff/current/phase-23.5.17-research-brief.md`.

Three confirmed answers:
1. **Trigger:** `StartInterval=1800`. Plist invokes
   `scripts/mas_harness/run_cycle.sh` which wraps
   `claude -p < cycle_prompt.md`.
2. **Bridge surfaces correctly:** `launchctl print` exits 113
   (Could not find service in domain). Bridge maps to
   `status="not_loaded"`.
3. **Criterion meetable:** `not_loaded` is in the documented set.

## Operational note — re-bootstrap at session end

Per researcher's recommendation:
```bash
launchctl enable gui/$(id -u)/com.pyfinagent.mas-harness
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist
```
The `enable` step is defensive — silent `bootstrap` failures after
`bootout` are documented in 2025-2026 macOS builds (per researcher's
recency scan).

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.mas-harness"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
```

## Plan steps

1. (DONE — RESEARCH) gate_passed: true.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. Self-evaluation by Main.
2. Bootstrapping mas-harness back IN this step — out of scope.
   Will do at session end as a documented operational step.
3. Treating the bootout state as a 23.5.17 defect — it's an
   intentional Main-action from earlier in the session, and the
   amended criterion accepts it.

## Out of scope

- The 2 sibling launchd substeps remaining (ablation, autoresearch).
- Bootstrapping mas-harness back (session-end action).
- Refactoring `scripts/mas_harness/run_cycle.sh`.

## References

- Research brief: `handoff/current/phase-23.5.17-research-brief.md`.
- Phase-23.5.13.3 amendment: `handoff/archive/phase-23.5.13.3/`.
- Plist: `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist`.
- Bridge: `backend/api/cron_dashboard_api.py:_launchctl_state`.
- Alan Siu launchctl 2.0 basics:
  https://www.alansiu.net/2023/11/15/launchctl-new-subcommand-basics-for-macos/
