---
step: phase-23.5.15
title: Cron job verification — com.pyfinagent.backend (launchd)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'''
research_brief: handoff/current/phase-23.5.15-research-brief.md
---

# Contract — phase-23.5.15

## Hypothesis

`com.pyfinagent.backend` (launchd, KeepAlive=true + RunAtLoad=true,
ThrottleInterval=5) appears in `/api/jobs/all` post-bridge with
`status="running"`. Per amended criterion (phase-23.5.13.3):
`status != "manifest"` AND `status in ("running","ok","failed",
"not_loaded","unknown")`. Both met.

## Research-gate summary

`researcher` agent `a38ca8bedd686a0ff` (tier=simple): 7 sources
read in full, 17 URLs, recency scan, 3 internal files.
`gate_passed: true`. Brief:
`handoff/current/phase-23.5.15-research-brief.md`.

Three confirmed answers:
1. **Trigger:** KeepAlive=true + RunAtLoad=true +
   ThrottleInterval=5. No `--reload` flag (production-mode).
2. **Bridge surfaces correctly:** `state=running, pid=85245`.
3. **Criterion meetable:** `running` is in the documented set.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
```

## Plan steps

1. (DONE — RESEARCH) gate_passed: true.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded
1. Self-evaluation by Main.
2. Treating `next_run=null` as a defect — by design for KeepAlive.

## Out of scope
- The 4 remaining launchd substeps.
- Adding `--reload` to backend plist.
- Plist-parsing for next-fire-time.

## References
- Research brief: `handoff/current/phase-23.5.15-research-brief.md`.
- Phase-23.5.13.3 amendment: `handoff/archive/phase-23.5.13.3/`.
- Plist: `~/Library/LaunchAgents/com.pyfinagent.backend.plist`.
- Bridge: `backend/api/cron_dashboard_api.py:_launchctl_state`.
