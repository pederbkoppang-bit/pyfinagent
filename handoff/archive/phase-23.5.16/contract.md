---
step: phase-23.5.16
title: Cron job verification — com.pyfinagent.frontend (launchd)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.frontend"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'''
research_brief: handoff/current/phase-23.5.16-research-brief.md
---

# Contract — phase-23.5.16

## Hypothesis

`com.pyfinagent.frontend` (launchd, KeepAlive=true + RunAtLoad=true,
ThrottleInterval=5; `next dev --port 3000` per plist) appears in
`/api/jobs/all` post-bridge with `status="running"`. Amended
criterion (per 23.5.13.3) cleanly met.

## Research-gate summary

`researcher` agent `ab1046afad38d7258` (tier=simple): 6 sources
read in full, 14 URLs, recency scan 2024-2026, 4 internal files.
`gate_passed: true`. Brief:
`handoff/current/phase-23.5.16-research-brief.md`.

Three confirmed answers:
1. **Trigger:** KeepAlive=true + RunAtLoad=true + ThrottleInterval=5.
2. **Bridge surfaces correctly:** `state=running, pid=94049, runs=2`
   (note: PID 94049 is post-23.4.0 frontend hotfix kickstart;
   different from any prior session value).
3. **Criterion meetable:** `running` is in the documented set.

**Adjacent finding (NOT in scope):** Next.js docs v16.2.6 prescribe
`next start` (production build) over `next dev` for production-mode
operationalization. pyfinagent uses `next dev` (per plist) for the
local-only deployment. Operationally tolerable per
`project_local_only_deployment.md`; not a verification defect.
Researcher noted Next.js 16.2 (Mar 2026) gave Turbopack a 4× dev
startup speedup — reduces but doesn't eliminate the dev-mode
overhead.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.frontend"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
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
3. Migrating to `next start` here — out of scope.

## Out of scope

- The 3 sibling launchd substeps remaining.
- Migrating to `next start` (operational decision).
- Plist-parsing for next-fire-time (KeepAlive has no next-fire).

## References

- Research brief: `handoff/current/phase-23.5.16-research-brief.md`.
- Phase-23.5.13.3 amendment: `handoff/archive/phase-23.5.13.3/`.
- Plist: `~/Library/LaunchAgents/com.pyfinagent.frontend.plist`.
- Bridge: `backend/api/cron_dashboard_api.py:_launchctl_state`.
- Next.js Deploying docs: https://nextjs.org/docs/app/getting-started/deploying
