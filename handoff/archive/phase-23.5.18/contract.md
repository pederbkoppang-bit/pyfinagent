---
step: phase-23.5.18
title: Cron job verification — com.pyfinagent.ablation (launchd)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.ablation"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'''
research_brief: handoff/current/phase-23.5.18-research-brief.md
---

# Contract — phase-23.5.18

## Hypothesis

`com.pyfinagent.ablation` (launchd, `StartCalendarInterval Hour=3
Minute=0` daily) appears in `/api/jobs/all` post-bridge with
`status="ok"` (last fire today 03:21 ET, exit 0, 4 runs total).
Amended criterion (per 23.5.13.3) cleanly met.

## Research-gate summary

`researcher` agent `a5f9ed0263cdf620f` (tier=simple): 6 sources
read in full, 16 URLs, recency scan, 5 internal files.
`gate_passed: true`. Brief:
`handoff/current/phase-23.5.18-research-brief.md`.

Three confirmed answers:
1. **Trigger:** `StartCalendarInterval Hour=3 Minute=0` (daily at
   03:00 ET). Last fire 2026-05-10 03:21, exit 0,
   `total_revenue delta=-0.5350 dsr=1.0000 verdict=keep`.
2. **Bridge surfaces correctly:** `state=not running, last exit
   code=0, runs=4` → bridge maps to `status="ok"`.
3. **Criterion meetable:** `ok` is in the documented set.

**Sleep behavior (researcher's external research):** if host
asleep at 03:00, launchd fires on next wake (Apple-documented).
Multiple missed intervals coalesce to one execution. `WakeSystem`
key is absent from the ablation plist — acceptable for a host
that sleeps rather than shuts down.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.ablation"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
```

## Plan steps

1. (DONE — RESEARCH) gate_passed: true.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. Self-evaluation by Main.
2. Treating `next_run=null` as a defect — by design for the
   amended launchd criterion (StartCalendarInterval next-fire-
   time computable from plist; out of scope to surface).

## Out of scope

- The 1 remaining launchd substep (autoresearch, 23.5.19).
- Plist-parsing for next-fire-time (StartCalendarInterval is
  computable but separate enhancement).
- Refactoring the ablation experiment runner.

## References

- Research brief: `handoff/current/phase-23.5.18-research-brief.md`.
- Phase-23.5.13.3 amendment: `handoff/archive/phase-23.5.13.3/`.
- Plist: `~/Library/LaunchAgents/com.pyfinagent.ablation.plist`.
- Runner: `scripts/ablation/run_ablation.py`.
- Bridge: `backend/api/cron_dashboard_api.py:_launchctl_state`.
- launchd.info sleep behavior: https://www.launchd.info/
- Apple Dev Forums sleep semantics:
  https://developer.apple.com/forums/thread/52369
