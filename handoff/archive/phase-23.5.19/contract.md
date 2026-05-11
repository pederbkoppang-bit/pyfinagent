---
step: phase-23.5.19
title: Cron job verification — com.pyfinagent.autoresearch (launchd) — final launchd substep
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.autoresearch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'''
research_brief: handoff/current/phase-23.5.19-research-brief.md
---

# Contract — phase-23.5.19

## Hypothesis

`com.pyfinagent.autoresearch` (launchd, `StartCalendarInterval
Hour=2 Minute=0` daily) appears in `/api/jobs/all` post-bridge with
`status="failed"` (last exit code 1, runs=4). Amended criterion
(per 23.5.13.3) cleanly met — `failed` is a documented status
value; the verifier passes while honestly surfacing the
long-standing .env bug.

## Research-gate summary

`researcher` agent `a3139efd55a0ecadb` (tier=simple): 6 sources
read in full (launchd.info, Spacelift exit-code-127, mihow gist,
judy2k gist, CheckTown env-validator, Lucas Pinheiro launchd PATH),
16 URLs, recency scan, 5 internal files. `gate_passed: true`.

Three confirmed answers:
1. **Trigger:** `StartCalendarInterval Hour=2 Minute=0`. Daily at
   02:00. No KeepAlive; RunAtLoad=false.
2. **Bridge surfaces correctly:** `state=not running, last exit
   code=1, runs=4`. Bridge maps to `status="failed"`.
3. **Criterion meetable:** `failed` is in the documented set.

**Critical update from researcher:** exit code has CHANGED
**127 → 1** since phase-23.3.4. Suggests partial operator
remediation of the `.env` leading-space bug — likely lines 24/25
("command not found" → exit 127) were fixed via `sed`, but line
56 (the ANTHROPIC_API_KEY leading-space) OR another error in the
python entrypoint still aborts the script with exit 1. The
description string in `_LAUNCHD_JOBS` ("FAILING exit 127") is now
stale; the live launchctl state is authoritative.

**Mechanism (researcher's external sources):** `set -euo pipefail`
at `run_nightly.sh:6` makes the script abort on the first bad
.env line. `KEY= value` parses as `KEY=""` followed by `value` as
a command — bash exits 127 if "value" isn't on PATH (was the
original bug). After partial fix, exit 1 likely originates from
the python entrypoint (the python `autoresearch` import or
runtime error after `set -a` succeeds).

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.autoresearch"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("status") in ("running","ok","failed","not_loaded","unknown"), f"status not in known set: {j}"; print("OK", j["id"], j["status"])'
```

## Plan steps

1. (DONE — RESEARCH) gate_passed: true.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.
   **This is the FINAL step of phase-23.5.** After flipping
   23.5.19, mark `phase-23.5` itself as `done` in the masterplan.

## Anti-patterns guarded

1. **Spinning the `failed` status as a verification defect** —
   the amended criterion ACCEPTS `failed`; the verifier passes;
   the underlying .env bug is honestly surfaced for operator
   action, NOT masked.
2. **Fixing the .env bug here** — out of scope; sandbox-blocked.
3. **Updating the `_LAUNCHD_JOBS:103` description string** to
   reflect the 127→1 exit-code transition — out of scope (would
   be cosmetic and pre-existing wording is the audit-trail of
   when the bug was first observed).
4. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Fixing `backend/.env` leading-space bug (operator-action;
  sandbox-blocked).
- Fixing the python autoresearch entrypoint (depends on the .env
  fix).
- Updating cosmetic description strings.
- Plist-parsing for next-fire-time.

## References

- Research brief: `handoff/current/phase-23.5.19-research-brief.md`.
- Phase-23.5.13.3 amendment: `handoff/archive/phase-23.5.13.3/`.
- Phase-23.3.5 audit (root cause):
  `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md`.
- Plist: `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist`.
- Wrapper: `scripts/autoresearch/run_nightly.sh`.
- Bridge: `backend/api/cron_dashboard_api.py:_launchctl_state`.
- Spacelift exit-code-127: https://spacelift.io/blog/exit-code-127
- launchd.info: https://www.launchd.info/
