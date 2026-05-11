---
step: phase-23.5.6
title: Cron job verification — prompt_leak_redteam (slack_bot)
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 -c ''import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="prompt_leak_redteam"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'''
research_brief: handoff/current/phase-23.5.6-research-brief.md
---

# Contract — phase-23.5.6

## Hypothesis

The `prompt_leak_redteam` job (registered at
`backend/slack_bot/scheduler.py:109-117` with
`CronTrigger(hour=3, minute=15, timezone=ZoneInfo("America/New_York"))`)
appears in `/api/jobs/all` with `status != "manifest"` and `next_run`
populated. **This is a TRUE liveness signal — no false-positive
caveat.**

Researcher confirmed `_nightly_prompt_leak_redteam` is a **pure
subprocess launcher** (`scheduler.py:443-471`) that calls
`subprocess.run(["python", str(script), "--min-pass", "0.80"], ...)`
with a filesystem path resolved from `__file__`. There are ZERO
HTTP calls to `_BACKEND_URL` or any URL inside the function. The
Docker-alias bug class that affected digests (fixed in 23.5.3.1)
and the watchdog (fixed in 23.5.2.6) does NOT apply here.

Live state: `next_run="2026-05-10T03:15:00-04:00"` (next 3:15 AM ET
fire is tomorrow). Bridge merge surfaces `status="scheduled"` from
the registry's startup-seed.

## Research-gate summary

`researcher` agent `aff1da525f9a69d38` ran tier=simple and returned
`gate_passed: true` with:
- 5 external sources fetched in full (≥5 floor): OWASP LLM01:2025
  + LLM07:2025 prompt-injection / system-prompt-leakage, CronMonitor
  2025 timezone guide, Hubifi audit-trail guide, NVISO Feb 2026
  automated LLM red-teaming.
- 10 snippet-only + 5 read-in-full = 15 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 7 internal files inspected.

Brief: `handoff/current/phase-23.5.6-research-brief.md`.

**Researcher's three answers:**
1. **No Docker-alias bug** — handler is subprocess-only, no HTTP.
2. **Criterion is sufficient** — clean PASS.
3. **Audit log healthy** — `handoff/prompt_leak_redteam_audit.jsonl`
   last run 2026-05-08 07:15 UTC: 7/7 attacks caught, 0/3 false
   positives, pass_rate=1.0.

**Adjacent finding (NOT a regression, NOT in scope):** no dedicated
test file for `_nightly_prompt_leak_redteam` in `tests/slack_bot/`.
Coverage gap; deferred.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.6.verification`:

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="prompt_leak_redteam"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

Decoded:
1. Verification command exits 0 and prints
   `OK prompt_leak_redteam <status> <next_run_iso>`.
2. `status != "manifest"` (currently `"scheduled"`).
3. `next_run is not None` (currently
   `"2026-05-10T03:15:00-04:00"`).

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:** verifier + experiment_results.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Treating subprocess-launcher as suspect for Docker-alias** —
   researcher confirmed: no HTTP calls; the bug class doesn't apply.
2. **Adding test coverage for the audit script** — out of scope;
   note the gap as adjacent finding.
3. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Adding test coverage for `_nightly_prompt_leak_redteam` (deferred).
- Tuning the audit-log retention.
- The 11 sibling jobs.
- Changing the `--min-pass 0.80` threshold.

## Backwards compatibility

Pure additive: 1 new verifier + rolling handoff files.

## Risk

- Backend availability requirement.
- Bridge regression (must still be live).
- DST spring-forward "skip" at 03:15 ET happens once per year — APScheduler-documented behavior, not a bug. Researcher cited.

## References

- Research brief:
  `handoff/current/phase-23.5.6-research-brief.md` (researcher
  `aff1da525f9a69d38`, 5 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.6.verification`.
- Job registration: `backend/slack_bot/scheduler.py:109-117`.
- Handler: `backend/slack_bot/scheduler.py:443-471` (subprocess-only).
- Audit log shape: `handoff/prompt_leak_redteam_audit.jsonl`.
- OWASP LLM01:2025: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- OWASP LLM07:2025: https://genai.owasp.org/llmrisk/llm072025-system-prompt-leakage/
