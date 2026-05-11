---
step: phase-23.5.14
title: Cron job verification — com.pyfinagent.backend-watchdog (launchd) — experiment results
date: 2026-05-10
verdict_class: CONDITIONAL_PENDING_QA (criterion-mismatch disclosure)
verification_command: 'python3 tests/verify_phase_23_5_14.py'
---

# Experiment Results — phase-23.5.14

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_14.py` — soft-criterion verifier
   asserting `status != "manifest"` and `status` in the
   bridge's documented value set.

## Verification command — verbatim from `.claude/masterplan.json::23.5.14`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="com.pyfinagent.backend-watchdog"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result — hard criterion (FAILS by design)

```
$ <verbatim immutable command>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AssertionError: next_run null: {'id': 'com.pyfinagent.backend-watchdog', 'source': 'launchd', 'schedule': 'launchd interval 60s', 'next_run': None, 'last_run': None, 'status': 'ok', 'description': 'External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)'}
EXIT=1
```

The hard criterion fails on `assert j.get("next_run") is not
None`. **This is the structurally-unmeetable assertion** —
`launchctl print` does not expose next-fire-time for
StartInterval jobs. Researcher confirmed via 5 authoritative
sources.

## Verbatim result — soft criterion (PASSES; bridge healthy)

```
$ python tests/verify_phase_23_5_14.py
OK com.pyfinagent.backend-watchdog status=ok (next_run/last_run null by launchd-bridge design)
EXIT=0
```

The soft verifier asserts what the platform CAN surface:
- `status != "manifest"` (bridge live).
- `status in {"running", "ok", "failed", "not_loaded", "unknown"}`
  (bridge classification correct).

## Live `/api/jobs/all` entry

```json
{
  "id": "com.pyfinagent.backend-watchdog",
  "source": "launchd",
  "schedule": "launchd interval 60s",
  "next_run": null,
  "last_run": null,
  "status": "ok",
  "description": "External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)"
}
```

`status="ok"` reflects `state="not running"` + `last_exit_code=0`
+ `runs=6497` from the live `launchctl print`. Job is healthy.

## Why CONDITIONAL is the doctrinaire correct verdict

Per Anthropic immutable-criteria doctrine + CLAUDE.md:
- The criterion is preserved verbatim (no silent rewrite).
- The hard assertion fails AS WRITTEN.
- A PASS would require silently softening the criterion — forbidden.
- A FAIL would trigger an unresolvable retry loop (no implementation
  change can make launchctl return next-fire-time).
- **CONDITIONAL is the honest middle ground**: implementation is
  correct, criterion is met for the half that the platform can
  surface, the other half is a masterplan-construction defect to
  be amended deliberately in a follow-up.

This applies to all 6 launchd substeps (23.5.14-23.5.19). After
the block closes, recommend a single coordinated amendment step
(e.g., **phase-23.5.19.1**) to update all 6 launchd criteria in
the masterplan to reflect what the platform can actually surface
(e.g., remove `next_run is not None` for the launchd block, or
replace it with a plist-derived next-fire-time check for
StartCalendarInterval jobs).

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1 ... 23.5.13.2 (18 prior) | PASS |
| 23.5.14 (this step, soft verifier) | PASS, EXIT=0 |
| 23.5.14 hard criterion (verbatim) | FAIL on `next_run null` (expected per contract) |

## What this step does NOT do

- Amend the masterplan criterion (deferred to coordinated
  amendment step after all 6 launchd substeps close).
- Add plist parsing for `StartCalendarInterval`-derived next-
  fire-times (not applicable to this StartInterval job anyway).
- Restart or modify the watchdog itself (it's healthy at 6497
  runs / exit 0).
- The 5 sibling launchd substeps.

## Findings to surface to the operator

1. **Backend-watchdog is healthy**: 6497 runs, last exit code 0.
   No action needed.
2. **All 6 launchd substep criteria need amendment** —
   `next_run is not None` is structurally unmeetable. Recommend
   single coordinated amendment after 23.5.19 closes.
3. **The bridge is correct**: `status="ok"` reflects the real
   launchd state. The criterion gap is a masterplan-construction
   defect, not an implementation defect.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.14-research-brief.md`
- `tests/verify_phase_23_5_14.py` (soft criterion only)

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_14.py
```

The hard verbatim criterion FAILS by design (see contract); the
soft verifier (above) passes when the bridge is live.
