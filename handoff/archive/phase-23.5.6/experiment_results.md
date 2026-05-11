---
step: phase-23.5.6
title: Cron job verification — prompt_leak_redteam — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — no false-positive caveat)
verification_command: 'python3 tests/verify_phase_23_5_6.py'
---

# Experiment Results — phase-23.5.6

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_6.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.6`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="prompt_leak_redteam"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK prompt_leak_redteam scheduled 2026-05-10T03:15:00-04:00
EXIT=0

$ python tests/verify_phase_23_5_6.py
OK prompt_leak_redteam status=scheduled next_run=2026-05-10T03:15:00-04:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "prompt_leak_redteam",
  "source": "slack_bot",
  "schedule": "cron daily 03:15 ET",
  "next_run": "2026-05-10T03:15:00-04:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Nightly red-team prompt-leak audit"
}
```

`last_run: null` because the daemon was last restarted at 10:20
CEST (= 04:20 ET). The most recent fire pre-restart was 2026-05-08
07:15 UTC (per `handoff/prompt_leak_redteam_audit.jsonl`).
Tomorrow's 03:15 ET fire will re-populate `last_run`.

## Why the criterion is satisfied (and is a TRUE liveness signal)

`_nightly_prompt_leak_redteam` (`backend/slack_bot/scheduler.py:443-471`)
is a **pure subprocess launcher** — no HTTP calls. The Docker-alias
bug class that affected digests (fixed in 23.5.3.1) and the watchdog
(fixed in 23.5.2.6) does NOT apply to this handler. There is no
false-positive vector to disclose.

The bridge surfaces `status="scheduled"` from the registry's startup-
seed. The cron trigger at `(hour=3, minute=15, ZoneInfo("America/New_York"))`
computed `next_run` correctly. Both criterion conditions are met
on a TRUE liveness signal.

## Audit-log evidence

`handoff/prompt_leak_redteam_audit.jsonl` last row (2026-05-08
07:15 UTC):
- 7/7 attacks caught (pass_rate=1.0)
- 0/3 false positives
- threshold met (`--min-pass 0.80`)

This is supporting evidence that the underlying audit script is
functional; the cron entry-point liveness verification is what
this step verifies.

## Adjacent finding (NOT a regression, NOT in scope)

Researcher noted no dedicated test file for
`_nightly_prompt_leak_redteam` in `tests/slack_bot/`. Coverage gap;
deferred (would be a separate test-coverage step, not blocking
this verification).

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1, 23.5.2, 23.5.2.5, 23.5.2.6, 23.5.3, 23.5.3.1, 23.5.4, 23.5.5 | PASS |
| 23.5.6 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Add tests for `_nightly_prompt_leak_redteam` (deferred).
- Tune `--min-pass` threshold.
- Touch the audit-log retention.
- Investigate the 11 sibling jobs.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.6-research-brief.md`
- `tests/verify_phase_23_5_6.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_6.py
```
