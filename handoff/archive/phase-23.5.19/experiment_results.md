---
step: phase-23.5.19
title: Cron job verification — com.pyfinagent.autoresearch — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA (failed status is honest, NOT a verifier defect)
verification_command: 'python3 tests/verify_phase_23_5_19.py'
---

# Experiment Results — phase-23.5.19

Verification-only step. **No code changes.** One artifact:
`tests/verify_phase_23_5_19.py`.

## Verbatim result

```
$ <verbatim immutable command from masterplan>
OK com.pyfinagent.autoresearch failed
EXIT=0

$ python tests/verify_phase_23_5_19.py
OK com.pyfinagent.autoresearch status=failed
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "com.pyfinagent.autoresearch",
  "source": "launchd",
  "schedule": "launchd cron 02:00 daily",
  "next_run": null,
  "last_run": null,
  "status": "failed",
  "description": "Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"
}
```

## Why criterion is met (amended; phase-23.5.13.3)

- `status="failed"` ≠ `"manifest"` ✓
- `status="failed"` ∈ `{"running","ok","failed","not_loaded","unknown"}` ✓

`launchctl print` shows `state=not running, last exit code=1,
runs=4`. Bridge maps `(not running, 1)` → `status="failed"` via
`_classify_launchctl_state` at `cron_dashboard_api.py:223-229`.

**Verifier passes WHILE honestly surfacing the bug.** This is the
intended design of the amendment — the bridge gives operator-
actionable signal, not silence.

## Critical update — exit code transition 127 → 1

Per researcher's empirical probe vs phase-23.3.4 description:
- **Phase-23.3.4 (2026-04-24):** exit 127 ("command not found")
  caused by `KEY= value` in `backend/.env` lines 24/25/56 — bash
  `set -a; . backend/.env` parsed `KEY=""` and tried to execute
  `value` as a command.
- **Phase-23.5.19 (2026-05-10, today):** exit 1, `runs=4`. The
  127 → 1 transition suggests **partial operator remediation** of
  the .env bug (likely lines 24/25 fixed via `sed`, but line 56
  or another error still aborts the script).

The description string in `_LAUNCHD_JOBS` ("FAILING exit 127") is
now stale; the live launchctl state is authoritative. Cosmetic
fix deferred (out of scope per contract).

## Mechanism (researcher's external sources)

`scripts/autoresearch/run_nightly.sh:6` uses `set -euo pipefail`
which makes the script abort on the first error. After the .env
sources successfully (lines 24/25 fixed), exit 1 likely
originates from the python entrypoint (autoresearch import or
runtime error).

## Sleep semantics

Same as 23.5.18: if host asleep at 02:00, launchd fires on next
wake (Apple-documented). Multiple missed intervals coalesce.
`WakeSystem` key absent — acceptable for sleeping hosts.

## Sibling verifiers — no regressions

All 24 prior phase-23.5 verifiers green; 23.5.19 PASS.

## Findings to surface to the operator

1. **autoresearch is failing nightly at 02:00 with exit 1**
   (no longer 127, but still broken). Root cause likely
   `backend/.env` line 56 (ANTHROPIC_API_KEY leading-space) or
   a python autoresearch import/runtime error.
2. **Operator-fix path** documented in
   `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md:71-87`
   (sandbox-blocked from this session).
3. **The bridge surfaces this honestly** — no masking. Verifier
   passes because `failed` is a documented status; the Slack
   dashboard shows `failed` so the operator sees the broken job.

## What this step does NOT do

- Fix `backend/.env` (sandbox-blocked + operator-action).
- Update the cosmetic description string in `_LAUNCHD_JOBS:103`.
- Refactor the autoresearch script.
- The 0 remaining launchd substeps (this is the LAST one).

## Phase-23.5 closeout (after this step closes)

After 23.5.19 status flips to done, mark `phase-23.5` itself as
done in the masterplan. Total: 25 substeps shipped; 24 PASS + 1
CONDITIONAL (23.5.14, by structural-unmeetability design before
the criterion amendment).

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/phase-23.5.19-research-brief.md`
- `tests/verify_phase_23_5_19.py`
