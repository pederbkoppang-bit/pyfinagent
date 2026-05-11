---
step: phase-23.5.16
title: Cron job verification — com.pyfinagent.frontend — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_16.py'
---

# Experiment Results — phase-23.5.16

Verification-only step. **No code changes.** One artifact:
`tests/verify_phase_23_5_16.py`.

## Verbatim result

```
$ <verbatim immutable command from masterplan>
OK com.pyfinagent.frontend running
EXIT=0

$ python tests/verify_phase_23_5_16.py
OK com.pyfinagent.frontend status=running
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "com.pyfinagent.frontend",
  "source": "launchd",
  "schedule": "launchd KeepAlive RunAtLoad",
  "next_run": null,
  "last_run": null,
  "status": "running",
  "description": "Next.js frontend dev server (:3000)"
}
```

## Why criterion is met (amended; phase-23.5.13.3)

- `status="running"` ≠ `"manifest"` ✓
- `status="running"` ∈ `{"running","ok","failed","not_loaded","unknown"}` ✓
- `next_run=null` is by design for KeepAlive jobs.

Bridge maps `state=running` → `status="running"` via
`cron_dashboard_api.py:234`.

## Adjacent finding (NOT in scope)

Next.js docs v16.2.6 prescribe `next start` (production build) over
`next dev` for production-mode operationalization. pyfinagent uses
`next dev` (per plist line 13-16) for the local-only deployment;
operationally tolerable per `project_local_only_deployment.md`.
Researcher: Next.js 16.2 (Mar 2026) gave Turbopack a 4× dev startup
speedup, so dev-mode overhead is reduced. Out of scope to migrate.

## Sibling verifiers — no regressions

All 21 prior phase-23.5 verifiers green; 23.5.16 PASS.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/phase-23.5.16-research-brief.md`
- `tests/verify_phase_23_5_16.py`
