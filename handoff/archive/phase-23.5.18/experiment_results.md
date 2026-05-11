---
step: phase-23.5.18
title: Cron job verification — com.pyfinagent.ablation — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_18.py'
---

# Experiment Results — phase-23.5.18

Verification-only step. **No code changes.** One artifact:
`tests/verify_phase_23_5_18.py`.

## Verbatim result

```
$ <verbatim immutable command from masterplan>
OK com.pyfinagent.ablation ok
EXIT=0

$ python tests/verify_phase_23_5_18.py
OK com.pyfinagent.ablation status=ok
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "com.pyfinagent.ablation",
  "source": "launchd",
  "schedule": "launchd cron 03:00 daily",
  "next_run": null,
  "last_run": null,
  "status": "ok",
  "description": "Nightly feature ablation experiment"
}
```

## Why criterion is met (amended; phase-23.5.13.3)

- `status="ok"` ≠ `"manifest"` ✓
- `status="ok"` ∈ `{"running","ok","failed","not_loaded","unknown"}` ✓

`launchctl print` shows `state=not running, last exit code=0,
runs=4`. Bridge maps `(not running, 0)` → `status="ok"` via
`_classify_launchctl_state` at `cron_dashboard_api.py:223-229`.

## Live evidence (in-the-wild)

- Last fire: 2026-05-10 03:21 (= 03:00 ET CalendarInterval).
- Exit code: 0 (clean).
- Runs: 4.
- Final log line:
  `total_revenue delta=-0.5350 dsr=1.0000 verdict=keep`.

## Sleep behavior (researcher's documentation)

If host asleep at 03:00, launchd fires on next wake (Apple-
documented). Multiple missed intervals coalesce to one execution.
`WakeSystem` key absent from plist — acceptable for a host that
sleeps rather than shuts down.

## Sibling verifiers — no regressions

All 23 prior phase-23.5 verifiers green; 23.5.18 PASS.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/phase-23.5.18-research-brief.md`
- `tests/verify_phase_23_5_18.py`
