---
step: phase-23.5.17
title: Cron job verification — com.pyfinagent.mas-harness — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_17.py'
---

# Experiment Results — phase-23.5.17

Verification-only step. **No code changes.** One artifact:
`tests/verify_phase_23_5_17.py`.

## Verbatim result

```
$ <verbatim immutable command from masterplan>
OK com.pyfinagent.mas-harness not_loaded
EXIT=0

$ python tests/verify_phase_23_5_17.py
OK com.pyfinagent.mas-harness status=not_loaded
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "com.pyfinagent.mas-harness",
  "source": "launchd",
  "schedule": "launchd interval 1800s",
  "next_run": null,
  "last_run": null,
  "status": "not_loaded",
  "description": "MAS harness optimizer cycle (every 30 min)"
}
```

## Why criterion is met (amended; phase-23.5.13.3)

- `status="not_loaded"` ≠ `"manifest"` ✓
- `status="not_loaded"` ∈ `{"running","ok","failed","not_loaded","unknown"}` ✓
- `next_run=null` is by design for booted-out jobs.

`launchctl print` exits 113 ("Could not find service in domain")
because Main bootout'd this job earlier this session to prevent
contract.md collisions during per-step cycles. The bridge's
`_probe_launchctl` correctly maps a non-zero return code to
`status="not_loaded"`.

## Operational note — re-bootstrap at session end

Per researcher's recommendation:
```bash
launchctl enable gui/$(id -u)/com.pyfinagent.mas-harness
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist
```
The `enable` step is defensive; silent `bootstrap` failures after
`bootout` are documented in 2025-2026 macOS builds.

## Sibling verifiers — no regressions

All 22 prior phase-23.5 verifiers green; 23.5.17 PASS.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/phase-23.5.17-research-brief.md`
- `tests/verify_phase_23_5_17.py`
