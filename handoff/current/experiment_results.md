# phase-4.17 (smoke-test) -- Planning cycle experiment results

## What was built

Masterplan extension: new phase `phase-4.17` (10 steps) appended after
`phase-4.16`. phase-4 step 4.9's stale `blocker` field updated to cite
`phase-4.17` as the supersession target. No code changes -- this is a
planning-only cycle.

## Files changed

1. `.claude/masterplan.json`:
   - Appended phase `phase-4.17` with 10 pending steps (4.17.1..4.17.10).
   - Each step carries `id, name, description, status="pending",
     harness_required=true, verification.command,
     verification.success_criteria, contract=null, retry_count=0,
     max_retries=3`.
   - Updated `phase-4 -> step 4.9 -> blocker` to:
     "Aggregate smoketest superseded by phase-4.17 sub-task tree
     (2026-04-24). Monolithic scripts/smoketest/aggregate.sh retired;
     new gate is phase-4.17.10."
2. `handoff/current/phase-smoke-test-research-brief.md` (pre-existing
   from the research spawn; 550+ lines).

## Verification command output (verbatim)

```
$ python3 -c "
import json
m = json.load(open('.claude/masterplan.json'))
phs = {p['id']: p for p in m['phases']}
assert 'phase-4.17' in phs
steps = phs['phase-4.17']['steps']
assert len(steps) == 10
...
"
PASS: phase-4.17 planned correctly, step 4.9 blocker updated

  4.17.1: Main/Orchestrator agent individual behavior
  4.17.2: Researcher agent individual behavior
  4.17.3: Q/A agent individual behavior
  4.17.4: Inter-agent handoff integrity
  4.17.5: CoALA memory layers (working / episodic / semantic / procedural)
  4.17.6: End-to-end signal generation with evidence traceability
  4.17.7: Paper trading execution against virtual portfolio
  4.17.8: Slack interface startup + command registration
  4.17.9: Self-update deploy system audit
  4.17.10: Aggregate gate -- full go_live_drills suite + critical-incident check
```

## Success-criteria coverage

All 4 planning-cycle criteria met:
1. phase-4.17 present with exactly 10 steps (`4.17.1` ... `4.17.10`). ✓
2. Every step has required fields + >= 3 success_criteria. ✓
3. phase-4 step 4.9 `blocker` updated to cite phase-4.17. ✓
4. JSON remains valid (json.load succeeds). ✓

## Notes

- **Execution** of 4.17.1 through 4.17.10 is the subject of 10
  follow-up harness cycles (one per step), which begin in the next
  messages. Each step has a concrete verification command that
  invokes a `scripts/go_live_drills/*_test.py` script. Several of
  those test scripts don't exist yet -- they'll be authored as part
  of each step's Generator phase.
- **Autonomous harness warning**: `backend/autonomous_harness.py`
  is deprecated. The researcher flagged this; none of the 10 smoke
  steps exercises it.
- **No real trading has been executed** to date. The paper-trading
  smoke step (4.17.7) is the first end-to-end proof that the virtual
  portfolio accepts an order + writes to BQ.

---

## Cycle-2 follow-up (2026-04-24)

Q/A-v1 returned CONDITIONAL with 2 coverage gaps flagged. Fix:

1. Added `4.17.11 OpenClaw runtime on Mac Mini -- cron/launchd
   health`. Verification script `scripts/go_live_drills/openclaw_runtime_test.py`.
   5 success criteria covering launchd plist load, venv/cwd
   targeting, last-invocation cadence, no-crashloop-in-log.
2. Added `4.17.12 F1 failure-discipline recovery drill (planted-
   fault injection)`. Verification script
   `scripts/go_live_drills/f1_recovery_drill.py`. 5 success criteria
   covering consecutive_fails=3, certified_fallback raised,
   revert-not-restart, CRITICAL logged, no-infinite-retry.

phase-4.17 now has 12 steps (was 10). All per-step fields still
valid. Updated planning-cycle verification re-runs:

```
$ python3 -c "
import json
m = json.load(open('.claude/masterplan.json'))
phs = {p['id']: p for p in m['phases']}
steps = phs['phase-4.17']['steps']
assert len(steps) == 12, f'want 12, got {len(steps)}'
assert [s['id'] for s in steps] == [f'4.17.{i}' for i in range(1, 13)]
for s in steps:
    assert s['status'] == 'pending'
    assert s['harness_required'] is True
    assert len(s['verification']['success_criteria']) >= 3
print('PASS: 12-step plan valid')
"
```
