---
name: masterplan
description: Load current masterplan state, show progress, and guide execution of the next step
---

# /masterplan — PyFinAgent Master Plan Navigator

## Current State

```!
python3 -c "
import json, sys
with open('.claude/masterplan.json') as f:
    mp = json.load(f)
print(f'# Masterplan: {mp[\"project\"]}')
print(f'Updated: {mp[\"updated_at\"]}')
print(f'Goal: {mp[\"goal\"]}')
print()
for phase in mp['phases']:
    icon = {'done': '[x]', 'in-progress': '[~]', 'pending': '[ ]', 'blocked': '[!]'}.get(phase['status'], '[ ]')
    gate = ''
    if phase.get('gate') and not phase['gate'].get('approved', True):
        gate = f' -- GATE: {phase[\"gate\"][\"reason\"]}'
    steps_done = sum(1 for s in phase.get('steps',[]) if s['status']=='done')
    steps_total = len(phase.get('steps',[]))
    progress = f' ({steps_done}/{steps_total})' if steps_total > 0 else ''
    print(f'{icon} {phase[\"id\"]}: {phase[\"name\"]}{progress}{gate}')
    for step in phase.get('steps', []):
        s_icon = {'done': '[x]', 'in-progress': '[~]', 'pending': '[ ]'}.get(step['status'], '[ ]')
        harness = ' [H]' if step.get('harness_required') else ''
        print(f'    {s_icon} {step[\"id\"]}: {step[\"name\"]}{harness}')
print()
# Find next actionable step
for phase in mp['phases']:
    if phase['status'] == 'blocked':
        print(f'## BLOCKED: {phase[\"id\"]} - {phase[\"name\"]}')
        print(f'Gate: {phase[\"gate\"][\"reason\"]}')
        print('Cannot proceed until gate is approved.')
        print()
        continue
    for step in phase.get('steps', []):
        if step['status'] in ('pending', 'in-progress'):
            print(f'## Next: {step[\"id\"]} - {step[\"name\"]} ({step[\"status\"]})')
            if step.get('contract'):
                print(f'Contract: {step[\"contract\"]}')
            if step.get('verification'):
                print(f'Verification: {step[\"verification\"][\"command\"]}')
                print(f'Criteria: {step[\"verification\"][\"success_criteria\"]}')
            if step.get('retry_count', 0) > 0:
                print(f'Retries: {step[\"retry_count\"]}/{step[\"max_retries\"]}')
            break
    else:
        continue
    break
"
```

## Harness Protocol

Every step follows: **RESEARCH -> PLAN -> GENERATE -> EVALUATE -> DECIDE -> LOG**

### Before starting any step:
1. Read the step's `contract` file if it exists
2. Check the step's `verification` criteria
3. **Pass the Research Gate** (mandatory — see RESEARCH.md checklist)
4. Write `handoff/contract_{step_id}.md` with hypothesis + success criteria

### After completing work:
1. Run the verification command
2. Check for PASS/FAIL in evaluator critique
3. Update `.claude/masterplan.json` step status to `done` (triggers TaskCompleted hook verification)

### Gate Checks
- **Approval gates**: Do NOT start work if `gate.approved` is false
- **Dependency gates**: Check that prerequisite phase status is `done`
- Phase 3 is currently gated on Peder's budget approval

## Updating Step Status

Mark a step in-progress:
```bash
python3 -c "
import json
from datetime import datetime, timezone
with open('.claude/masterplan.json') as f: mp = json.load(f)
for p in mp['phases']:
    for s in p.get('steps', []):
        if s['id'] == 'STEP_ID':
            s['status'] = 'in-progress'
mp['updated_at'] = datetime.now(timezone.utc).isoformat()
with open('.claude/masterplan.json', 'w') as f: json.dump(mp, f, indent=2)
"
```

Mark a step done (triggers TaskCompleted hook -> harness-verifier):
```bash
python3 -c "
import json
from datetime import datetime, timezone
with open('.claude/masterplan.json') as f: mp = json.load(f)
for p in mp['phases']:
    for s in p.get('steps', []):
        if s['id'] == 'STEP_ID':
            s['status'] = 'done'
mp['updated_at'] = datetime.now(timezone.utc).isoformat()
with open('.claude/masterplan.json', 'w') as f: json.dump(mp, f, indent=2)
"
```

## Legend
- `[x]` = done
- `[~]` = in-progress
- `[ ]` = pending
- `[!]` = blocked (gate not approved)
- `[H]` = harness-required (verification gate enforced)
