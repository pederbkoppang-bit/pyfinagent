---
name: masterplan
description: Load current masterplan state, show progress, and guide execution of the next step
allowed-tools: Bash(python3 *)
argument-hint: "[step-id]"
---

# /masterplan — PyFinAgent Master Plan Navigator

## Current State

```!
python3 -c "
import json, os, sys
from collections import Counter

ARG = os.environ.get('CLAUDE_ARGUMENTS', '').strip()
SHOW_ALL = ARG in ('--all', 'all', '-a')
FILTER_ID = None if SHOW_ALL else (ARG or None)

with open('.claude/masterplan.json') as f:
    mp = json.load(f)

# phase-81.0: statuses are now partitioned EXHAUSTIVELY. Previously ACTIVE and
# DONE were two hand-enumerated sets, and anything in NEITHER was invisible to
# the phase render while still being walked by the next-actionable loop -- which
# is how `/masterplan` came to announce `Next actionable: 5.2` from phase-5, a
# phase it never printed. MEASURED 2026-07-31, three phases leaked, not one:
#   phase-5  status='deferred'  11 open steps
#   phase-36 status='deferred'  16 open steps  (holds the NEWEST logged cycle)
#   phase-77 status key ABSENT -> None, 3 open steps
# So a fix that special-cases the literal string 'deferred' would still leak
# phase-77. OPEN is therefore defined as the COMPLEMENT of DONE -- there is no
# third bucket a status can fall into unnoticed.
DONE = {'done', 'completed', 'superseded', 'dropped'}
KNOWN_OPEN = {'in-progress', 'in_progress', 'pending', 'blocked', 'proposed', 'deferred'}

def is_open(s):
    \"\"\"Open == not DONE. Total over every status, including absent/None.\"\"\"
    return (s or '') not in DONE

# Statuses that are neither DONE nor a known-open label. These are NOT dropped
# (that was the bug); they are counted as open and reported loudly so a schema
# drift surfaces at the boot path instead of hiding work.
def unknown_statuses(mp):
    seen = set()
    for p in mp['phases']:
        if is_open(p.get('status')) and (p.get('status') or '') not in KNOWN_OPEN:
            seen.add(repr(p.get('status')))
        for s in p.get('steps', []):
            if is_open(s.get('status')) and (s.get('status') or '') not in KNOWN_OPEN:
                seen.add(repr(s.get('status')))
    return sorted(seen)

# Retained for compatibility with any inline reference below.
ACTIVE = KNOWN_OPEN

def phase_icon(s):
    if s == 'blocked': return '[!]'
    if s in ('in-progress', 'in_progress'): return '[~]'
    if s == 'done': return '[x]'
    if s in ('superseded', 'dropped'): return '[-]'
    return '[ ]'

def step_icon(s):
    if s in ('in-progress', 'in_progress'): return '[~]'
    if s == 'done': return '[x]'
    if s == 'blocked': return '[!]'
    if s in ('superseded', 'dropped'): return '[-]'
    return '[ ]'

def render_step(s, indent='    '):
    harness = ' [H]' if s.get('harness_required') else ''
    pri = f' [{s[\"priority\"]}]' if s.get('priority') else ''
    print(f'{indent}{step_icon(s.get(\"status\",\"\"))} {s[\"id\"]:8s}{pri} {s[\"name\"]}{harness}')

def render_phase_full(p):
    icon = phase_icon(p.get('status',''))
    gate = ''
    if isinstance(p.get('gate'), dict) and not p['gate'].get('approved', True):
        gate = f'  GATE: {p[\"gate\"].get(\"reason\",\"\")[:120]}'
    sd = sum(1 for s in p.get('steps',[]) if s.get('status') == 'done')
    st = len(p.get('steps', []))
    prog = f' ({sd}/{st})' if st else ''
    print(f'{icon} {p[\"id\"]}: {p[\"name\"]}{prog} -- status={p.get(\"status\",\"?\")}{gate}')
    for s in p.get('steps', []):
        render_step(s)

print(f'# Masterplan: {mp[\"project\"]}')
print(f'Updated: {mp[\"updated_at\"]}')
print(f'Goal: {mp[\"goal\"]}')
print()

# Filter mode: expand one phase or one step
if FILTER_ID:
    matched = False
    for p in mp['phases']:
        if p['id'] == FILTER_ID or p['id'] == f'phase-{FILTER_ID}':
            render_phase_full(p); matched = True; break
        for s in p.get('steps', []):
            if s['id'] == FILTER_ID:
                print(f'Phase: {p[\"id\"]} -- {p[\"name\"]}')
                print()
                print(f'{step_icon(s.get(\"status\",\"\"))} {s[\"id\"]} -- {s[\"name\"]}')
                print(f'  status: {s.get(\"status\",\"?\")}, priority: {s.get(\"priority\",\"-\")}, harness: {s.get(\"harness_required\",False)}')
                if s.get('depends_on_step'): print(f'  depends_on_step: {s[\"depends_on_step\"]}')
                if s.get('audit_basis'): print(f'  audit_basis: {s[\"audit_basis\"]}')
                v = s.get('verification', {})
                if v:
                    print(f'  verification.command: {v.get(\"command\",\"-\")}')
                    print(f'  verification.success_criteria:')
                    for c in v.get('success_criteria', []): print(f'    - {c}')
                    if v.get('live_check'): print(f'  verification.live_check: {v[\"live_check\"]}')
                if s.get('notes'): print(f'  notes: {s[\"notes\"]}')
                matched = True; break
        if matched: break
    if not matched:
        print(f'No phase or step found matching: {FILTER_ID}')
    sys.exit(0)

# Default condensed view (or --all dump)
if SHOW_ALL:
    print('## All phases')
    for p in mp['phases']: render_phase_full(p)
else:
    phase_status = Counter(p.get('status','?') for p in mp['phases'])
    step_status = Counter(s.get('status','?') for p in mp['phases'] for s in p.get('steps',[]))
    print('## Summary')
    print(f'  Phases: {sum(phase_status.values())} total -- ' + ' | '.join(f'{v} {k}' for k,v in sorted(phase_status.items(), key=lambda x: -x[1])))
    print(f'  Steps:  {sum(step_status.values())} total -- ' + ' | '.join(f'{v} {k}' for k,v in sorted(step_status.items(), key=lambda x: -x[1])))
    print()

    # phase-81.0: every phase that is not DONE, including 'deferred' and
    # missing-status phases. This set MUST be the same one the next-actionable
    # loop walks, or the render can recommend a step it never showed.
    active_phases = [p for p in mp['phases'] if is_open(p.get('status'))]
    shown_phase_ids = {p['id'] for p in active_phases}
    total_open_steps = sum(
        1 for p in mp['phases'] for s in p.get('steps', []) if is_open(s.get('status'))
    )
    shown_open_steps = 0
    unknowns = unknown_statuses(mp)
    if unknowns:
        print(f'## STATUS DRIFT -- {len(unknowns)} unrecognised status value(s): ' + ', '.join(unknowns))
        print('   Counted as OPEN (never silently dropped). Reconcile the schema or add the label to KNOWN_OPEN.')
        print()
    print(f'## Active phases ({len(active_phases)})')
    for p in active_phases:
        icon = phase_icon(p.get('status',''))
        sd = sum(1 for s in p.get('steps',[]) if s.get('status') == 'done')
        st = len(p.get('steps', []))
        prog = f' ({sd}/{st})' if st else ''
        gate_note = ''
        if isinstance(p.get('gate'), dict) and not p['gate'].get('approved', True):
            gate_note = f'  GATE: {p[\"gate\"].get(\"reason\",\"\")[:80]}...'
        print(f'{icon} {p[\"id\"]:14s}{prog:8s} {p[\"name\"]}{gate_note}')
        # Show only non-done steps inline
        active_steps = [s for s in p.get('steps', []) if is_open(s.get('status'))]
        if active_steps:
            for s in active_steps[:8]:
                render_step(s, indent='      ')
            shown_open_steps += min(len(active_steps), 8)
            if len(active_steps) > 8:
                # phase-81.0: denominator, not a bare ellipsis -- the reader can
                # now tell how much of the phase they are NOT seeing.
                print(f'      ... +{len(active_steps)-8} more of {len(active_steps)} open (use /masterplan {p[\"id\"]} for full)')

# phase-81.0: global not-shown denominator. Re-derived here rather than reusing
# the dashboard's locals so this holds under every invocation branch.
_open_total = sum(1 for p in mp['phases'] for s in p.get('steps',[]) if is_open(s.get('status')))
_open_shown = sum(min(sum(1 for s in p.get('steps',[]) if is_open(s.get('status'))), 8)
                  for p in mp['phases'] if is_open(p.get('status')))
print()
print(f'## Coverage: {_open_shown} of {_open_total} open steps shown above -- {_open_total - _open_shown} NOT shown')

# Find next actionable step.
# phase-81.0 INVARIANT: never name a step from a phase this render suppressed.
# The old loop skipped only `status in DONE`, so it walked 'deferred' and
# missing-status phases that the display loop had filtered out -- producing
# `Next actionable: 5.2` from a phase that appeared nowhere in the output.
# Both loops now share is_open(), so the recommendation is always reachable.
print()
for phase in mp['phases']:
    if phase.get('status') == 'blocked':
        print(f'## BLOCKED: {phase[\"id\"]} -- {phase[\"name\"]}')
        if isinstance(phase.get('gate'), dict):
            print(f'   Gate: {phase[\"gate\"].get(\"reason\",\"\")}')
        continue
    if not is_open(phase.get('status')): continue
    for step in phase.get('steps', []):
        if is_open(step.get('status')) and step.get('status') in ('pending', 'in-progress', 'in_progress'):
            pri = f' [{step[\"priority\"]}]' if step.get('priority') else ''
            print(f'## Next actionable: {step[\"id\"]}{pri} -- {step[\"name\"]} (status={step.get(\"status\",\"?\")})')
            v = step.get('verification', {})
            if v.get('command'): print(f'   verification: {v[\"command\"]}')
            if step.get('retry_count', 0) > 0: print(f'   retries: {step[\"retry_count\"]}/{step.get(\"max_retries\",3)}')
            print()
            print(f'   /masterplan {step[\"id\"]}  -- see full step detail')
            sys.exit(0)

print('No actionable steps -- all phases done or blocked.')
"
```

## Usage

| Invocation | Shows |
|---|---|
| `/masterplan` | Condensed dashboard: summary counts + active phases + their non-done steps + next actionable step |
| `/masterplan <phase-id>` | Expand one phase, all steps (e.g. `/masterplan phase-26`) |
| `/masterplan <step-id>` | Expand one step's full detail (e.g. `/masterplan 26.0`) |
| `/masterplan --all` | Legacy full dump (every phase, every step — large) |

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
