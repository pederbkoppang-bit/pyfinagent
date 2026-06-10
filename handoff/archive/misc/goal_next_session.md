# CLOSURE -- walk pyfinagent to PRODUCTION_READY

N* = `.claude/masterplan.json::goal` = Profit - Risk - Burn. Every contract.md declares N* delta (P/R/B, quantified or speculative). Unarticulable -> DEFERRED.

## State (cycle 62, commit cfb74e5f, main)

- 609/660 done (92.3%). 4 deferrals. 3 P3 pending unblocked: 37.3.1, 40.3.1, 40.8.2 (operator-gated).
- 614 backend / 62 frontend tests / TS build green / tsc --noEmit exit 0.
- DoD: 8/14 backend PASS (3,4,8,10,11,12,13,14) + DoD-1 calendar-pending. 5 live-blocked (2/5/6/7/9). 0/12 UX DoD.
- Owner-gates closed this session: 38.1, 38.4, 39.1 (all default-OFF).
- Frontend foundation IN: @tanstack/react-table v8.21.3 + @tremor/react 3.18.7 + DataTable + LiveBadge + SectorBarList + 27 vitest tests.

## Read FIRST (in order)

1. CLAUDE.md
2. handoff/current/closure_roadmap.md
3. handoff/current/production_ready_audit_2026-05-23.md
4. handoff/harness_log.md tail cycles 53-62
5. .claude/rules/research-gate.md

## Critical path

35.1 || 44.1 (DONE) -> 36.1 (DONE) || 44.2 -> 37.1 (DONE) || 44.7 -> 35.2/35.3 -> sweep -> 44.10 + backend streams -> 43.0 FINAL GATE.

**Highest-leverage unblocked: phase-44.2 cockpit.** Needs route split, Manage->Drawer migration, sub-routes (positions/trades/nav/reality-gap/exit-quality), Tremor BarList sector concentration, LiveBadge per row, operator_approval_44.2.md + Playwright trace + Lighthouse a11y >=95.

## Operator-only (don't fabricate around)

- `.env` / `.env.example` writes (tool-blocked)
- BQ schema mutations outside autonomous-loop Step 7
- `launchctl unload + load` autoresearch.plist (starts DoD-1 calendar)
- Playwright + Lighthouse for phase-44.X
- Live paper-trading cycles (closes DoD-2/5/6/7/9 over 1-2 weeks)

After any `npm install` in frontend/: `launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.frontend"` (pkill races launchd watchdog).

## 10 integration gates (per step)

1. pytest >= 614 backend + 62 frontend
2. TS build + ast.parse green
3. New UI/operator-visible backend behind flag default OFF
4. BQ migrations idempotent (--verify exits 0)
5. New env vars in backend/.env.example + CLAUDE.md
6. Contract declares N* delta
7. Zero emojis
8. ASCII loggers (scripts/qa/ascii_logger_check.py exits 0)
9. Single source of truth
10. log FIRST (harness_log), flip LAST (masterplan)

## Harness protocol (mandatory)

- Researcher SPAWNED FIRST every step (no carve-outs, per feedback_never_skip_researcher)
- contract.md with N* delta BEFORE generate (per feedback_contract_before_generate)
- Q/A SINGLE spawn after generate, 5-item compliance audit FIRST (per feedback_qa_harness_compliance_first)
- harness_log.md append BEFORE status flip
- Fresh Q/A respawn only AFTER fixing blockers + updating handoff files

## Circuit breakers

- 3x CONDITIONAL same step-id -> Q/A returns FAIL
- >3 cycles on one step -> blocked + STOP
- N* delta unarticulable -> DEFERRED

## Closure gate (phase-43.0)

26-criterion DoD (14 backend + 12 UX). ALL PASS -> file `handoff/current/PRODUCTION_READY_request_<date>.md`. Owner types "PRODUCTION_READY: APPROVED" -> masterplan.project.status -> production_ready.

## Next actionable step query

```bash
source .venv/bin/activate && python -c "
import json
d = json.load(open('.claude/masterplan.json'))
done = {s['id'] for ph in d['phases'] for s in ph.get('steps',[]) if s.get('status')=='done'}
for ph in d['phases']:
    if ph.get('id','').startswith('phase-5'): continue
    for s in ph.get('steps',[]):
        if s.get('status')!='pending': continue
        dep = s.get('depends_on_step',''); lc = s.get('verification',{}).get('live_check')
        if dep and dep not in done: continue
        if lc or 'owner-gated' in s.get('name','').lower(): continue
        print(f\"{s['id']:12s} P:{s.get('priority','?'):3s}  {s.get('name','')[:90]}\")"
```

STOP only on production_ready or operator block. ~30-50 cycles remaining (mostly frontend 44.X + live-cycle waits).
