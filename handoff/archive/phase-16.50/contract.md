---
step: phase-16.50
title: Dead-file sweep -- delete 4 verified-dead modules + handoff/current cleanup + meta_coordinator decision
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - DELETE backend/agents/{planner_enhanced,evidence_engine,feature_generator,openclaw_monitor}.py (4 files, 932 LOC)
  - KEEP backend/agents/openclaw_client.py (2 live callers found by defensive grep)
  - KEEP backend/agents/meta_coordinator.py (misleading DEPRECATED header; load-bearing in autonomous_loop + skill_optimizer)
  - DELETE ~100+ stale phase-X-research-brief.md files in handoff/current/ (archive copies verified)
  - HOLD 10 phase-15.x briefs (no handoff/archive/phase-15.x/ — these may be unique evidence)
---

# Sprint Contract -- phase-16.50

## Research-gate summary

`handoff/current/phase-16.50-research-brief.md`. tier=simple, internal-only,
gate_passed=true. Defensive grep audit corrected the prior explore-agent
inventory:

- **openclaw_client.py** (originally flagged dead) actually has 2 live
  callers (`backend/api/mas_events.py:173`,
  `backend/agents/multi_agent_orchestrator.py:253`). KEEP.
- **meta_coordinator.py** (originally flagged "deprecated but still
  imported") is actually load-bearing: `autonomous_loop.py:50`
  instantiates at module level + L290/L308/L625 call methods in hot
  loop path; `skill_optimizer.py:825` lazy-imports + calls. KEEP.
  Header is misleading — separate cycle to update header text or
  migrate callers.

## Scope (verified-dead targets)

### DELETE (4 files, 932 LOC)
- `backend/agents/planner_enhanced.py` (336 LOC; 0 live importers; superseded by `planner_agent.py`)
- `backend/agents/evidence_engine.py` (185 LOC; 0 live importers; phase-3.1 stub)
- `backend/agents/feature_generator.py` (195 LOC; 0 live importers; Karpathy AutoResearch stub)
- `backend/agents/openclaw_monitor.py` (216 LOC; 0 live importers; OpenClaw integration stub)

### handoff/current/ stale brief sweep
- DELETE all `phase-X.Y-research-brief.md` files where `handoff/archive/phase-X.Y/` exists AND contains the same brief
- HOLD `phase-15.x-research-brief.md` files (10) — no archive dir exists; preserve as unique evidence
- KEEP rolling files (`contract.md`, `experiment_results.md`, `evaluator_critique.md`, `research_brief.md`)
- KEEP current cycle briefs (16.48, 16.49, 16.50)

## Concrete plan

1. Delete the 4 verified-dead Python modules.
2. Run `python -c "import ast; ast.parse(open('backend/agents/orchestrator.py').read())"` and similar on key remaining modules to confirm no syntax errors introduced.
3. Run full pytest sweep: `python -m pytest backend/tests/ tests/meta_evolution/ tests/regression/ -v --no-header -q 2>&1 | tail -3`. Should remain green.
4. Sweep handoff/current/ stale briefs: for each phase-X.Y-research-brief.md where archive dir exists with the same file, delete the current copy. Spare phase-15.x and the 3 current-cycle briefs.
5. Confirm `git status --short | wc -l` shows the expected delta: 4 deleted .py files + N deleted stale briefs.

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent && \
! test -f backend/agents/planner_enhanced.py && \
! test -f backend/agents/evidence_engine.py && \
! test -f backend/agents/feature_generator.py && \
! test -f backend/agents/openclaw_monitor.py && \
test -f backend/agents/openclaw_client.py && \
test -f backend/agents/meta_coordinator.py && \
python -c "from backend.agents import multi_agent_orchestrator, orchestrator, planner_agent" && \
python -m pytest backend/tests/test_anthropic_fallback.py backend/tests/test_outcome_tracker.py tests/regression/test_no_calendar_shadow.py tests/meta_evolution/ -v --no-header -q 2>&1 | tail -3 | head -1
```

(The 8th line tests that the regression sweep still ends with "X passed" — the import test on line 8 fails fast if any of the 4 deletions broke a module.)

Plus:
- `handoff_briefs_swept`: handoff/current/ has fewer .md files than before; phase-15.x briefs preserved.
- `no_meta_coordinator_changes`: meta_coordinator.py + its 2 importers UNCHANGED.
- `no_openclaw_client_changes`: openclaw_client.py UNCHANGED.

## What Q/A must audit

1. The 4 dead files are gone (test -f returns false).
2. openclaw_client.py + meta_coordinator.py + their callers UNCHANGED.
3. Existing modules still import cleanly (multi_agent_orchestrator, orchestrator, planner_agent).
4. Full regression pytest sweep PASSES (no import-time errors).
5. handoff/current/ has been swept (fewer briefs, but phase-15.x preserved + rolling files preserved + current-cycle briefs preserved).
6. git status shows only the deleted files + handoff sweep + handoff/* rolling for this cycle. No edits to backend code.
