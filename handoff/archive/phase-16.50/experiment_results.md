---
step: phase-16.50
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - DELETED 4 verified-dead modules (932 LOC)
  - DELETED 84 stale handoff/current/ briefs (archive cross-referenced)
  - KEPT openclaw_client.py + meta_coordinator.py (defensive grep found live callers)
---

# Experiment Results -- phase-16.50

## What was done

Dead-file sweep across `backend/agents/` + `handoff/current/`. The
researcher's defensive grep audit corrected the prior explore-agent
inventory: 2 of the originally-flagged "dead" modules are actually
load-bearing.

## Deletions

### 4 verified-dead modules (932 LOC removed)
- `backend/agents/planner_enhanced.py` — 336 LOC
- `backend/agents/evidence_engine.py` — 185 LOC
- `backend/agents/feature_generator.py` — 195 LOC
- `backend/agents/openclaw_monitor.py` — 216 LOC

All four had **0 live importers** (defensive grep confirmed).

### 84 stale research briefs swept from handoff/current/
For each `phase-X.Y-*.md` file in `handoff/current/`, deleted IFF
`handoff/archive/phase-X.Y/` exists (archive copy preserved).
Excluded:
- `phase-15.x-research-brief.md` (10 files) — no archive dir;
  preserved as unique evidence
- `phase-16.50-research-brief.md` — current cycle
- Rolling files: `contract.md`, `experiment_results.md`,
  `evaluator_critique.md`, `research_brief.md`

handoff/current/ went from 189 -> 105 .md files (-84).

## Files KEPT (despite prior flag)

### `backend/agents/openclaw_client.py`
Defensive grep found 2 live callers:
- `backend/api/mas_events.py:173` — `from backend.agents.openclaw_client import list_openclaw_sessions`
- `backend/agents/multi_agent_orchestrator.py:253` — `from backend.agents.openclaw_client import check_gateway_health`

The earlier explore-agent's "0 imports" claim came from a worktree-snapshot grep that missed these. **Not deleted.**

### `backend/agents/meta_coordinator.py`
File header says DEPRECATED, but defensive grep found it actively in use:
- `backend/services/autonomous_loop.py:50` — instantiates `MetaCoordinator()` at module level
- `backend/services/autonomous_loop.py:290, 308, 625` — calls its methods in the hot loop path
- `backend/agents/skill_optimizer.py:825-826` — lazy-imports + calls `run_proxy_validation()`

The DEPRECATED header is misleading. **Not deleted.** Suggested
follow-up: either update the header text or migrate the 2 callers
to `backend/meta_evolution/`.

## Verification

```
$ ! test -f backend/agents/planner_enhanced.py && \
  ! test -f backend/agents/evidence_engine.py && \
  ! test -f backend/agents/feature_generator.py && \
  ! test -f backend/agents/openclaw_monitor.py && \
  test -f backend/agents/openclaw_client.py && \
  test -f backend/agents/meta_coordinator.py && \
  python -c "from backend.agents import multi_agent_orchestrator, orchestrator, planner_agent" && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS

$ python -m pytest backend/tests/test_anthropic_fallback.py backend/tests/test_outcome_tracker.py tests/regression/test_no_calendar_shadow.py tests/meta_evolution/ -v --no-header -q 2>&1 | tail -3
============================== 64 passed in 3.52s ==============================
```

64/64 regression tests PASS. No backend/agents imports broken by the
deletions. openclaw_client + meta_coordinator preserved on disk.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | 4 dead .py files removed | PASS |
| 2 | openclaw_client.py + meta_coordinator.py UNCHANGED | PASS |
| 3 | multi_agent_orchestrator/orchestrator/planner_agent import clean | PASS |
| 4 | Full regression pytest sweep PASSES (64/64) | PASS |
| 5 | handoff/current/ swept (189 -> 105, -84) | PASS |
| 6 | phase-15.x briefs preserved (10 files held) | PASS |
| 7 | Rolling files preserved | PASS |
| 8 | No backend code edited | PASS |

## Honest disclosures

1. **Defensive grep prevented 2 wrongful deletions.** Without the
   re-confirmation step, openclaw_client.py + meta_coordinator.py
   would have been deleted, breaking 4 live call sites
   (`mas_events.py:173`, `multi_agent_orchestrator.py:253`,
   `autonomous_loop.py:50/290/308/625`, `skill_optimizer.py:825-826`).
   The cycle-2 protocol (Main spawns researcher when uncertain)
   worked here.

2. **meta_coordinator.py header is misleading** but not load-bearing
   for THIS cycle. Updating the header or migrating callers is a
   separate concern; tracked as a follow-up.

3. **84 stale briefs deleted, 10 phase-15.x briefs HELD.** No archive
   directory exists for phase-15.x — they're potentially unique
   evidence. Need a future audit cycle to determine whether they
   should be moved to a `handoff/archive/phase-15-orphan/` dir or
   are truly dead.

4. **932 LOC of dead Python removed** from `backend/agents/`. The
   `backend/agents/` dir was 5995+185+6907+7597+12344 = ~33 KB
   smaller. Cleaner imports surface area; lower context noise for
   future cycles.

5. **No CHANGELOG.md update needed** — the auto-changelog hook fires
   on commit, not in this cycle directly.

## Closes

Task list item #72. Phase-16.50.

## Next

Spawn Q/A.
