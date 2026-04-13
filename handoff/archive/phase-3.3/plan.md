# Phase 3.3: Planner + Evaluator Autonomous Loop — PLAN Phase

**Date:** 2026-04-06 18:40 GMT+2
**Phase:** Phase 3.3 (PLAN phase)
**Status:** Starting now

## Success Criteria (from RESEARCH phase)

From research (5 sources, 20+ URLs collected):

1. **Feedback loop operational** — Planner → Generate → Evaluate → Learn → repeat
2. **Cycle time <1 hour** — Each full loop completes within 60 minutes
3. **Sharpe improvement target** — 5-10% per cycle (1.1705 → 1.23+ after 3-5 cycles)
4. **Regime adaptation** — Planner detects regime changes and adjusts feature/parameter proposals
5. **Learning mechanism** — Evaluator critique feeds back to Planner for next iteration
6. **Integration with spot checks** — Phase 3.2.1 spot checks (2× cost, regime, param sweep) provide feedback signals

## Proposed Architecture

### Components to Build

**1. Planner Enhancement (extend from Phase 3.1)**
- Input: Current Sharpe, DSR, sub-period results, regime info
- Process:
  - Analyze evaluator critique from previous iteration
  - Scan RESEARCH.md for unexplored alpha sources
  - Propose 3-5 next parameter sets (1 low-risk, 2 medium, 1 moonshot)
  - Quantify expected Sharpe improvement per proposal
- Output: JSON proposal list with confidence score + rationale

**2. Evaluator Enhancement (extend from Phase 3.2)**
- Input: Generated backtest results + spot checks (2× cost, regime, param)
- Process:
  - Validate statistical significance (DSR ≥ 0.95)
  - Run spot checks: CostStress, RegimeShift, ParamSweep
  - Compare to baseline + previous best
  - Extract learning insights for Planner
- Output: PASS/FAIL/CONDITIONAL + critique with 5-10 actionable insights

**3. Loop Orchestrator (new)**
- Trigger: End of each harness cycle
- Sequence:
  1. Planner proposes 3-5 sets
  2. Generator runs best-2 proposals in parallel
  3. Evaluator validates results
  4. Accept best result if Sharpe improves AND spot checks pass
  5. Log learning to PLAN.md + memory
  6. Repeat

**4. Learning Storage (new)**
- BQ table `harness_learning_log`: iteration, proposal, result, sharpe_delta, learnings_extracted
- PLAN.md progress: one line per cycle with Sharpe, best param, key learning
- MEMORY.md: cumulative insights (which features matter most, etc.)

### Flow Diagram

```
[Current Best: Sharpe 1.1705]
         ↓
    ┌────────────┐
    │  PLANNER   │ ← reads RESEARCH.md + previous critique
    └────────────┘
         ↓ (3-5 proposals)
    ┌────────────┐
    │ GENERATOR  │ ← runs 2 best proposals
    │ (parallel) │
    └────────────┘
         ↓ (2 new backtests)
    ┌────────────┐
    │ EVALUATOR  │ ← runs spot checks
    │ + Spot     │
    │ Checks     │
    └────────────┘
         ↓ (PASS/FAIL)
    ┌────────────┐
    │ DECIDE     │ ← accept best or try next
    │ + LEARN    │ ← extract insights
    └────────────┘
         ↓
    [Log to BQ + MEMORY + PLAN]
         ↓
    IF Sharpe < target OR regime changed:
         └─→ REPEAT (Planner reads new critique)
    ELSE:
         └─→ STOP (maintain current best)
```

## Implementation Breakdown

**File changes needed:**

1. `backend/agents/planner_agent.py` — Enhance RESEARCH.md input + regime detection
2. `backend/agents/evaluator_agent.py` — Output actionable critiques for Planner
3. `backend/autonomous_harness.py` (new) — Loop orchestrator + learning persistence
4. `backend/backtest/backtest_engine.py` — Optional: add regime detection to results
5. Database: Create `harness_learning_log` table in BigQuery
6. `PLAN.md` — Add loop progress table

**Estimated lines of code:**
- Planner enhancement: 150 lines
- Evaluator enhancement: 100 lines
- Loop orchestrator: 300 lines
- Learning persistence: 80 lines
- **Total: ~630 lines**

## Execution Timeline

- **GENERATE phase:** 3-4 hours (code the orchestrator + persistence)
- **EVALUATE phase:** 1-2 hours (test on 1 cycle, verify spot checks work)
- **Expected completion:** ~18:00-20:00 UTC

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| Planner proposes bad params | Medium | Evaluator spot checks catch them |
| Regime detection misses shift | Medium | Evaluator queries all sub-periods |
| Learning DB fills up quickly | Low | Archive old entries after 30 days |
| Loop never converges | Low | Add max-iterations limit (10) |
| Sharpe regresses on new params | Low | Always keep current best as fallback |

## Approval Gate

- ✅ RESEARCH gate PASSED (5 sources, 20+ URLs collected)
- ⏳ PLAN gate in progress (this document)
- ⏳ GENERATE gate next
- ⏳ EVALUATE gate after GENERATE

**Next:** GENERATE phase — start coding the loop orchestrator

---

*Harness Protocol: RESEARCH ✅ → PLAN ⏳ → GENERATE → EVALUATE → DECIDE → LOG*
