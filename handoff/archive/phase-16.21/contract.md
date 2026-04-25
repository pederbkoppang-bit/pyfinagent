---
step: phase-16.21
title: Layer-1 analysis + outcome/memory loops (closes 16.2)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.21

## Research-gate summary

`handoff/current/phase-16.21-research-brief.md`. tier=simple, 5 in-full, 15 URLs, recency scan, gate_passed=true.

## Verification command status (immutable)

Three independent imports tested. All FAIL:
1. `from backend.tasks.analysis import run_analysis_pipeline` -> ImportError (function not in `tasks/analysis.py`; only `run_analysis_task` exists at line 35)
2. `from backend.services.outcome_tracker import evaluate_recent` -> ImportError (only `OutcomeTracker` class at line 28)
3. `from backend.agents.memory import retrieve_memories` -> ImportError (only `FinancialSituationMemory` class + `build_situation_description` + `generate_reflection`)

**Same structural pattern as phase-16.20**: masterplan verification command targets functions that were never written. Researcher confirmed and Main double-checked the underlying classes (researcher had some inaccuracies — `AnalysisOrchestrator` does NOT exist in tasks/analysis.py, classes need init args).

## Hypothesis

The underlying analysis pipeline DOES work for Monday paper-trading (it's exercised via the `/api/analyze` endpoint and the daily autonomous loop, NOT via these missing module-level wrappers). What 16.21 was supposed to verify (the wrappers as test shims) is missing. Same disposition as 16.20: honest CONDITIONAL with follow-up tickets, NOT a forced PASS or scope-expansion to write the wrappers.

## Why CONDITIONAL, not implement-now

Per the ExitPlanMode-approved plan: "verification-only sweep + Anthropic key swap reminder". Implementing 3 module-level wrappers is feature work, not a 2-line patch. Researcher's claim that wrappers are "5-10 lines each" was based on an incorrect assumption that the underlying entry points were trivial — they're not (OutcomeTracker needs `settings`, FinancialSituationMemory needs `name`, AnalysisOrchestrator is in a different module entirely). The wrappers would need real plumbing: ~40-80 lines + tests + an end-to-end Vertex AI live-call (which itself was the original 16.2 blocker before commit f2e8ce28).

**Monday is NOT blocked** because:
- The daily paper-trading cycle runs through `paper_trading.py::_scheduled_run` → `autonomous_loop.run_daily_cycle()` which uses the AnalysisOrchestrator class directly via the existing infrastructure, not these missing test-wrapper functions
- The /api/analyze endpoint is the live entry to the 28-agent pipeline; it doesn't go through `run_analysis_pipeline` either
- Monday's signal generation is independent of whether 16.21's verification command can be made to pass

## Plan steps

1. Run the immutable verification command verbatim. (Already done above — 3 ImportErrors.)
2. Probe the underlying classes to confirm they EXIST (even if not callable as researcher described). Done — classes present but with positional-arg init.
3. Spawn Q/A; ask the same question as 16.20: CONDITIONAL or FAIL?
4. If CONDITIONAL accepted: log + flip + 3 follow-up tickets + explicit "16.2 stays in-progress" disclosure (mirror of 16.3 in 16.20).
5. Proceed to 16.22.

## What Q/A must audit

1. ImportError on each of the 3 functions (re-verify independently).
2. Verify the daily cycle does NOT depend on these wrappers (`grep -n 'run_analysis_pipeline\|evaluate_recent\|retrieve_memories' backend/services/autonomous_loop.py backend/api/paper_trading.py` should return 0 hits).
3. Decide CONDITIONAL or FAIL on the same calculus as 16.20.
4. Confirm `phase-16.2` masterplan status remains `in-progress` (must NOT be silently flipped by closing 16.21).

## References

- `handoff/current/phase-16.21-research-brief.md`
- `backend/tasks/analysis.py` -- has `run_analysis_task` (different signature) at line 35
- `backend/services/outcome_tracker.py` -- has `OutcomeTracker` class
- `backend/agents/memory.py` -- has `FinancialSituationMemory` class
- `handoff/archive/phase-16.20/` -- precedent for CONDITIONAL closure on missing-function pattern
