---
step: phase-16.26
title: Implement 3 wrapper shims for analysis + outcome + memory (closes 16.21)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.26

## Research-gate summary

`handoff/current/phase-16.26-research-brief.md`. tier=simple, 6 in-full, 16 URLs, recency scan, gate_passed=true.

## Key research findings

1. **All 3 wrapper functions are absent**. Confirmed by grep.
2. **`AnalysisOrchestrator`** lives at `backend/agents/orchestrator.py:302`. Entry: `async run_full_analysis(ticker)` line 989. Returns `report` dict.
3. **`final_score` lives at `report["final_synthesis"]["final_weighted_score"]`** — wrapper must flatten to top-level `final_score`.
4. **`OutcomeTracker(settings).evaluate_all_pending()`** returns `list[dict]`, `[]` on empty portfolio. Clean.
5. **`FinancialSituationMemory("default")`** loads 5 seed archetypes at `__init__` — tech query matches seed at memory.py:31.
6. **Vertex AI quota fix (commit f2e8ce28) is applied.** SDK has 4x auto-retry. Pipeline should run cleanly.
7. **`run_analysis_pipeline('AAPL')` takes 3-8 min** (13 async steps, Vertex AI calls). Long-running verification.

## Hypothesis

3 wrapper shims (~10-15 lines each) added to their respective modules. Verification command runs all 3 in sequence, all PASS.

## Success Criteria (verbatim, immutable)

```
source .venv/bin/activate && python3 -c "from backend.tasks.analysis import run_analysis_pipeline; r = run_analysis_pipeline('AAPL', run_id='uat-16.26'); assert r and r.get('final_score') is not None; print('ok')" && python3 -c "from backend.services.outcome_tracker import evaluate_recent; print(evaluate_recent(limit=5))" && python3 -c "from backend.agents.memory import retrieve_memories; ms = retrieve_memories('tech sector momentum 2025'); print(f'memories: {len(ms)}')"
```

- analysis_pipeline_returns_final_score
- outcome_tracker_returns_or_explains_empty
- bm25_retrieve_returns_at_least_1_memory_or_empty_explained

## Plan steps

1. Add `run_analysis_pipeline(ticker, run_id)` to `backend/tasks/analysis.py` end (~17 lines)
2. Add `evaluate_recent(limit=20)` to `backend/services/outcome_tracker.py` end (~12 lines)
3. Add `retrieve_memories(query, n_matches=5)` to `backend/agents/memory.py` end (~8 lines)
4. Run verification (allow 8-10 min for the analysis pipeline)
5. Spawn Q/A

## What Q/A must audit

1. All 3 wrappers exist at module level
2. `run_analysis_pipeline` returns dict with `final_score` key
3. `evaluate_recent` returns list (or safe dict on BQ failure)
4. `retrieve_memories` returns ≥1 memory for tech query
5. AST clean all 3 files
6. No regression on `run_analysis_task` (Celery task) or `OutcomeTracker` class or `FinancialSituationMemory` class
7. Pytest no regression
