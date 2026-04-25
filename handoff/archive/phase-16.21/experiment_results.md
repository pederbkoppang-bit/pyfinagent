---
step: phase-16.21
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: CONDITIONAL
---

# Experiment Results -- phase-16.21

## What was done

Ran the immutable verification command verbatim (chain of 3 imports + assertions). Captured failures. Verified underlying classes exist. No code changes.

### Files touched
- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this)
- `handoff/current/phase-16.21-research-brief.md`

## Verification (verbatim)

### Probe 1: `run_analysis_pipeline`
```
$ python3 -c "from backend.tasks.analysis import run_analysis_pipeline"
ImportError: cannot import name 'run_analysis_pipeline' from 'backend.tasks.analysis'
```

### Probe 2: `evaluate_recent`
```
$ python3 -c "from backend.services.outcome_tracker import evaluate_recent"
ImportError: cannot import name 'evaluate_recent' from 'backend.services.outcome_tracker'
```

### Probe 3: `retrieve_memories`
```
$ python3 -c "from backend.agents.memory import retrieve_memories"
ImportError: cannot import name 'retrieve_memories' from 'backend.agents.memory'
```

**Exit codes: 1, 1, 1.** None of the three target functions exist in the modules.

## Bonus probe: underlying classes (Monday-readiness check)

Researcher's claim that wrappers are 5-10 trivial lines was wrong. Actual state:

| Module | Has | Issues |
|--------|-----|--------|
| `backend/tasks/analysis.py` | `run_analysis_task(self, ticker)` at line 35 (Celery-style task with `self` arg) | NO `AnalysisOrchestrator` class here either |
| `backend/services/outcome_tracker.py` | `OutcomeTracker` class at line 28 | `__init__` requires `settings` positional arg |
| `backend/agents/memory.py` | `FinancialSituationMemory` class at line 57 + `build_situation_description` + `generate_reflection` | `__init__` requires `name` positional arg |

The underlying functionality is present but the test-shim wrappers are missing.

## Daily-cycle dependency check (Monday-readiness)

```
$ grep -c 'run_analysis_pipeline\|evaluate_recent\|retrieve_memories' \
    backend/services/autonomous_loop.py \
    backend/api/paper_trading.py
backend/api/paper_trading.py:0
backend/services/autonomous_loop.py:0
```

**Zero references.** The daily paper-trading cycle does NOT depend on these missing wrappers. Monday is NOT blocked by this gap.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | analysis_pipeline_returns_final_score | FAIL | ImportError before call |
| 2 | outcome_tracker_returns_or_explains_empty | FAIL | ImportError before call |
| 3 | bm25_retrieve_returns_at_least_1_memory_or_empty_explained | FAIL | ImportError before call |

Mechanically: 0 of 3. Same disposition as 16.20: CONDITIONAL with follow-ups, since the underlying functionality is intact and Monday is not blocked.

## Honest disclosures

1. **Same masterplan-aspirational-command pattern as 16.20.** Verification command was written assuming these wrappers exist; they don't. This is a documentation/code-state mismatch the harness exists to surface.

2. **Researcher had inaccuracies.** Claimed `AnalysisOrchestrator` is in `backend/tasks/analysis.py` — it's not. Claimed wrappers would be "5-10 lines each" — they'd be more (need to handle init args, Vertex AI calls, BM25 loading). I did not act on the inaccurate suggestion to write the wrappers.

3. **Researcher's commit f2e8ce28 finding** (Vertex AI 429 quota fix) is potentially relevant for the underlying pipeline's reliability, but I did NOT exercise the live pipeline in this cycle (out of scope; that's the missing wrapper's job).

4. **`phase-16.2`** in the masterplan currently `in-progress` (its blocker was the same Vertex 429 issue). 16.21 was supposed to close 16.2. Per Q/A's 16.20 precedent, **16.2 must stay in-progress** until the wrapper is implemented + the live pipeline runs cleanly + a fresh Q/A returns PASS. Q/A please confirm this condition.

5. **No code changes this cycle.**

## Follow-up tickets to file (if Q/A accepts CONDITIONAL)

1. Implement `run_analysis_pipeline(ticker, run_id)` in `backend/tasks/analysis.py` — sync wrapper around the actual analysis-orchestrator path (which lives in `backend/agents/orchestrator.py`, not `tasks/analysis.py`).
2. Implement `evaluate_recent(limit=5)` in `backend/services/outcome_tracker.py` — module-level wrapper that constructs `OutcomeTracker(settings)` and calls `evaluate_all_pending()`.
3. Implement `retrieve_memories(query)` in `backend/agents/memory.py` — module-level wrapper that constructs `FinancialSituationMemory(name="default")` and calls `get_memories(query)`.
4. After all 3 land, RE-RUN 16.21's verification command (now will exercise live Vertex AI pipeline). 16.2 closes simultaneously when fresh Q/A returns PASS.

## Next

Spawn Q/A. Same judgment-call ask as 16.20.
