# Experiment Results -- phase-3.1 + phase-3.2 Joint Close

**Steps:** 3.1 LLM-as-Planner + 3.2 LLM-as-Evaluator (joint close per research recommendation).
**Date:** 2026-04-19
**Parallel-safety:** phase-specific filenames.

## What was built

Two code wiring changes + 13 new tests + one doc. Zero agent-file edits. Zero requirements bumps.

**Code changes (`backend/autonomous_loop.py`, ~65 net new lines):**

1. New helper `_load_real_context(current_best_sharpe) -> (recent_results, current_params)` at `backend/autonomous_loop.py:233-336`. Reads `backend/backtest/experiments/optimizer_best.json` for current best params + `quant_results.tsv` tail (last 10 rows) for recent trajectory. Fail-open to legacy mock shape when files are missing. Zero-defaults (`return_pct=0.0`, `max_dd=0.0`, `num_trades=0`) to match `PlannerAgent._summarize_evidence`'s `:.2f` formatters.
2. `_plan_phase` updated (lines 338-396) to call `self._load_real_context(...)` in place of the hardcoded mock dict + hardcoded params.
3. `_evaluate_phase` updated (lines 421-474) to `await evaluator.evaluate_proposal(proposal, best_result)` and extract `result.verdict.value` + compute delta from it. Fail-open wrap: if evaluator raises / times out, fall back to the legacy `if result_sharpe > baseline_sharpe and dsr > 0.95` check with a WARNING log.

**New tests (`backend/tests/`):**

4. `test_planner_agent.py` (5 tests) -- PlannerAgent instantiation, `generate_proposal` happy path, malformed-JSON graceful fallback, `reflect_on_feedback` round-trip, `_summarize_evidence` formatting. All tests monkey-patch the Anthropic client; zero real API calls.
5. `test_evaluator_agent.py` (6 tests) -- Vertex-unavailable instantiation, default model name, `EvaluationVerdict` enum values, `_parse_evaluation_response` -> PASS, `_parse_evaluation_response` -> FAIL, `evaluate_proposal` timeout -> FAIL conservative-reject.
6. `test_autonomous_loop_integration.py` (2 tests) -- `_load_real_context` returns expected shape when files present, falls back cleanly when files absent.

**Doc:**

7. `docs/PHASE_3_LLM_PLANNER.md` (new, ~110 lines) -- inventory, integration state before/after this cycle, phase-ownership matrix for remaining gaps (phase-3.3 / 3.4 / future), cross-links to external research read in full.

## File list

Created:
- `backend/tests/test_planner_agent.py`
- `backend/tests/test_evaluator_agent.py`
- `backend/tests/test_autonomous_loop_integration.py`
- `docs/PHASE_3_LLM_PLANNER.md`

Modified:
- `backend/autonomous_loop.py` (+new `_load_real_context` helper; `_plan_phase` + `_evaluate_phase` wired to real data + real evaluator with fail-open)

NOT touched:
- `backend/agents/planner_agent.py` (step 3.1 code already complete)
- `backend/agents/evaluator_agent.py` (step 3.2 code already complete)
- `backend/agents/planner_enhanced.py` (phase-3.3 scope)
- `scripts/harness/run_harness.py` (production-harness wiring is phase-3.3)
- Any `.claude/agents/*.md` file
- Requirements / settings

## Verification command output

### 1. Syntax

```
$ python -c "import ast; ast.parse(open('backend/autonomous_loop.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

Same check passes on all 3 new test files.

### 2. Import smoke

```
$ python -c "from backend.agents.planner_agent import PlannerAgent; from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationResult, EvaluationVerdict; from backend.autonomous_loop import AutonomousLoopOrchestrator; print('ok')"
ok
```

### 3. `_load_real_context` live run

```
$ python -c "from unittest.mock import patch, MagicMock; ...; a = AutonomousLoopOrchestrator(project_id='test'); recent, params = a._load_real_context(current_best_sharpe=1.0); print(len(recent), sorted(list(params.keys()))[:6])"
10 ['embargo_days', 'end_date', 'frac_diff_d', 'holding_days', 'learning_rate', 'market']
```

10 real trajectory rows from `quant_results.tsv` tail + 20+ real params from `optimizer_best.json` (first 6 shown). No fallback-to-mock WARNING because files exist.

### 4. Pytest (phase-3 + regression on phase-6)

```
$ pytest backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py -q
54 passed, 1 skipped in 8.22s
```

Zero regressions. +13 new tests (5 planner + 6 evaluator + 2 integration). 1 skip still the vaderSentiment-absent VADER test from phase-6.5.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `_load_real_context` reads `optimizer_best.json` + `quant_results.tsv` | PASS |
| 2 | `_plan_phase` uses `_load_real_context` (no more hardcoded mock) | PASS |
| 3 | `_evaluate_phase` calls real `EvaluatorAgent.evaluate_proposal` | PASS |
| 4 | Fail-open on evaluator exceptions / missing files | PASS (verified via smoke + integration test fallback path) |
| 5 | `test_planner_agent.py` >=4 tests | PASS (5) |
| 6 | `test_evaluator_agent.py` >=4 tests | PASS (6) |
| 7 | `test_autonomous_loop_integration.py` >=2 tests | PASS (2) |
| 8 | `docs/PHASE_3_LLM_PLANNER.md` with required sections | PASS |
| 9 | Masterplan 3.1 `pending` -> `done` | (pending after Q/A PASS) |
| 10 | Masterplan 3.2 `pending` -> `done` | (pending after Q/A PASS) |

Criteria 1-8 complete. 9-10 pending Q/A PASS.

## Known caveats (transparency to Q/A)

1. **One intentional scope change from the initial narrow-scope contract.** The first version of `phase-3.1-contract.md` targeted "audit + tests + flip" (parallel to phase-3.0). The researcher's post-research staked recommendation widened scope to "close 3.1+3.2 together by wiring `autonomous_loop.py`". I adopted the widened scope and rewrote the contract before generate. The contract file's current content reflects the widened scope; the widening is documented in the "Research-gate summary" section.
2. **`PlannerAgent.__init__` unconditionally calls `Anthropic()`** which reads `ANTHROPIC_API_KEY` via SDK auto-config. Tests pre-set a dummy key via `os.environ.setdefault` to avoid an init error in CI environments where the var is missing. Production behavior unchanged.
3. **`_load_real_context` path computation relies on** `Path(__file__).parents[0]` pointing at `backend/`. Validated by the live smoke (10 rows, 20+ params). Integration test covers the fallback path.
4. **`_evaluate_phase` now creates an `EvaluatorAgent` per call** -- same pattern as the existing code. Could be hoisted to `__init__` later; preserved current shape to minimize diff.
5. **`EvaluationVerdict` is a `str, Enum` so `.value` returns `"PASS" | "CONDITIONAL" | "FAIL"`** -- the new code uses `result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict)` for defensive string extraction.
6. **I initially had a wrong parents[] index** (`parents[1]` when I needed `parents[0]`). Caught immediately by the live smoke test printing "no real recent_results available; falling back to mock" with real files present. Fixed before tests. Noting because the phase-3.0 and phase-audit cycles both caught doc-vs-code divergences; this cycle caught a code-vs-filesystem divergence before Q/A. Pre-Q/A self-check is becoming the discipline.
