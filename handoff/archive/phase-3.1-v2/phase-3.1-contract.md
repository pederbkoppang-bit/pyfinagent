# Sprint Contract -- phase-3.1 LLM-as-Planner + phase-3.2 LLM-as-Evaluator (joint closure)

**Written:** 2026-04-19 PRE-commit.
**Step ids:** `3.1` + `3.2` in phase-3 (closed together per research recommendation -- both agents exist and the integration gap IS the work).
**Parallel-safety:** phase-specific filenames.

## Research-gate summary

Researcher spawned today. Envelope: `{tier: moderate, external_sources_read_in_full: 5, snippet_only_sources: 6, urls_collected: 11, recency_scan_performed: true, internal_files_inspected: 9, gate_passed: true}`. Brief: `handoff/current/phase-3.1-research-brief.md` (171 lines; 4 recency findings).

Researcher's staked recommendation (adopted):
> "Phase 3.1 is approximately 70% done -- the planner and evaluator exist and are functional -- but the harness cycle was never formally closed because the integration is missing. The gap is not more code in the agent files: it is that `autonomous_loop.py`'s `_plan_phase()` feeds hardcoded mock data instead of real `quant_results.tsv` history, and its `_evaluate_phase()` bypasses the 5-rubric `EvaluatorAgent` with a 2-line Sharpe check. Wire `autonomous_loop.py` to real data and promote it as the canonical LLM planner entry point, then close phase-3.1 and phase-3.2 together."

Adopted as contract scope.

## Hypothesis

Replacing `autonomous_loop.py`'s mock-dict + bypassed-evaluator with real TSV/JSON reads + a real `EvaluatorAgent.evaluate_proposal()` call closes the integration gap flagged by research finding #5-7 and makes the Plan->Generate->Evaluate loop actually use the phase-3.1/3.2 agents. Adding unit tests closes the zero-tests gap (finding #8).

## Success criteria

Both masterplan steps have `verification: null` + `harness_required: False`. Defining in-contract.

**Functional (code wiring):**
1. `backend/autonomous_loop.py:253-276` mock dict replaced with a read of `backend/backtest/experiments/optimizer_best.json` (current best params + sharpe + dsr) AND a tail of `backend/backtest/experiments/quant_results.tsv` (recent kept/discarded rows). Read paths are via `pathlib.Path`; fail-open if files missing (fall back to mocks with WARNING log).
2. `backend/autonomous_loop.py:354-358` 2-line Sharpe check replaced with a real call to `EvaluatorAgent.evaluate_proposal(proposal, backtest_results, history)`. The call MUST preserve the existing return contract `(verdict, sharpe_delta)` by extracting `result.verdict` + computing the delta. Fail-open: if evaluator raises / times out, fall back to the 2-line check with WARNING.
3. New helper `_load_real_context(self) -> tuple[list[dict], dict]` in `AutonomousLoopOrchestrator` implementing the file reads from (1). Returns `(recent_results, current_params)` in the same shape the existing mock produces so downstream code changes are minimal.
4. `_plan_phase()` updated to call `self._load_real_context()` instead of hardcoding. Mock fallback retained for fully-missing-files dev environments.

**Functional (tests):**
5. `backend/tests/test_planner_agent.py` (new, >=4 tests): `PlannerAgent` instantiates without API key; `generate_proposal()` monkeypatched to return stable JSON and the method is exercised end-to-end without real API call; `reflect_on_feedback()` round-trips a feedback dict; JSON parsing handles both valid and malformed Claude responses.
6. `backend/tests/test_evaluator_agent.py` (new, >=4 tests): `EvaluatorAgent` instantiates without Vertex (`model=None` fallback); `_parse_evaluation_response` deterministically maps a known JSON to an `EvaluationResult` with correct verdict enum; `evaluate_proposal()` timeout path returns FAIL fail-open; mock evaluator path in `_call_model` is exercised.
7. `backend/tests/test_autonomous_loop_integration.py` (new, >=2 tests): `AutonomousLoopOrchestrator._load_real_context()` returns expected shape when files exist; falls back cleanly when files missing.

**Functional (docs):**
8. `docs/PHASE_3_LLM_PLANNER.md` (new) with: implementation inventory with source-verified file:line anchors (planner_agent, evaluator_agent, autonomous_loop, planner_enhanced as forward-work); integration state after this cycle (autonomous_loop now reads real data + calls real evaluator); known gaps with phase ownership (planner_enhanced consolidation -> phase-3.3; BacktestEngine-in-loop -> phase-3.3; production-harness wiring -> phase-3.3; true multi-agent parallel -> phase-3.4); cross-links to the 4 external sources read in full (Anthropic multi-agent, arXiv 2409.06289, arXiv 2412.20138, arXiv 2602.23330, Karpathy autoresearch).

**Functional (status):**
9. Masterplan `phase-3.1` status: `pending` -> `done`; `completed_at` set.
10. Masterplan `phase-3.2` status: `pending` -> `done`; `completed_at` set. `verification` + `contract` fields preserved (currently null on 3.2).

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('backend/autonomous_loop.py').read())"` and for each new test file -> exit 0
- Import smokes: `python -c "from backend.agents.planner_agent import PlannerAgent; from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationResult, EvaluationVerdict; from backend.autonomous_loop import AutonomousLoopOrchestrator; print('ok')"` -> `ok`
- Unit tests: `pytest backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py -q` -> all pass
- Zero regressions on prior phase-6 tests: `pytest backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py -q` -> same 41p/1s as before
- `_load_real_context` smoke: `python -c "from backend.autonomous_loop import AutonomousLoopOrchestrator as AL; a = AL(max_iterations=1); r, p = a._load_real_context(); print('recent:', len(r), 'param_keys:', len(p))"` -> prints plausible numbers

**Non-goals:**
- NOT wiring `PlannerAgent` into `scripts/harness/run_harness.py` production path (phase-3.3).
- NOT unifying `planner_agent.py` + `planner_enhanced.py` (phase-3.3).
- NOT replacing the mock backtest results in `_generate_phase()` (phase-3.3 -- requires BacktestEngine wiring which touches a much larger surface).
- NOT adding the Information Coefficient (IC) metric from arXiv 2409.06289 to EvaluatorAgent (future phase).

## Plan steps

1. Read `backend/agents/planner_agent.py` + `backend/agents/evaluator_agent.py` in full so tests match real method signatures.
2. Replace mock dict at `autonomous_loop.py:253-276` with `_load_real_context()` call; implement helper using `optimizer_best.json` + `quant_results.tsv` tail.
3. Replace 2-line check at `:354-358` with `await EvaluatorAgent.evaluate_proposal(...)` + fail-open wrap.
4. Write 3 new test files.
5. Write `docs/PHASE_3_LLM_PLANNER.md`.
6. Run all verification commands.

## References

- `handoff/current/phase-3.1-research-brief.md` (171 lines)
- `handoff/archive/phase-3.1/contract.md` (original Apr-3 plan)
- `backend/agents/planner_agent.py:43-151`
- `backend/agents/evaluator_agent.py:78-174` (`evaluate_proposal` + timeout fallback)
- `backend/autonomous_loop.py:231-359` (Plan+Eval phase bodies)
- `backend/backtest/experiments/optimizer_best.json` (current best context)
- `backend/backtest/experiments/quant_results.tsv` (experiment history)

## Researcher agent id

`a42e3712e48a8c44a`
