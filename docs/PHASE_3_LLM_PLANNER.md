# Phase 3 LLM-Guided Research: Planner + Evaluator

**Phases closed by this document:** 3.1 (LLM-as-Planner) + 3.2 (LLM-as-Evaluator).
**Phase lineage:** 3.1 contract (2026-04-03, `handoff/archive/phase-3.1/`) -> dormant code -> 3.1 joint close with 3.2 (2026-04-19, this doc).

## Implementation inventory

### Planner (`backend/agents/planner_agent.py`)

- `PlannerAgent.__init__(model="claude-opus-4-6")` -- thin wrapper over `anthropic.Anthropic()`.
- `PlannerAgent.generate_proposal(recent_results, current_best_sharpe, current_params, weaknesses=None) -> dict` (`backend/agents/planner_agent.py:43-151`): builds `{"proposals": [...], "reasoning": ..., "meta_plan_alignment": ...}` by feeding a `META_PLAN` system prompt + evidence summary to Claude and parsing the returned JSON. Malformed-JSON fallback returns an empty-proposals dict rather than raising.
- `PlannerAgent.reflect_on_feedback(proposal, feedback) -> dict` (`:181-...`): second-pass refinement given a prior proposal + its evaluation outcome.
- `_summarize_evidence` (`:153-179`): formats up to 5 recent backtest rows + optional weaknesses string into a prose context block.

### Evaluator (`backend/agents/evaluator_agent.py`)

- `EvaluatorAgent.__init__(model_name="gemini-2.0-flash")` (`:86-102`): instantiates a Vertex `GenerativeModel`; fail-open to `None` if Vertex is unavailable (tests rely on this).
- `EvaluatorAgent.evaluate_proposal(proposal, backtest_results, history=None) -> EvaluationResult` (`:104-174`): 30-second deadline, timeout returns FAIL conservative-reject with a red-flag `"Evaluation timed out -- cannot verify safety"`.
- 5-rubric `EvaluationResult` dataclass (`:55-75`): `statistical_validity_score`, `robustness_score`, `simplicity_score`, `reality_gap_score`, `risk_check_score`, `overall_score` (average) + red/yellow/green flag lists.
- `EvaluationVerdict` enum: `PASS | CONDITIONAL | FAIL` (`:48-52`).
- `_parse_evaluation_response(response_text, proposal, backtest_results) -> EvaluationResult` (`:400+`): deterministic mapping from Claude/Gemini JSON to the dataclass.

### Loop orchestration (`backend/autonomous_loop.py`)

- `AutonomousLoopOrchestrator.run_loop()` (`:97-229`): up to `max_iterations` (default 10) cycles of PLAN -> GENERATE -> EVALUATE -> DECIDE -> LEARN, stopping on `target_sharpe` (1.23) or budget.
- `_plan_phase()` (`:338-396` after phase-3.1 edits): calls `PlannerAgent.generate_proposal` on real evidence.
- `_evaluate_phase()` (`:421-474` after phase-3.1 edits): calls `EvaluatorAgent.evaluate_proposal` (real 5-rubric); fail-open to legacy Sharpe+DSR gate if evaluator raises.
- `_load_real_context(current_best_sharpe)` (NEW, `:233-336`): reads `backend/backtest/experiments/optimizer_best.json` (current best params) + tails `quant_results.tsv` (last 10 rows). Fail-open to legacy mock-shape on missing files.

### Forward work (out of scope for phase-3.1/3.2 close)

- `backend/agents/planner_enhanced.py` (336 lines) -- `EnhancedPlannerAgent` with regime conditioning + RESEARCH.md reading. **Phase-3.3 scope.**
- `_generate_phase` in `autonomous_loop.py:288-312` still returns `_get_mock_backtest_results(...)` -- **phase-3.3 scope** (requires BacktestEngine integration, non-trivial).
- `scripts/harness/run_harness.py:149` production harness still uses a rule-based planner. **Phase-3.3 scope.**
- IC (Information Coefficient) metric from arXiv 2409.06289 -- not yet in `EvaluatorAgent`. Future phase.
- Unified `PlannerAgent` + `EnhancedPlannerAgent` entry point -- phase-3.3 consolidation.

## Integration state (after phase-3.1 close)

Before this cycle:
```
_plan_phase() -> PlannerAgent.generate_proposal(MOCK recent_results, MOCK current_params)
_evaluate_phase() -> 2-line check (sharpe > baseline AND dsr > 0.95)
```

After this cycle:
```
_plan_phase() -> _load_real_context() -> PlannerAgent.generate_proposal(REAL tsv tail, REAL optimizer_best.json params)
_evaluate_phase() -> await EvaluatorAgent.evaluate_proposal(proposal, backtest_result)
                  -> fail-open to legacy Sharpe+DSR gate on exception
```

The planner now sees real experiment trajectory (kept/discarded runs, per-row status + delta + DSR). The evaluator now applies the full 5-rubric scoring with red/yellow/green flags instead of a boolean gate.

## Gaps with explicit phase ownership

| Gap | Phase owner | Trigger |
|-----|------------|--------|
| `_generate_phase` returns mock backtests, not real `BacktestEngine.run()` | 3.3 | When phase-3.3 integrates the real backtest path |
| `run_harness.py:149` rule-based planner vs LLM planner | 3.3 | Promote `autonomous_loop.py` as canonical OR wire `PlannerAgent` into `run_harness.py` |
| `planner_agent.py` + `planner_enhanced.py` coexist with unclear canonical entry | 3.3 | Merge or delete the loser |
| IC metric in evaluator | Future | Comes with live data / actual IC-capable backtest harness |
| Parallel multi-agent planner (Anthropic's documented pattern) | 3.4 | Adoption decision after phase-3.3 wires the serial path |

## External research anchors

All fetched in full 2026-04-19 during the phase-3.1 research gate; envelope `gate_passed=true`:

- Anthropic -- [How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system): lead agent with extended thinking spawns parallel subagents; token budget drives 80% of performance variance.
- arXiv 2409.06289v3 (EMNLP 2025) -- "Automate Strategy Finding with LLM in Quant Investment": Seed Alpha Factory + multi-modal evaluator (IC + Sharpe).
- arXiv 2412.20138v3 (TradingAgents): structured document-exchange is the correct inter-agent protocol.
- arXiv 2602.23330v1 (Expert Investment Teams): fine-grained task decomposition beats abstract roles.
- Karpathy autoresearch (2026-03): propose-measure-keep/discard ratchet. 700 experiments / 48h / 11% improvement.

## Cross-references

- `handoff/current/phase-3.1-research-brief.md` -- research gate for this cycle.
- `handoff/current/phase-3.1-contract.md` -- joint 3.1+3.2 contract.
- `handoff/archive/phase-3.1/contract.md` -- original 2026-04-03 plan.
- `backend/tests/test_planner_agent.py` -- unit tests (5 passing).
- `backend/tests/test_evaluator_agent.py` -- unit tests (6 passing).
- `backend/tests/test_autonomous_loop_integration.py` -- integration tests (2 passing).
- `ARCHITECTURE.md` -- pipeline overview (layer 3 is this).
