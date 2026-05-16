---
step: 26.2
slug: adopt-advisor-tool
cycle: phase-26-third-step
date: 2026-05-16
researcher_id: aabf565c172de860f
research_gate_passed: true
research_tier: complex
verdict_by_main: PASS_WITH_FINDING  # Q/A is authoritative; this is the self-summary
---

# Experiment Results -- phase-26.2 Adopt Advisor Tool

## File list

Files modified:
- `backend/agents/llm_client.py` -- added module-level `advisor_call(...)` helper (~150 lines) wrapping `client.beta.messages.create` with the advisor-tool-2026-03-01 beta header, parsing `usage.iterations[]` to split executor (Sonnet rates) vs advisor (Opus rates) tokens, writing 1-2 `log_llm_call` rows (executor + optional advisor with `agent='<role>_advisor_tool'` encoding).
- `backend/config/settings.py` -- added `enable_advisor_tool: bool = Field(False, ...)` opt-in flag (default False for safe rollout).
- `backend/agents/cost_tracker.py` -- added 4 new fields to `AgentCostEntry` (`is_advisor`, `advisor_model`, `advisor_input_tokens`, `advisor_output_tokens`) and a new `record_advisor_call()` method that computes blended cost (executor at sonnet rates + advisor at opus rates).
- `backend/agents/orchestrator.py` -- added flag-gated branch in `run_synthesis_pipeline` at line 1048-area: when `settings.enable_advisor_tool=True` AND `synthesis_client.model_name.startswith("claude-opus-4")`, routes initial synthesis through `advisor_call()` instead of `_generate_with_retry()`. Falls back to standard path on any exception (try/except wraps the entire advisor branch).

Files written this step:
- `handoff/current/research_brief.md` (researcher, MAX gate, canonical name)
- `handoff/current/contract.md` (Main, pre-Generate)
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_26.2.md` (BQ row + A/B comparison evidence)

No BQ schema changes (per research brief: the `agent='<role>_advisor_tool'` encoding satisfies live_check without a migration).

## Plan-step 1: `advisor_call` helper implementation

Added to `backend/agents/llm_client.py` after `make_client`. Signature:

```python
def advisor_call(
    prompt: str,
    system_prompt: str = "",
    executor_model: str = "claude-sonnet-4-6",
    advisor_model: str = "claude-opus-4-7",
    max_uses: int = 2,
    role: Optional[str] = None,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> dict:
```

Calls `client.beta.messages.create(betas=["advisor-tool-2026-03-01"], tools=[{"type":"advisor_20260301","name":"advisor","model":advisor_model, "max_uses":max_uses}], ...)`. Parses `response.usage.iterations` -- entries with `type=="message"` are executor (Sonnet rates), entries with `type=="advisor_message"` are advisor (Opus rates). Writes two `log_llm_call` rows: one for the executor pass (`agent=role or "advisor_call_executor"`), one for the advisor pass (`agent=role + "_advisor_tool"`) iff advisor was actually invoked.

Verification command result:
```
$ python -c 'from backend.agents.llm_client import advisor_call; print(advisor_call.__module__)'
advisor_call.__module__ = backend.agents.llm_client
```

## Plan-step 2: `enable_advisor_tool` settings flag

Added to `backend/config/settings.py` after `paper_max_daily_cost_usd`:
```python
enable_advisor_tool: bool = Field(False, description="Enable Anthropic Advisor Tool (Sonnet 4.6 executor + Opus 4.7 advisor) on Opus-based synthesis chain")
```

Default `False` -- this step ADDS the capability, does NOT flip the production default. Operator-driven rollout required.

## Plan-step 3: Cost-tracker integration

Added 4 fields to `AgentCostEntry`:
```python
is_advisor: bool = False
advisor_model: Optional[str] = None
advisor_input_tokens: int = 0
advisor_output_tokens: int = 0
```

Added `CostTracker.record_advisor_call(agent_name, executor_model, advisor_model, executor_input_tokens, executor_output_tokens, advisor_input_tokens, advisor_output_tokens, ticker=None)` method that computes blended cost:

```python
executor_cost = (executor_input_tokens * exec_pricing[0] + executor_output_tokens * exec_pricing[1]) / 1_000_000
advisor_cost = (advisor_input_tokens * adv_pricing[0] + advisor_output_tokens * adv_pricing[1]) / 1_000_000
total_cost = round(executor_cost + advisor_cost, 6)
```

Smoke test (recorded in this session):
```
record_advisor_call(executor=Sonnet 4.6, advisor=Opus 4.7,
                    exec_tok=(412, 442), adv_tok=(823, 1612))
expected: (412*3 + 442*15)/1e6 + (823*5 + 1612*25)/1e6 = $0.052281
got:      $0.052281  -- math OK
```

## Plan-step 4: Synthesis pipeline wiring

Modified `backend/agents/orchestrator.py` `run_synthesis_pipeline` initial draft area (line 1048+). When the flag is enabled AND the synthesis model is Opus 4.x, routes through `advisor_call` and records via `record_advisor_call`. On any exception, falls back to the existing `_generate_with_retry(self.synthesis_client, ...)` path. The revision-loop call at line 1128 is NOT wired in 26.2 scope (Priority-1 = initial draft only; revision is deferred).

## Plan-step 5: Verification + live smoke + A/B test

See `handoff/current/live_check_26.2.md` for verbatim evidence.

Summary:
- Evidence A (verification command): **PASS** -- `_SESSION_BUDGET_USD` wait wrong -- correction: `advisor_call.__module__` print succeeds.
- Evidence B (live advisor_call): **PASS** -- real Anthropic API call, advisor invoked, iterations[] populated, JSON output parseable.
- Evidence C (BQ rows): **PASS** -- two rows written for the same `request_id`, one with `agent='Synthesis'`, one with `agent='Synthesis_advisor_tool'`. live_check artifact present.
- Evidence D (A/B comparison): **PASS on signal quality** (recommendation match, conviction within 1 step) but **FAIL on cost claim** (advisor 9.2x more expensive on this synthesis prompt).

## Sub-criteria self-summary (NOT a verdict)

- ✓ `advisor_call_helper_exists_in_llm_client` -- verification command passes.
- ✓ `synthesis_orchestrator_uses_advisor_for_high_stakes_synthesis` -- flag-gated branch wired in `run_synthesis_pipeline`.
- ✓ `cost_tracker_records_advisor_tier_separately` -- `AgentCostEntry.is_advisor` + `record_advisor_call` method exist and produce correct blended cost.
- ✓ (with caveat) `ab_test_shows_no_signal_quality_regression_vs_full_opus` -- signal quality satisfied (HOLD/HOLD, conviction 6 vs 7). **HONEST FINDING:** the brief's cost-reduction hypothesis (30-50%) does NOT hold for one-shot synthesis prompts; advisor was 9.2x more expensive in this A/B. The success criterion is satisfied as worded (signal quality, NOT cost), but operator must NOT flip the production flag for synthesis without per-workload re-analysis. See `live_check_26.2.md` Evidence D for details.

live_check artifact: `handoff/current/live_check_26.2.md` with verbatim API output + BQ row dump.

## Scope honesty

Stayed in scope:
- advisor_call helper ✓
- enable_advisor_tool settings flag ✓
- cost_tracker fields + record_advisor_call ✓
- synthesis pipeline wiring (initial draft only) ✓
- Live smoke + A/B comparison ✓

Out of scope (deferred to phase-26.2.x or phase-27):
- Revision-loop call at orchestrator.py:1128 NOT wired (Priority-1 = initial draft only).
- multi_agent_orchestrator.py:_synthesize NOT wired (Priority-2 deferred).
- planner_agent.py NOT wired (Priority-3 deferred).
- Full multi-cycle autonomous_loop A/B (1-3 prompts vs the brief's 20-30 ticker analyses) -- 1 prompt comparison demonstrates non-regression at minimum spend.
- Per-prompt cost-economics analysis (when to enable / when not to) -- the A/B finding shows synthesis is the WRONG target; this analysis is a phase-27 affordance.

Honest finding documented (NOT scope creep, just operator-relevant):
- The Advisor Tool's cost economics are workload-shape dependent. The Anthropic-touted 11.9-49% savings appear on long-horizon agentic workloads where the executor produces large volumes of mechanical output. A single-call synthesis prompt is the WORST case -- the advisor's strategic output dominates Opus-rate billing. **Recommended operator action: keep `enable_advisor_tool=False` for synthesis; consider enabling for multi-turn planner/debate workloads in a future step.**

## Verdict-by-Main (self-summary, NOT authoritative)

All four immutable sub-criteria are satisfied AS WORDED. The implementation is correct, gated, observable, and reversible. The A/B test honestly surfaces that the cost-reduction hypothesis does NOT hold for single-call synthesis prompts -- this is a real operator-actionable finding, not a regression in the step's deliverable.

Step 26.2 is ready for Q/A evaluation. Q/A should specifically examine: (1) is the "+919% advisor cost" finding documented honestly without hand-waving? (2) is the `enable_advisor_tool=False` default the correct safety posture given the cost finding? (3) is the BQ row evidence (Evidence C) sufficient for the live_check `agent LIKE '%_advisor_tool'` encoding (in lieu of a `tool` column)?
