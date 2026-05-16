# Sprint Contract -- phase-26.2
Step: Adopt Advisor Tool (Sonnet executor + Opus advisor) on synthesis chain

## Research Gate
researcher_aabf565c172de860f (tier=complex, MAX gate per user instruction 2026-05-16) gate_passed=true.
Brief: `handoff/current/research_brief.md` (canonical filename; archive captures correctly on close).
- 7 unique external URLs read in full via WebFetch (3 Tier-1: Anthropic Advisor Tool doc, Anthropic release notes, ICML 2025 OpenReview AAl89VNNy1 cascade-routing paper; 4 Tier-2: arXiv 2405.15842 model-cascading-for-code, builder.io advisor-pattern post, MindStudio Anthropic-advisor-strategy post, testingcatalog launch coverage). 4 snippet-only; 11 URLs total. 3-variant search discipline applied (current-year, last-2-year, year-less canonical).
- Recency scan (2024-04 -> 2026-05) performed: Advisor Tool launched 2026-04-09 (beta header `advisor-tool-2026-03-01`); Opus 4.7 GA 2026-04-16 as the only valid advisor model; model-cascading literature May 2025 validates 49% cost savings best-case + 26% average; ICML 2025 cascade-routing proves up to 14% routing-strategy improvement. No contradicting evidence found.
- Internal grep at file:line covered all 6 required modules. Highest-leverage adoption point: `orchestrator.py::run_synthesis_pipeline` (line 961-1140), specifically the synthesis client constructed at line 363-424 (`make_client(deep_model_name, ...)`).
- Cross-check finding: Advisor Tool is **client.beta.messages.create** ONLY -- NOT supported on `client.messages.create`. Beta header `advisor-tool-2026-03-01` + tools `[{"type":"advisor_20260301","name":"advisor","model":advisor_model}]`. Response shape interleaves text + server_tool_use + advisor_tool_result + text blocks. `usage.iterations[]` separates executor tokens (Sonnet rates) from advisor tokens (Opus rates); the top-level `usage` reflects executor only.
- Key findings:
  1. Pairing table: Sonnet 4.6 -> Opus 4.7 is the natural pyfinagent pair (Sonnet already in `llm_client.py:546` alias map). HTTP 400 on invalid pairings.
  2. Benchmarks (Tier-1 + Tier-2): Sonnet+Opus-advisor on SWE-bench 74.8% accuracy vs Sonnet-solo 72.1%, 11.9% cheaper than Opus-solo. Anthropic's own numbers: 11% cost reduction + 2% quality improvement.
  3. NOT available on Bedrock or Vertex (Anthropic API + AWS only). pyfinagent uses Anthropic direct API -- compatible.
  4. `iterations[]` field ONLY appears when advisor is actually invoked. Empty/missing for non-advisor responses. Cost-tracking code must handle both cases.
  5. Recommended live_check encoding: `agent='<role>_advisor_tool'` in `log_llm_call`. NO schema migration required (agent column is already STRING).

## Hypothesis
Implementing `advisor_call(prompt, system_prompt, executor_model='claude-sonnet-4-6', advisor_model='claude-opus-4-7', max_uses, role, config)` in `llm_client.py` using `client.beta.messages.create(betas=['advisor-tool-2026-03-01'], tools=[{'type':'advisor_20260301', 'name':'advisor', 'model':advisor_model}], ...)` provides a drop-in cost-reduction primitive for the synthesis chain. Gated behind `enable_advisor_tool` settings flag (default False). When enabled and the configured synthesis model is `claude-opus-4-*`, `run_synthesis_pipeline` routes through `advisor_call` instead of `generate_content`, yielding 25-45% estimated cost reduction (per brief's analysis of typical synthesis call token shapes). The `usage.iterations[]` array is parsed to write two `log_llm_call` rows: one for the executor pass (`agent='<role>'`), one for the advisor pass (`agent='<role>_advisor_tool'`), enabling BQ live_check on the advisor row. A controlled comparison test on a representative synthesis prompt demonstrates non-regression on `conviction` and `recommendation` JSON fields.

## Success Criteria (immutable, copied verbatim from .claude/masterplan.json step 26.2)
```
source .venv/bin/activate && python -c 'from backend.agents.llm_client import advisor_call; print(advisor_call.__module__)'
```
Plus sub-criteria:
- `advisor_call_helper_exists_in_llm_client` -- satisfied by the import + print in the verification command.
- `synthesis_orchestrator_uses_advisor_for_high_stakes_synthesis` -- satisfied by adding a feature-gated branch in `orchestrator.py::run_synthesis_pipeline` that calls `advisor_call` when `settings.enable_advisor_tool == True` AND the configured synthesis model starts with `claude-opus-4-*`.
- `cost_tracker_records_advisor_tier_separately` -- satisfied by adding `is_advisor: bool`, `advisor_input_tokens: int`, `advisor_output_tokens: int` fields to `AgentCostEntry` and a companion `record_advisor_call()` method that computes blended cost = executor_tokens * sonnet_rate + advisor_tokens * opus_rate.
- `ab_test_shows_no_signal_quality_regression_vs_full_opus` -- satisfied by a controlled comparison on a representative synthesis-style prompt run through BOTH paths; comparison of `conviction` / `recommendation` JSON fields shows no regression (defined as: both fields present, value match within 1 step on conviction scale, recommendation token-similarity > 0.7).

live_check: `handoff/current/live_check_26.2.md` -- verbatim Python output of one `advisor_call` invocation showing the iterations[] split, plus the BQ row queried back showing `agent LIKE '%_advisor_tool'` and `provider='anthropic'`.

## Plan (PRE-commit; will NOT diverge in Generate)

1. **Implementation in `backend/agents/llm_client.py`:**
   - Add `advisor_call(prompt: str, system_prompt: str = "", executor_model: str = "claude-sonnet-4-6", advisor_model: str = "claude-opus-4-7", max_uses: int = 2, role: Optional[str] = None, max_tokens: int = 4096) -> dict` function near the existing call helpers.
   - Construct the Anthropic client lazily (inherit existing pattern from `_check_cost_budget`-adjacent helpers).
   - Invoke `client.beta.messages.create(model=executor_model, max_tokens=max_tokens, system=system_prompt, messages=[{"role":"user","content":prompt}], betas=["advisor-tool-2026-03-01"], tools=[{"type":"advisor_20260301","name":"advisor","model":advisor_model}])`.
   - Parse the response: extract `response.content` text blocks (excluding server_tool_use + advisor_tool_result); extract `response.usage.iterations` if present; classify each iteration as executor (`type=="message"`) vs advisor (`type=="advisor_message"`).
   - Write two `log_llm_call` rows:
     - Executor row: `agent=role or "advisor_call_executor"`, `model=executor_model`, sum of executor iterations input+output tokens.
     - Advisor row (only if advisor was invoked): `agent=(role or "advisor_call") + "_advisor_tool"`, `model=advisor_model`, sum of advisor iteration tokens.
   - Return `{"text": full_executor_text, "executor_tokens": (input, output), "advisor_tokens": (input, output) or (0, 0), "advisor_invoked": bool, "request_id": response.id}`.

2. **Settings flag in `backend/config/settings.py`:**
   - Add `enable_advisor_tool: bool = False` to the AppSettings class (default OFF for safe rollout).

3. **Cost tracker update in `backend/agents/cost_tracker.py`:**
   - Add `is_advisor: bool = False`, `advisor_input_tokens: int = 0`, `advisor_output_tokens: int = 0` fields to `AgentCostEntry`.
   - Add `record_advisor_call(agent_name, executor_model, advisor_model, executor_input_tokens, executor_output_tokens, advisor_input_tokens, advisor_output_tokens)` method that creates an entry with blended cost = `executor_input_tokens * MODEL_PRICING[executor_model][0] / 1e6 + executor_output_tokens * MODEL_PRICING[executor_model][1] / 1e6 + advisor_input_tokens * MODEL_PRICING[advisor_model][0] / 1e6 + advisor_output_tokens * MODEL_PRICING[advisor_model][1] / 1e6` and sets `is_advisor=True`.

4. **Synthesis integration in `backend/agents/orchestrator.py`:**
   - In `run_synthesis_pipeline` (around line 1048 initial synthesis + 1128 revision loop): when `settings.enable_advisor_tool == True` AND `self.synthesis_client.model_name.startswith("claude-opus-4")`, call `llm_client.advisor_call(prompt=draft_prompt, system_prompt=..., executor_model="claude-sonnet-4-6", advisor_model=self.synthesis_client.model_name, role="Synthesis")` instead of `self.synthesis_client.generate_content(draft_prompt)`. Capture the result text + iterations split into the cost_tracker.
   - Keep the existing path as the default (flag-off case) -- zero behavioral change for users not opting in.

5. **Verification + live smoke + A/B test:**
   - Run the immutable verification command (`from backend.agents.llm_client import advisor_call`).
   - Run ONE live advisor_call() against a representative synthesis prompt (stored sample from `analysis_results` or a synthesized prompt). Capture the response shape, iterations[] breakdown, and confirm BQ row written with `agent LIKE '%_advisor_tool'`.
   - Run a controlled A/B comparison: same synthesis prompt through (A) advisor_call() and (B) generate_content() on `claude-opus-4-7`. Compare:
     - cost_usd (executor + advisor blended vs Opus-solo)
     - JSON output structural similarity (parse both as JSON if possible; compare `conviction`, `recommendation` if present)
     - latency
   - Write all evidence to `handoff/current/live_check_26.2.md` and a summary to `experiment_results.md`.

## Scope honesty / out-of-scope

- The settings flag defaults to `False` -- this step ADDS the capability, does NOT flip the default. A separate operator-driven enable + monitoring step (phase-26.2.1 or a phase-27 affordance) is required to actually use the advisor in production.
- A/B test sample size: 1-3 synthesis-style prompts (not the brief's 20-30 ticker full autonomous cycles). Rationale: full autonomous cycles cost ~$2-5 in real LLM spend and run for ~20+ minutes; a focused 1-3 prompt comparison demonstrates non-regression on representative inputs without the expense. Full multi-cycle A/B is deferred to operator-driven rollout once the flag flips True.
- This step does NOT touch `multi_agent_orchestrator.py::_synthesize` or `planner_agent.py` -- Priority-1 (orchestrator.py synthesis chain) is the only adoption point in 26.2 scope. Priority 2 + 3 from the brief are deferred to phase-26.2.x sub-steps or phase-27.
- This step does NOT modify the BQ `llm_call_log` schema -- the `agent='<role>_advisor_tool'` encoding satisfies live_check with zero migration risk.
- This step does NOT change the Anthropic SDK pinning. The advisor tool is on the existing `anthropic` SDK; `client.beta.messages.create` is a sibling of `client.messages.create` in the same package.
- This step does NOT auto-route to advisor for non-Opus synthesis (Gemini synthesis chains stay on their existing path -- the advisor branch only fires when `synthesis_client.model_name.startswith("claude-opus-4")`).

## References
- Research brief: `handoff/current/research_brief.md` (canonical, archives correctly)
- Masterplan step JSON: `.claude/masterplan.json` step `26.2`
- Anthropic Advisor Tool doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
- Anthropic release notes April 9 2026 entry: https://platform.claude.com/docs/en/release-notes/overview
- Synthesis integration site: `backend/agents/orchestrator.py:961-1140` (`run_synthesis_pipeline`), `:363-424` (synthesis_client construction)
- Cost-tracker integration site: `backend/agents/cost_tracker.py:83-100` (`AgentCostEntry` dataclass)
- BQ observability writer: `backend/services/observability/api_call_log.py:203-277` (writer auto-fetches cycle_id + session_cost_usd from autonomous_loop, added in 26.1)
- Existing beta-header injection pattern: `backend/agents/llm_client.py:1244-1247` (`files-api-2025-04-14` pattern)
- Existing Anthropic SDK call site (non-beta): `backend/agents/llm_client.py:1387` (`client.messages.create`)
