# Sprint Contract -- phase-26.3
Step: Wire Gemini code_execution on 4 quant skills

## Research Gate
researcher_af1c2e9efe7672f1d (tier=complex, MAX gate per user instruction 2026-05-16) gate_passed=true.
Brief: `handoff/current/research_brief.md` (canonical).
- 7 unique external URLs read in full via WebFetch (3 Tier-1: ai.google.dev/code-execution doc, Vertex AI code-execution-api doc, ai.google.dev/code-execution Python lang variant; 2 Tier-2: Google engineering blog gemini-20-deep-dive, GCP cookbook sample notebook; 2 Tier-3: Google AI Developers Forum failure-case thread, Medium quant practitioner blog). 5 snippet-only; 12 URLs total. 4-variant search (exceeds 3-variant floor).
- Recency scan (2024-04 -> 2026-05) reported: Gemini 2.0 file-input + Matplotlib output 2025; Gemini 3 family + ADK first-class promotion 2026; no contradictions; canonical 30s/Python-only/no-custom-library constraints stable.
- Internal grep covered 10 modules.
- **3 critical findings:**
  1. **GeminiClient does NOT write log_llm_call rows** -- ClaudeClient writes them at line 1548 of llm_client.py; GeminiClient does not. The `llm_call_log` table currently has NO Gemini rows. **Prerequisite for live_check.**
  2. **quant_strategy.md is NOT a pipeline agent** -- it is an optimizer research skill loaded by `quant_optimizer.py::_propose_llm()`. The 4 grep hits must include quant_optimizer.py.
  3. **Current text extraction skips `code_execution_result.output`** -- `GeminiClient.generate_content` lines 892-898 only extract `part.text`; arithmetic results in `part.code_execution_result.output` would be silently dropped.
- API surface: REST `{"tools":[{"code_execution":{}}]}`; Python SDK `types.Tool(code_execution=types.ToolCodeExecution())`. Response parts interleave text/executable_code/code_execution_result. Outcome enum: OUTCOME_OK / OUTCOME_FAILED / OUTCOME_DEADLINE_EXCEEDED. 30s runtime cap, Python only, no custom libraries.
- The 4 quant skills' routing:
  - `quant_model_agent` -> orchestrator.py:952-957 `general_client`
  - `scenario_agent` -> orchestrator.py:945-950 `general_client`
  - `enhanced_macro_agent` -> orchestrator.py:908-915 `grounded_client` (needs BOTH google_search + code_execution)
  - `quant_strategy` -> `quant_optimizer.py::_propose_llm()` (separate optimizer path)

## Hypothesis
Wiring `code_execution` on 3 pipeline quant agents (quant_model_agent, scenario_agent, enhanced_macro_agent) + the `quant_optimizer._propose_llm` path satisfies the 4-grep-hit verification requirement. Dedicated `GeminiModelBundle` for the quant-only path avoids enabling code_execution for the other 12 enrichment agents. Extending GeminiClient text extraction surfaces `code_execution_result.output` so verified arithmetic isn't dropped. Adding `log_llm_call` writes in `_generate_with_retry` for Gemini calls (analogous to ClaudeClient line 1548) closes the prerequisite gap so live_check BQ rows are queryable with `agent LIKE '%_code_exec'`.

## Success Criteria (immutable, copied verbatim from .claude/masterplan.json step 26.3)
```
grep -rn 'code_execution' backend/agents/ --include='*.py' | wc -l
```
Must produce >=4 hits.

Plus sub-criteria:
- `code_execution_tool_added_to_4_quant_skill_configs` -- satisfied by 4 wiring points: (a) new `_quant_exec_vertex` GeminiModelBundle for quant_model + scenario agents; (b) extension of `_grounded_vertex.tools` for enhanced_macro_agent; (c) quant_optimizer.py `_propose_llm` wiring; (d) GeminiClient text extractor update (counts as `code_execution` reference in llm_client.py).
- `regression_test_shows_sharpe_arithmetic_consistent_pre_post` -- satisfied by a controlled comparison: 1-2 tickers with code_execution disabled vs enabled. Compare `signal` / `risk_profile` / regime label; expect MATCH within tolerance.
- `llm_call_log_records_code_execution_tool_usage` -- satisfied by `log_llm_call` writes in `_generate_with_retry` with `agent=f"{agent_name}_code_exec"` when the bundle contains `ToolCodeExecution`.

live_check: `handoff/current/live_check_26.3.md` -- verbatim BQ row dump showing 1+ rows with `agent LIKE '%_code_exec'` from a real quant_model_agent invocation. Plus the executable_code + code_execution_result.output captured verbatim.

## Plan (PRE-commit; will NOT diverge in Generate)

1. **Extend GeminiClient text extraction** (llm_client.py:892-898) to surface `part.code_execution_result.output` (when `outcome == OUTCOME_OK`) and `part.executable_code.code`. Append with clear delimiter.
2. **Create `_quant_exec_vertex` GeminiModelBundle** in orchestrator.py (alongside `_general_vertex` and `_grounded_vertex` around line 410-436). Include `code_execution` tool. Add `self.quant_exec_client = make_client(...)`.
3. **Re-route quant_model_agent and scenario_agent** to use `self.quant_exec_client` (orchestrator.py:945-957). Single-line changes per call site.
4. **Extend `_grounded_vertex.tools`** (orchestrator.py:432-436) to include `code_execution` alongside `_google_search_tool`. If the API rejects this combination at runtime, fall back to code_execution-only (documented).
5. **Add Gemini log_llm_call writes** in `_generate_with_retry` (orchestrator.py:510-578) after `ct.record()`. Suffix agent name with `_code_exec` when bundle contains `ToolCodeExecution`.
6. **Wire `quant_optimizer.py::_propose_llm`** to include `code_execution` tool. 4th grep hit.
7. **Update 4 skill prompt files** with a `## Code Execution Tasks` section instructing the model to invoke code_execution for specific arithmetic verifications.
8. **Verification + live smoke + regression**:
   - Grep verification (>=4).
   - 1 live Gemini call with code_execution; capture executable_code + outcome + output + final text.
   - BQ row check: `WHERE agent LIKE '%_code_exec'`.
   - 1-ticker pre/post regression comparison.

## Scope honesty / out-of-scope

- quant_optimizer wiring verified by grep + code inspection, NOT runtime (optimizer runs in separate harness; out of scope for 26.3 smoke).
- Regression test: N=1 ticker (within 1-2 pre-committed). A divergence is allowable but must be documented.
- Skill prompt updates are minimal `## Code Execution Tasks` sections, not full prompt redesign (phase-27).
- Gemini log_llm_call writes ONLY fire when bundle has ToolCodeExecution. Other 12 enrichment agents remain UN-LOGGED to llm_call_log. Universal Gemini observability deferred.
- Combined `[google_search, code_execution]` tool list for enhanced_macro_agent: brief claims "explicitly supported by Gemini 2.0"; if rejected at runtime, fall back to code_execution-only.
- No BQ schema migration. `agent` field encoding re-uses 26.2's `_advisor_tool` pattern.

## References
- Research brief: `handoff/current/research_brief.md`
- Masterplan step JSON: `.claude/masterplan.json` step `26.3`
- Gemini code_execution: https://ai.google.dev/gemini-api/docs/code-execution
- GeminiClient: `backend/agents/llm_client.py:826-940`
- Orchestrator bundles: `backend/agents/orchestrator.py:390-436`
- _generate_with_retry: `backend/agents/orchestrator.py:510-578`
- Quant call sites: orchestrator.py:945-957, 908-915
- log_llm_call writer: `backend/services/observability/api_call_log.py:203-277`
- ClaudeClient log_llm_call template: `backend/agents/llm_client.py:1548-1561`
