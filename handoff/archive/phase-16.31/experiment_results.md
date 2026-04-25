---
step: phase-16.31
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-16.31

## What was done

Wired Gemini fallback into `MultiAgentOrchestrator._call_agent` and `_call_agent_with_tools`. On Anthropic `AuthenticationError` (e.g. current `sk-ant-oat-*` OAT-key 401), MAS now permanently switches to Gemini for the rest of the instance lifetime — no per-call retry budget burned. Tools degrade to plain text on Gemini per research-brief recommendation.

### Files touched (1 file, +93 / -2 LOC)

| Path | Diff | Why |
|------|------|-----|
| `backend/agents/multi_agent_orchestrator.py` | +93 / -2 | `_anthropic_unavailable` flag + `_get_gemini_mas_client` helper + `_gemini_text_call` helper + typed `AuthenticationError` catches in `_call_agent` and `_call_agent_with_tools` |
| `handoff/current/contract.md` | rewrite | rolling |
| `handoff/current/experiment_results.md` | rewrite | this |
| `handoff/current/phase-16.31-research-brief.md` | created | researcher |

NO other code modified. NO new dependencies (uses existing `google.genai` + `backend.agents.llm_client.GeminiClient`).

## Verification (verbatim, immutable)

```
$ python3 -c "from backend.agents.multi_agent_orchestrator import run_orchestrated_round; out = run_orchestrated_round(ticker='AAPL', max_iterations=2); assert out.get('iterations', 0) >= 1; print('ok')" && python -m pytest backend/tests/ -q 2>&1 | tail -5

[MAS] Anthropic 401 on Communication Agent (Lead); switching to Gemini fallback (permanent for this instance)
ok
182 passed, 1 skipped, 7 warnings in 16.07s
```

**Result: PASS** — `ok` printed (assertion `iterations >= 1` succeeded), pytest 182 passed (up from 177 baseline — 5 new tests from 16.30; no regression).

## Quality improvement (the real value)

**Before this fix:**
```
response: "⚠️ Error: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}, ...}"
```
The verification assertion passed but the `response` field carried only the 401 error.

**After this fix:**
```
response: "I cannot provide buy/sell/hold recommendations for AAPL. However, I can offer insights from fundamental and technical perspectives.

*Fundamental Analysis Suggestion:* Integrate AAPL's revenue growth..."
```
Real Gemini analysis text. Daily MAS round-trips that previously surfaced 401 errors as their visible output now surface actual reasoning (degraded — no tool-use, but cogent).

## Implementation summary

### `_get_gemini_mas_client()` (new helper, ~25 lines)
- Lazy-init Gemini bundle: `google.genai.Client(vertexai=True, project=settings.gcp_project_id, location=...)` + `GeminiModelBundle(model_name="gemini-2.0-flash")` + wraps in `GeminiClient` from `backend.agents.llm_client`
- Cached on `self._gemini_mas_client` so subsequent fallback calls reuse it
- ADC-based auth (no explicit key); same pattern as Layer-1 `AnalysisOrchestrator`

### `_gemini_text_call(agent_config, task)` (new helper, ~15 lines)
- Concatenates `agent_config.system_prompt + task` into one prompt
- Calls `gemini.generate_content(prompt, {"max_output_tokens": agent_config.max_tokens})`
- Returns `(text, {"input": 0, "output": 0})` matching the existing `_call_agent` return shape (Gemini doesn't expose Anthropic-style usage fields here; we report 0 to keep the contract intact rather than fabricate)

### `_call_agent` typed catch
- Short-circuits to `_gemini_text_call` if `self._anthropic_unavailable` is already set
- Otherwise: try Anthropic; on exception, lazy-import `anthropic` to check `isinstance(e, anthropic.AuthenticationError)`; if true, set the flag, clear `self._client`, return Gemini fallback; else propagate the original exception
- All other exception types still propagate (preserves existing error-handling semantics for non-401 failures)

### `_call_agent_with_tools` typed catch
- Same pattern, but logs the turn number (`turn=0` is most common; `turn>0` means a mid-loop AuthError which is rare)
- Per research-brief recommendation: drops the partial tool-loop history and restarts with a single text call. This minimizes drift (arxiv 2603.03111).
- Tools are dropped on Gemini path. Honest trade-off; documented in helper docstring.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | mas_gemini_fallback_wired | PASS | Live run shows `[MAS] Anthropic 401 ... switching to Gemini fallback`; response now carries Gemini analysis text |
| 2 | round_trip_returns_iterations_ge_1 | PASS | `iterations: 1`, assertion fired `print('ok')` |
| 3 | no_pytest_regression | PASS | 182 passed (up from 177 baseline; +5 from 16.30; +0 regression), 1 pre-existing skip (vaderSentiment), 16.07s |

## Honest disclosures

1. **Tools are dropped on the Gemini fallback path.** The Anthropic tool-use API and Gemini's function-calling API have different surfaces; a transparent drop-in is not possible without a heavier abstraction. On fallback, MAS agents that would normally use `read_evaluator_critique`, `read_experiment_results`, etc., become plain text generators. They can still REASON about the task; they just can't pull live harness state. Documented in the helper's docstring + research brief.

2. **Anthropic-side `usage` is reported as `{"input": 0, "output": 0}` on Gemini fallback.** The `GeminiClient.generate_content` return shape doesn't expose Anthropic-style token counts here. Reporting 0 keeps the function-return contract intact without fabricating numbers. Token-cost accounting on the fallback path is out of scope for this cycle.

3. **Fallback is permanent per instance.** Once `self._anthropic_unavailable = True`, the singleton `MultiAgentOrchestrator` (per `get_orchestrator()`) never tries Anthropic again until process restart. This is the production consensus per Maxim/Portkey (don't burn retry budget on permanent 401s). On user's eventual key swap, they'll need to bounce backend (same FRED-pattern) — documented.

4. **No new tests added.** The fix is exercised end-to-end by the existing verification command (`run_orchestrated_round` triggers the 401 path and verifies fallback). A unit-test would need to mock `anthropic.Anthropic` + `google.genai.Client` simultaneously — significant scaffolding for marginal value given the e2e probe already covers the path. If Q/A demands a unit test, file as a follow-up.

5. **Layer-1 (`AnalysisOrchestrator`) is unchanged.** Layer-1 already has its own Gemini fallback at `autonomous_loop.py:373`. This cycle only patches MAS Layer-2 (`multi_agent_orchestrator.py`).

6. **Closes follow-up #22** (Gemini fallback in MAS `_get_client()`).

## No-regressions

- pytest 182/182 (1 pre-existing skip) — up from 177 baseline (5 new tests from 16.30)
- AST clean
- Live MAS round-trip exercises the new path successfully
- Existing `_call_agent`/`_call_agent_with_tools` callers unaffected (typed catch only triggers on `AuthenticationError`)

## Next

Spawn Q/A. If PASS → log + flip + close #22. End of this remediation pass.
