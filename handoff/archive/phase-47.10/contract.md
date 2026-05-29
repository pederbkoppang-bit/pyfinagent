# Contract — phase-47.10: generate_content max_tokens floor (symmetric close of the Opus-4.8 max_tokens audit)

**Cycle:** 11 (Priority 3 "audit per-agent max_tokens at xhigh" — closing the audit asymmetry: 47.9 floored the orchestrator Opus path; this floors the SECOND Opus path, `llm_client.generate_content`). **LLM spend:** $0 (static + unit test). **Severity:** LOW / defensive symmetry (operator-override-only reachability — see below).

## Research-gate summary
`researcher` `a2073408b08340a8d` (gate **PASSED**): 5 sources read in full, 9 snippet-only, 14 URLs, recency scan, 7 internal files. Brief: `handoff/current/research_brief_phase_47_10_generate_content_floor.md` (builds on 47.9's brief — same governing external fact).

Key findings:
- **External (re-confirmed):** Anthropic adaptive-thinking doc verbatim "Use `max_tokens` as a hard limit on total output (thinking + response text)"; at high/max effort the model "can be more likely to exhaust the `max_tokens` budget". The **effort doc decisively settles** the effort-without-thinking question: "Set `thinking: {type: 'adaptive'}` to enable thinking; **without it, requests run without thinking**" → effort alone creates ZERO hidden thinking tokens. So the floor must gate on `thinking_requested`, NOT on effort.
- **The gap is real** in `generate_content`: `max_tokens` (`:1285`, default 2048) → kwargs `:1332` with NO floor, while `:1384-1394` sets adaptive thinking + `:1427-1451` sets `output_config.effort` (xhigh for Opus) — identical to what 47.9 fixed in the orchestrator.
- **REACHABILITY VERDICT — OPERATOR-OVERRIDE-ONLY (low/latent, not a live default-config bug):** the only live thinking-on-Claude path is `risk_debate.py:62` (RiskJudge), gated on `ClaudeClient.supports_thinking`; `debate.py:66` is GeminiClient-gated (can never route thinking to Claude). Reaching the gap needs TWO simultaneous non-default flips: `ENABLE_THINKING=true` (`settings.py:35` default **False**) AND `DEEP_THINK_MODEL=claude-opus-4-8/-4-7` (`settings.py:30` default **gemini-2.5-pro**, deliberately reverted off Opus in phase-37.2 to stop a credit regression). If both flip, the RiskJudge adaptive call shares `max_tokens` of just 1024–1536 between thinking + the verdict. This is a defensive symmetry fix.

## Hypothesis
Flooring `generate_content`'s `max_tokens` on the Opus-4.8/4.7 adaptive-thinking path (mirroring 47.9) removes the last unfloored Opus thinking path, so the max_tokens-at-xhigh audit is symmetric and complete — closing the inconsistency of having one Opus path protected and the other not, at $0.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-47.10)
1. generate_content floors max_tokens to the thinking-safe value (16384, the same _OPUS_ADAPTIVE_MIN_MAX_TOKENS as 47.9) ONLY when thinking is requested AND the model is Opus-4.8/4.7, via a pure unit-tested helper; the floor is a no-op when thinking is off OR the model is not Opus (effort-without-thinking is NOT floored, per the Anthropic effort doc)
2. large configured max_output_tokens budgets are respected (floor is a max(), never lowers a caller's higher budget); the helper introduces no import cycle (defined locally in llm_client.py)
3. a pytest guard asserts the helper floors the thinking+Opus case, no-ops the thinking-off and non-Opus cases, and respects a large budget; ast.parse clean; pytest green; llm_client imports clean
4. the silent text-tail stop_reason=max_tokens swallow (llm_client.py ~:1591) is NOT in scope and is flagged as a documented follow-up

## Plan steps
1. `llm_client.py`: add module const `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384` + pure helper `_opus_adaptive_max_tokens(max_tokens, model_id, thinking_requested)` before `class ClaudeClient` (`:1184`): returns `max(int(max_tokens), FLOOR)` iff `thinking_requested and model_id.startswith(("claude-opus-4-8","claude-opus-4-7"))`, else `int(max_tokens)`.
2. In `generate_content`, after the effort block (~after `:1451`): `kwargs["max_tokens"] = _opus_adaptive_max_tokens(kwargs["max_tokens"], model_id, thinking_requested)`. (`thinking_requested` + `model_id` are already locals from `:1382-1383`.)
3. `tests/agents/test_phase_47_10_generate_content_floor.py`: `_opus_adaptive_max_tokens(2048, "claude-opus-4-8", True)==16384` (floored); `(2048, "claude-opus-4-8", False)==2048` (thinking off → no-op); `(2048, "claude-sonnet-4-6", True)==2048` (non-Opus → no-op); `(30000, "claude-opus-4-8", True)==30000` (large respected); const==16384.
4. Verify: `ast.parse` llm_client.py + import check + `pytest tests/agents/test_phase_47_10_generate_content_floor.py -q`.

## Out-of-scope (FLAGGED follow-ups, disclosed to Q/A)
- The silent **text** `stop_reason=="max_tokens"` swallow in ClaudeClient (~`:1591-1594`) — behavior/cost change, deferred (same call made in 47.9).
- COMMUNICATION router effort, the `:987` `_handle_direct` emoji, the openclaw token literal — pre-existing, unrelated.

## References
- `handoff/current/research_brief_phase_47_10_generate_content_floor.md` + `..._47_9_opus48_finish.md`
- Anthropic adaptive-thinking + effort docs ("without [thinking arg], requests run without thinking")
- phase-47.9 (`multi_agent_orchestrator._adaptive_max_tokens`, the orchestrator twin this mirrors)
