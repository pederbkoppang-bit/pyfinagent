# Experiment Results — phase-47.10: generate_content max_tokens floor (symmetric close of the Opus-4.8 max_tokens audit)

**Cycle:** 11 (Priority 3 "audit per-agent max_tokens at xhigh" — closing the audit asymmetry left by 47.9). **LLM spend:** $0. **Severity:** LOW / defensive symmetry (operator-override-only reachability). **Result:** ready for Q/A.

## Root cause + reachability (research-validated, brief `..._47_10_...`, gate PASSED 5 sources)
47.9 floored the ALWAYS-adaptive orchestrator Opus path but explicitly deferred `llm_client.generate_content`, which has the SAME gap: `max_tokens` (`:1285` default 2048 → kwargs `:1332`) is shared with adaptive thinking (`:1384-1394`) + xhigh effort (`:1427-1451`) for Opus-4.8/4.7, with NO floor. **Reachability = operator-override-only (low/latent):** needs BOTH `ENABLE_THINKING=true` (`settings.py:35` default False) AND `DEEP_THINK_MODEL=claude-opus-4-8/-4-7` (`settings.py:30` default gemini-2.5-pro, reverted off Opus in phase-37.2). The live near-miss is `risk_debate.py` RiskJudge sharing a 1024–1536 `max_tokens`. The effort doc decisively validated NOT flooring effort-without-thinking ("without [the adaptive thinking arg], requests run without thinking" → no hidden thinking tokens).

## Edits (2 edits / 1 file; 0 emojis)
- `llm_client.py` (before `class ClaudeClient`, `:1184`): added module const `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384` (twin of the orchestrator's 47.9 value; defined locally to avoid an import cycle) + pure helper `_opus_adaptive_max_tokens(max_tokens, model_id, thinking_requested)` → `max(int(max_tokens), FLOOR)` iff `thinking_requested AND model_id.startswith(("claude-opus-4-8","claude-opus-4-7"))`, else `int(max_tokens)`.
- `generate_content` (after the effort block, `:1473`): `kwargs["max_tokens"] = _opus_adaptive_max_tokens(kwargs["max_tokens"], model_id, thinking_requested)` — `thinking_requested` + `model_id` are already locals (`:1382-1383`).

## Verbatim verification output (immutable command)
```
$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read()); import backend.agents.llm_client as L; assert L._opus_adaptive_max_tokens(2048,'claude-opus-4-8',True)==16384 and ...(2048,'claude-opus-4-8',False)==2048 and ...(2048,'claude-sonnet-4-6',True)==2048 and ...(30000,'claude-opus-4-8',True)==30000; print('ast+helper OK')"
ast+helper OK
$ python -m pytest tests/agents/test_phase_47_10_generate_content_floor.py -q
......                                                                   [100%]
6 passed in 0.17s
```
Behavioral (non-tautological): floors thinking+Opus (2048→16384, 1024→16384); no-op thinking-off (2048→2048); no-op non-Opus (sonnet/haiku/gemini/""→2048); respects large budget (30000→30000); boundary (16384→16384); floor == orchestrator's 47.9 floor (imported both, asserted equal).

## Success-criteria mapping (masterplan phase-47.10)
1. floors max_tokens to 16384 ONLY on thinking+Opus via a pure unit-tested helper; no-op thinking-off / non-Opus (effort-without-thinking NOT floored) — **MET** (test_floors_opus_with_thinking + test_noop_when_thinking_off + test_noop_when_not_opus).
2. large budgets respected; no import cycle (local definition; llm_client imports clean) — **MET** (test_respects_larger_caller_budget; import succeeded in the immutable command).
3. pytest guard + ast clean + green + import clean — **MET** (ast+helper OK; 6 passed).
4. silent text-tail swallow NOT in scope, flagged — **MET** (see below).

## Scope honesty / FLAGGED follow-ups (NOT fixed)
$0 static change; live confirmation defers to the operator-override config (ENABLE_THINKING + Opus deep-think) + a RiskJudge cycle. NOT fixed (flagged, in contract): the silent **text** `stop_reason=="max_tokens"` swallow in ClaudeClient (~`:1591-1594`) — behavior/cost change, deferred (same call made in 47.9). Unrelated pre-existing: COMMUNICATION router effort, `multi_agent_orchestrator.py:987` `_handle_direct` emoji, openclaw token literal.

## Files
backend/agents/llm_client.py, tests/agents/test_phase_47_10_generate_content_floor.py, .claude/masterplan.json (phase-47.10), handoff/current/{contract.md, research_brief_phase_47_10_generate_content_floor.md}.
