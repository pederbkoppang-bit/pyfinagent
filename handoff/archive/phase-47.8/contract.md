# Contract — phase-47.8: Opus-4.8 stale-pin sweep

**Cycle:** 9 (priority 3 "/claude-api sweep" + goal "exploit Opus 4.8 fully"). **LLM spend:** $0 (static/structural edits + unit test, no live LLM call).

## Research-gate summary
`researcher` agent (gate PASSED) → `handoff/current/research_brief_phase_47_8_opus48_sweep.md` (26 KB).
Floor met: ≥5 sources read in full (Anthropic Opus 4.8 model card / effort doc / migration notes), recency scan present, internal code-audit leg classified every `claude-opus-4-7` occurrence in `backend/`.

Key research findings:
- Commit `8ecc9efe` bumped the *canonical* pins 4-7→4-8 (`model_tiers.py` mas_main/mas_qa/autoresearch_strategic, `cost_tracker` pricing add, `llm_client` accept-lists, `settings_api`) but missed ~8 operative files.
- **CRITICAL operative bug** — `backend/agents/multi_agent_orchestrator.py:1061`: `if agent_config.model.startswith("claude-opus-4-7"):` is now **False** for the 4-8 pin, so a 4-8 agent falls into the ELSE branch that sets a manual `budget_tokens` + `temperature=1`. Opus 4.8 **rejects** manual budget_tokens + sampling params with a **400** (4.8 inherits 4.7's adaptive-thinking-only / no-sampling constraint). This silently breaks any Layer-2 MAS agent pinned to 4-8.
- **Missing-4-8 map entries**: `harness_memory.MODEL_CONTEXT_WINDOWS` has 4-7=1M but **no 4-8** → a 4-8 `get_context_window()` lookup falls to the 128K default (premature context resets / masking). `app_home.AVAILABLE_MODELS` dropdown lacks 4-8 → operator can't select it.
- **Operative stale 4-7 DEFAULT pins** (no 4-8 present, these are the live default): `ticket_queue_processor` agent_model_map (main/q-and-a/default), `rag_agent_runtime` vision default, `planner_agent` (__init__ + factory), `backend/autonomous_loop.py` planner_model, `openclaw_client` main/qa overrides, `multi_agent_orchestrator` masker model (:154) + should_reset_context default (:936).
- **LEAVE (legit compat)**: `cost_tracker` 4-7 pricing, `harness_memory`/`settings_api`/`model_tiers` 4-7 entries (4-8 already present alongside), `llm_client` registry/provider-map/accept-lists (4-8 already at :471/:584/:1989 — 4-7 is valid legacy fallback), `settings.py` deep_think_model historical note, `main.py` historical comments. Verified via grep that 4-8 is already present in every KEEP location.

## Hypothesis
Bumping the operative 4-7 defaults to 4-8, ADDING the two missing 4-8 map entries, and (most importantly) widening the orchestrator thinking/sampling branch to include 4-8 will: (a) eliminate the latent 400 on any 4-8 Layer-2 agent, (b) give 4-8 its true 1M window in harness memory, (c) let the operator pick 4-8 in the Slack home, and (d) make the whole stack default to the now-canonical flagship — all at $0 and same $5/$25 pricing.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-47.8)
1. CRITICAL: multi_agent_orchestrator.py thinking/sampling branch widened so claude-opus-4-8 takes the adaptive-only/no-sampling path (startswith includes claude-opus-4-8); 4-8 no longer hits the manual budget_tokens+temperature=1 ELSE branch that Opus 4.8 rejects with a 400
2. missing-4-8 map entries ADDED (4-7 kept): harness_memory.MODEL_CONTEXT_WINDOWS['claude-opus-4-8']=1_000_000; app_home.AVAILABLE_MODELS includes claude-opus-4-8
3. operative stale 4-7 DEFAULT pins bumped to 4-8 (ticket_queue_processor agent_model_map main/q-and-a + default, rag_agent_runtime vision default, planner_agent + autonomous_loop planner defaults, openclaw_client main/qa); legit compat 4-7 entries (MODEL_EFFORT_FALLBACK, cost_tracker MODEL_PRICING, llm_client xhigh/effort accept-lists) PRESERVED
4. a pytest guard asserts the 4-8 map entries + bumped defaults + 4-7-preserved-in-compat; ast.parse clean on all edited files; pytest green

## Plan steps
1. **CRITICAL first** — `multi_agent_orchestrator.py:1061` widen `startswith` to `("claude-opus-4-8","claude-opus-4-7")`.
2. ADD `harness_memory.py:52` `"claude-opus-4-8": 1_000_000,` (keep 4-7) — load-bearing for the :936 default bump.
3. ADD `app_home.py:20` `"claude-opus-4-8",` to AVAILABLE_MODELS (keep others).
4. Bump operative defaults 4-7→4-8: `ticket_queue_processor.py:166,167,171`; `rag_agent_runtime.py:187` (+ :204 docstring); `planner_agent.py:58,275`; `backend/autonomous_loop.py:74`; `openclaw_client.py:49,50`; `multi_agent_orchestrator.py:154,936`.
5. Doc consistency (non-operative): `multi_agent_orchestrator.py:26,27`, `streaming_integration.py:10,11`, `openclaw_client.py:10` roster comments → 4-8.
6. LEAVE every compat/pricing/accept-list/historical 4-7 entry (verified 4-8 already co-present).
7. Write `tests/agents/test_phase_47_8_opus48_pins.py`: assert MODEL_CONTEXT_WINDOWS has 4-8=1M AND keeps 4-7; AVAILABLE_MODELS has 4-8; the bumped signatures/maps default to 4-8; line 1061 includes 4-8; cost_tracker still has 4-7 pricing (compat preserved).
8. Verify: `ast.parse` 8 files + pytest green.

## References
- `handoff/current/research_brief_phase_47_8_opus48_sweep.md`
- Anthropic Opus 4.8 model card + effort doc (adaptive-only thinking, xhigh accepted, $5/$25)
- CLAUDE.md Effort-policy section (2026-05-28 4-7→4-8 model bump basis)
- Prior commit `8ecc9efe` (the partial canonical-pin bump this step completes)
