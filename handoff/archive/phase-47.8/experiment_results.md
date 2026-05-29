# Experiment Results — phase-47.8: Opus-4.8 stale-pin sweep

**Cycle:** 9 (priority 3 "/claude-api sweep" + goal "exploit Opus 4.8 fully"). **LLM spend:** $0 (static/structural edits + unit test). **Result:** ready for Q/A.

## Root cause (research-validated)
Commit `8ecc9efe` bumped the *canonical* pins 4-7→4-8 but missed ~8 operative files. The most dangerous miss: `multi_agent_orchestrator.py:1061` `if agent_config.model.startswith("claude-opus-4-7"):` is **False** for the now-4-8 pin → a 4-8 Layer-2 agent fell into the ELSE branch that sets manual `budget_tokens` + `temperature=1`, which **Opus 4.8 rejects with a 400** (4.8 inherits 4.7's adaptive-thinking-only / no-sampling constraint). Plus two missing-4-8 map entries (`harness_memory` context-window → 4-8 truncated to 128K; `app_home` dropdown) and several operative stale 4-7 default pins.

## Edits (11 code edits across 9 files; 0 emojis added)
**Criterion 1 (CRITICAL):**
- `multi_agent_orchestrator.py:1061` — `startswith("claude-opus-4-7")` → `startswith(("claude-opus-4-8", "claude-opus-4-7"))`. 4-8 now takes the adaptive-only/no-sampling path (no 400).

**Criterion 2 (missing-4-8 maps ADDED, 4-7 kept):**
- `harness_memory.py:52` — ADD `"claude-opus-4-8": 1_000_000` (4-7 kept at :53; comment updated to "4.8/4.7/4.6"). Behaviorally verified: `get_context_window("claude-opus-4-8")` now returns `1_000_000` (was `128_000` — the unknown-model default).
- `app_home.py:20` — ADD `"claude-opus-4-8"` as the first entry of `AVAILABLE_MODELS` (4-7 kept at :21).

**Criterion 3 (operative stale 4-7 DEFAULT pins → 4-8):**
- `ticket_queue_processor.py:166,167,171` — `agent_model_map` main + q-and-a + `.get()` default → 4-8 (research deliberately Sonnet-4-6).
- `rag_agent_runtime.py:187` vision-query `model` default → 4-8 (+ docstring :204 + :194 "Opus 4.8's vision").
- `planner_agent.py:58,275` — `PlannerAgent.__init__` + `get_planner_agent` defaults → 4-8.
- `autonomous_loop.py:74` — `AutonomousLoopOrchestrator.__init__ planner_model` default → 4-8 (flows into `PlannerAgent(model=self.planner_model)` at :367).
- `multi_agent_orchestrator.py:154` masker `model_name` + `:936` `should_reset_context` default → 4-8.
- `openclaw_client.py:49,50` — `AGENT_MODEL_OVERRIDES` main + qa → `anthropic/claude-opus-4-8` (research stays Sonnet).
- Doc-only consistency: `multi_agent_orchestrator.py:26,27`, `streaming_integration.py:10,11`, `openclaw_client.py:10` roster comments → 4-8.

**LEAVE (legit 4-7 compat — verified 4-8 co-present):** `cost_tracker` 4-7 pricing (4-8 added in 47.3), `harness_memory` 4-7 window, `model_tiers` EFFORT_SUPPORTED_MODELS + MODEL_EFFORT_FALLBACK (both versions), `llm_client` accept-lists/provider-map (4-8 at :471/:584/:1989), `settings_api` pricing+accept, `settings.py`/`main.py` historical notes.

## Verbatim verification output (corrected immutable command)
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in [...9 files...]]; print('ast OK 9 files')"
ast OK 9 files
$ python -m pytest tests/agents/test_phase_47_8_opus48_pins.py -q
...........                                                              [100%]
11 passed in 0.88s
```
Behavioral confirmations (the guards exercise the real lookup/signatures, not source greps):
- `get_context_window("claude-opus-4-8") == 1_000_000` ; `get_context_window("zzz-nonexistent") == 128_000` (proves the 4-8 entry is load-bearing, not tautological).
- `inspect.signature(...).default == "claude-opus-4-8"` for PlannerAgent, get_planner_agent, AutonomousLoopOrchestrator.planner_model, MultiAgentOrchestrator.should_reset_context, multimodal_index_claude.
- `AGENT_MODEL_OVERRIDES["main"/"qa"] == "anthropic/claude-opus-4-8"`, `["research"]` unchanged.
- AST-extracted `agent_model_map` main/q-and-a == 4-8, `.get` default == 4-8, narrow 4-7 default absent.
- Compat: MODEL_PRICING 4-7 kept, EFFORT_SUPPORTED_MODELS + fallback hold both, llm_client provider-map routes both.
- All 9 edited modules import cleanly (probed pre-test).

## Immutable-command correction (disclosed to Q/A — NOT goalpost-moving)
The command I authored when adding 47.8 ast-checked `backend/services/autonomous_loop.py` (the 47.7 learn-loop file) instead of the file actually edited this cycle, `backend/autonomous_loop.py` (the orchestrator with the `planner_model` default). Corrected the path + added `streaming_integration.py` (9 files now) BEFORE any Q/A evaluation. The success_criteria (the immutable part) were untouched. Same class of pre-evaluation false-negative fix as phase-47.1.

## Success-criteria mapping (masterplan phase-47.8)
1. Orchestrator branch widened to include 4-8 (no 400) — **MET** (test_orchestrator_thinking_branch_includes_4_8; narrow form asserted absent).
2. 4-8 map entries added, 4-7 kept — **MET** (harness_memory behavioral 1M + 128K-default contrast; app_home 4-8 first, 4-7 kept).
3. operative 4-7 defaults → 4-8; compat 4-7 preserved — **MET** (5 signature guards + AST map guard + 2 compat guards).
4. pytest guard + ast clean + green — **MET** (ast OK 9 files; 11 passed).

## Scope honesty / deferred
$0 static/structural change (no live LLM call). The 1061 400-fix's *live* confirmation defers to the next real Layer-2 MAS run on a 4-8 pin (cron/manual) — no Anthropic spend incurred here. Out-of-scope smells noted as follow-ups, NOT fixed: a hardcoded `OPENCLAW_GATEWAY_TOKEN` literal in `openclaw_client.py:33-36` (pre-existing secret-in-source) and a pre-existing emoji in `app_home.py` AGENT_DISPLAY (outside my AVAILABLE_MODELS edit). `scripts/harness/run_autonomous_loop.py:73 planner_model="claude-opus-4-6"` (2 versions behind) remains a separate follow-up flagged in prior cycles.

## Files
backend/agents/multi_agent_orchestrator.py, backend/agents/harness_memory.py, backend/slack_bot/app_home.py, backend/services/ticket_queue_processor.py, backend/agents/rag_agent_runtime.py, backend/agents/planner_agent.py, backend/autonomous_loop.py, backend/agents/openclaw_client.py, backend/slack_bot/streaming_integration.py, tests/agents/test_phase_47_8_opus48_pins.py, .claude/masterplan.json (phase-47.8), handoff/current/{contract.md, research_brief_phase_47_8_opus48_sweep.md}.
