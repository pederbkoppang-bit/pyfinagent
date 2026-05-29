# Experiment Results — phase-47.9: Priority-3 completion (Opus-4.8 max_tokens floor + driver-pin finish)

**Cycle:** 10 (Priority 3 remainders: "audit per-agent max_tokens at xhigh" + finish "/claude-api sweep"). **LLM spend:** $0 (static/structural edits + unit test, no live LLM call). **Result:** ready for Q/A.

## Root cause (research-validated, brief `..._47_9_opus48_finish.md`, gate PASSED 6 sources)
On the Opus-4.8 **adaptive** thinking path (`multi_agent_orchestrator.py:1061` IF-branch), `max_tokens` is a HARD ceiling on **thinking + visible text COMBINED** (Anthropic adaptive-thinking doc: "Use max_tokens as a hard limit on total output (thinking + response text)"). Layer-2 agents run at `effort=max` (`model_tiers.py:221-225`) but the adaptive call set `max_tokens=agent_config.max_tokens + 2048` (~2548-5048 for the 500-3000 configured agents) → adaptive thinking at max effort can exhaust that and **silently starve the visible answer** (the plain-text `stop_reason=="max_tokens"` path only logs + returns partial; retry fires only on a `tool_use` tail). Separately, three stale `claude-opus-4-6` driver pins the backend-only 47.8 sweep didn't reach put the planner on a 2-versions-old model.

## Edits (7 edits across 4 files; 0 emojis added)
**Criterion 1 — adaptive max_tokens floor (pure, testable):**
- `multi_agent_orchestrator.py` — added module const `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384` + pure helper `_adaptive_max_tokens(configured) = max(configured + 2048, FLOOR)`. In the Opus-4.8/4.7 adaptive branch (`:1095`): `_max_tokens = _adaptive_max_tokens(agent_config.max_tokens)`. ELSE (manual-thinking, budget_tokens=2048 bounded) branch (`:1104`): `_max_tokens = agent_config.max_tokens + 2048` (unchanged behavior). The create (`:1107`) and tool_use retry (`:1217`, now `min(_max_tokens * 2, 32768)`) both use the floored ceiling. 16384 matches the prior retry cap, covers the largest intended output (Synthesis 4096) + ample thinking, well under Opus 4.8's 128k cap. max_tokens is a CEILING (not a target) → higher floor is $0 unless the model needs the room.

**Criterion 2 — three 4-6 driver pins → 4-8:**
- `scripts/harness/run_autonomous_loop.py:73` `planner_model="claude-opus-4-6"` → `"claude-opus-4-8"` (was overriding the now-correct 4-8 default at `backend/autonomous_loop.py:74`).
- `scripts/mas_harness/run_cycle.sh:63` `--model claude-opus-4-6` → `claude-opus-4-8` (launchd MAS-cycle `claude -p` CLI flag).
- grep: **zero** remaining `claude-opus-4-6` anywhere in `scripts/`.

**Criterion 3 — PlannerAgent parse hardened (consequence of #2 putting it on 4.8):**
- `planner_agent.py` — added `_first_text(response)` (joins text-typed blocks, skips thinking/tool_use, falls back to `content[0].text`); replaced both `response.content[0].text` sites (`:176`, `:282`). Tolerant of a leading thinking block, so the planner-on-4.8 cannot raise on `content[0]`.

## Audited but NOT changed (grep-all-consumers diligence, disclosed)
- `_call_agent` (`:1006`, one-shot classify/plan/synthesis/quality-gate) passes **no `thinking` arg** → not on the adaptive-starvation path; `max_tokens=agent_config.max_tokens` is pure output budget; its `:1012` extraction already joins text blocks robustly (no content[0] fragility). Left unchanged. (If Opus 4.8 ever defaults thinking ON for plain calls, this would need the same floor — flagged for the live-smoke follow-up.)
- `:817 max_tokens=2000`, `:903 max_tokens=800` — fixed small budgets on non-adaptive/Gemini paths; not flagged by research.

## Verbatim verification output (immutable command)
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/agents/multi_agent_orchestrator.py','backend/agents/planner_agent.py','scripts/harness/run_autonomous_loop.py']]; print('ast OK 3 py files')"
ast OK 3 py files
$ bash -n scripts/mas_harness/run_cycle.sh && echo 'sh OK'
sh OK
$ python -m pytest tests/agents/test_phase_47_9_max_tokens_floor.py -q
........                                                                 [100%]
8 passed in 0.18s
```
Extra self-checks: hot-path imports OK (`multi_agent_orchestrator` + `planner_agent` load; floor const=16384; helper(500)=16384). `_adaptive_max_tokens`: 500→16384, 3000→16384, 4096→16384, 14336→16384 (boundary), 30000→32048 (large respected), always ≥ configured. `_first_text`: thinking-block-first → returns text; multi-text → joined; untyped single block → content[0] fallback; empty/None → "".

## Success-criteria mapping (masterplan phase-47.9)
1. adaptive branch floors via pure unit-tested helper (low→floor, high respected), ELSE unchanged, retry above floor — **MET** (test_adaptive_max_tokens_* + test_orchestrator_uses_the_floor_on_adaptive_branch_only).
2. three 4-6 driver pins → 4-8; no operative 4-6 left in scripts/ — **MET** (test_run_autonomous_loop_* + test_run_cycle_sh_* + grep clean).
3. PlannerAgent parse thinking-block tolerant — **MET** (4 `_first_text` tests).
4. pytest guard + ast + bash -n clean + green — **MET** (ast OK 3; sh OK; 8 passed).

## Scope honesty / FLAGGED follow-ups (NOT fixed)
$0 static/structural change; the floor + planner-on-4.8 **live** confirmation defers to the next real Layer-2 MAS cycle (Anthropic-metered = operator-gated). Deferred by design (in contract): `llm_client.generate_content` max_tokens floor (separate Layer-1 path), COMMUNICATION router `effort=max`+`max_tokens=500` (owner-directive collision — operator call), and making the silent **text** `stop_reason=max_tokens` path retry (behavior/cost change — the floor makes text truncation rare regardless). Pre-existing out-of-scope: a `👋` emoji in `multi_agent_orchestrator.py:987 _handle_direct` (NOT in this diff; flagged like the 47.8 app_home emoji).

## Files
backend/agents/multi_agent_orchestrator.py, backend/agents/planner_agent.py, scripts/harness/run_autonomous_loop.py, scripts/mas_harness/run_cycle.sh, tests/agents/test_phase_47_9_max_tokens_floor.py, .claude/masterplan.json (phase-47.9), handoff/current/{contract.md, research_brief_phase_47_9_opus48_finish.md}.
