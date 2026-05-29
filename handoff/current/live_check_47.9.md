# Live Check — phase-47.9: Opus-4.8 max_tokens floor + driver-pin finish

`verification.live_check` = "n/a -- deterministic static/structural unit test ($0, no live LLM call)."
Verbatim output, 2026-05-29.

## Immutable command (ast 3 py + bash -n sh + pytest) -- exit 0
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in [
    'backend/agents/multi_agent_orchestrator.py','backend/agents/planner_agent.py',
    'scripts/harness/run_autonomous_loop.py']]; print('ast OK 3 py files')"
ast OK 3 py files
$ bash -n scripts/mas_harness/run_cycle.sh && echo 'sh OK'
sh OK
$ python -m pytest tests/agents/test_phase_47_9_max_tokens_floor.py -q
........                                                                 [100%]
8 passed in 0.18s
```

## The floor is genuine (behavioral on the pure helper)
```
_adaptive_max_tokens(500)    -> 16384   (floored -- the starvation fix)
_adaptive_max_tokens(4096)   -> 16384   (largest intended output still floored)
_adaptive_max_tokens(14336)  -> 16384   (boundary)
_adaptive_max_tokens(30000)  -> 32048   (large configured budget respected, +2048)
```
A small configured budget (the 500-3000 of the Layer-2 agents) is raised to a thinking-safe
ceiling; a large budget is respected. max_tokens is a CEILING, so this costs nothing unless the
model actually needs the room.

## Applied to the adaptive branch ONLY (verified at source)
```
:1095  _max_tokens = _adaptive_max_tokens(agent_config.max_tokens)   # Opus-4.8/4.7 adaptive
:1104  _max_tokens = agent_config.max_tokens + 2048                  # ELSE (manual) -- UNCHANGED
:1107  max_tokens=_max_tokens,                                       # create uses the floored ceiling
:1217  _retry_max = min(_max_tokens * 2, 32768)                      # retry above the floor
```
The non-adaptive ELSE branch caps thinking at budget_tokens=2048, so it is not starved -- left
unchanged. `_call_agent` (:1006) passes no thinking arg (not on the adaptive path) -- audited, unchanged.

## Driver pins finished (grep clean)
```
scripts/harness/run_autonomous_loop.py:73  planner_model="claude-opus-4-8"
scripts/mas_harness/run_cycle.sh:63        --model claude-opus-4-8
grep -rn "claude-opus-4-6" scripts/        -> (none -- clean)
```

## PlannerAgent on 4.8 cannot raise on content[0]
`_first_text(thinking-block-first response) == "the answer"` (Opus 4.8 can emit a thinking block
before text; the old `content[0].text` would have returned the thinking block / raised). Joins
multi-text, falls back to content[0] for untyped single blocks, safe on empty/None.

## Deferred (NOT live this cycle)
The floor + planner-on-4.8 LIVE confirmation defers to the next real Layer-2 MAS cycle (cron/manual;
Anthropic-metered = operator-gated). Flagged follow-ups (NOT fixed, in contract): generate_content
floor, COMMUNICATION effort (owner-directive collision), silent text-truncation retry (behavior/cost
change). Pre-existing out-of-scope: `multi_agent_orchestrator.py:987` `_handle_direct` emoji (not in diff).
