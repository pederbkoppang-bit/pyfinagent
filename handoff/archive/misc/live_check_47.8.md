# Live Check — phase-47.8: Opus-4.8 stale-pin sweep

`verification.live_check` = "n/a -- deterministic static/structural unit test ($0, no live LLM call)."
Verbatim output, 2026-05-29.

## Immutable command (ast 9 files + pytest) -- exit 0
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in [
    'backend/agents/multi_agent_orchestrator.py','backend/agents/harness_memory.py',
    'backend/slack_bot/app_home.py','backend/services/ticket_queue_processor.py',
    'backend/agents/rag_agent_runtime.py','backend/agents/planner_agent.py',
    'backend/autonomous_loop.py','backend/agents/openclaw_client.py',
    'backend/slack_bot/streaming_integration.py']]; print('ast OK 9 files')"
ast OK 9 files
$ python -m pytest tests/agents/test_phase_47_8_opus48_pins.py -q
...........                                                              [100%]
11 passed in 0.88s
```

## The fix is genuine, not tautological (behavioral contrast)
```
get_context_window("claude-opus-4-8")   -> 1_000_000   (POST-fix; was 128_000 pre-fix)
get_context_window("zzz-nonexistent")   ->   128_000   (the unknown-model default)
```
A dropped 4-8 entry would silently truncate the harness context window 1M -> 128K. The test asserts
both the 1M hit AND the 128K default, so it cannot pass by accident.

## The CRITICAL 400-bug fix (criterion 1)
`multi_agent_orchestrator.py:1061` post-edit:
```
if agent_config.model.startswith(("claude-opus-4-8", "claude-opus-4-7")):
```
A 4-8 pin now takes the adaptive-thinking-only / no-sampling path. Pre-edit (narrow single-string
`startswith("claude-opus-4-7")`) a 4-8 agent fell into the ELSE branch (manual budget_tokens +
temperature=1) which Opus 4.8 rejects with a 400. The guard asserts the widened tuple is present AND the
narrow form is absent.

## No operative 4-7 default remains; compat 4-7 preserved
grep post-edit: every remaining `claude-opus-4-7` in backend/*.py (non-test) is a legit
pricing/effort-fallback/accept-list/provider-map/historical-comment entry (4-8 verified co-present in
each). No live LLM call, no Anthropic spend, no operator-gated flag touched.

## Deferred (NOT live this cycle)
The 1061 400-fix's live confirmation defers to the next real Layer-2 MAS run on a 4-8 pin (cron/manual) --
that path is Anthropic-metered = operator-gated. Out-of-scope follow-ups (NOT fixed here): hardcoded
OPENCLAW_GATEWAY_TOKEN literal (openclaw_client.py), pre-existing app_home AGENT_DISPLAY emoji,
run_autonomous_loop.py:73 planner_model=claude-opus-4-6.
