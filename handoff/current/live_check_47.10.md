# Live Check — phase-47.10: generate_content max_tokens floor

`verification.live_check` = "n/a -- deterministic static/structural unit test ($0, no live LLM call).
Reachability is operator-override-only." Verbatim output, 2026-05-29.

## Immutable command (ast + helper asserts + pytest) -- exit 0
```
$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read()); import backend.agents.llm_client as L; assert L._opus_adaptive_max_tokens(2048,'claude-opus-4-8',True)==16384 and L._opus_adaptive_max_tokens(2048,'claude-opus-4-8',False)==2048 and L._opus_adaptive_max_tokens(2048,'claude-sonnet-4-6',True)==2048 and L._opus_adaptive_max_tokens(30000,'claude-opus-4-8',True)==30000; print('ast+helper OK')"
ast+helper OK
$ python -m pytest tests/agents/test_phase_47_10_generate_content_floor.py -q
......                                                                   [100%]
6 passed in 0.17s
```

## The floor is gated correctly (behavioral)
```
_opus_adaptive_max_tokens(2048, "claude-opus-4-8",  True)  -> 16384  (thinking+Opus: FLOORED)
_opus_adaptive_max_tokens(1024, "claude-opus-4-7",  True)  -> 16384  (thinking+Opus: FLOORED)
_opus_adaptive_max_tokens(2048, "claude-opus-4-8",  False) ->  2048  (thinking OFF: no-op)
_opus_adaptive_max_tokens(2048, "claude-sonnet-4-6",True)  ->  2048  (non-Opus: no-op)
_opus_adaptive_max_tokens(30000,"claude-opus-4-8",  True)  -> 30000  (large budget respected)
```
Gate = `thinking_requested AND Opus-4.8/4.7`. Effort-without-thinking is NOT floored -- validated against
Anthropic's effort doc ("without [the adaptive thinking arg], requests run without thinking"). Floor value
== the orchestrator's 47.9 floor (both imported + asserted equal), so the two Opus paths are now symmetric.

## Applied at the right place (verified at source)
`generate_content` routes `kwargs["max_tokens"]` through the helper AFTER the thinking + effort resolution
(`:1475-1481`), using the already-resolved `thinking_requested` + `model_id` locals.

## Reachability (why this is LOW severity / defensive)
Operator-override-only: needs ENABLE_THINKING=true (default False) AND DEEP_THINK_MODEL=claude-opus-4-8
(default gemini-2.5-pro, reverted off Opus in phase-37.2). Not a live default-config bug -- a symmetry fix
so no Opus thinking path is left unfloored.

## Deferred (NOT this cycle)
Silent text-tail `stop_reason=max_tokens` swallow (~`:1591`) -- behavior/cost change, flagged follow-up.
