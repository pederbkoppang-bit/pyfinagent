---
name: rail-failforward-72-0-2
description: Step 72.0.2 (fail-forward off a dead CC rail) -- BOTH step premises refuted; the lite path bypasses make_client; rail-dead never raises; the Vertex leg has a None-trap
metadata:
  type: project
---

Step 72.0.2 asked for a fail-forward at the `make_client` provider-order seam so a
dead CC rail degrades to Vertex-Gemini instead of lite->HOLD. Research refuted TWO
premises and found a third trap.

**1. The cited seam range was wrong.** The step cited
`backend/agents/llm_client.py:1983-2042`; that range is `BatchClient.poll/fetch`.
`make_client` starts at `:2044`, the CC-rail branch is `:2114-2133`, the
routing-breach `raise` is `:2144` (the "78.1 comment cites :2163" is now inside a
phase-78.16 comment block). Line numbers in step text drift -- re-derive every time.

**2. The path that actually emits the HOLD does NOT use `make_client`.**
`_run_claude_analysis` imports `anthropic` directly and builds
`anthropic.Anthropic(...)` at `autonomous_loop.py:2478`, or calls
`claude_code_invoke` when the rail flag is on. So a provider-order-only fix cannot
change the lite->HOLD chain. A second seam at `_select_lite_analyzer:2196` is
required -- and `_run_gemini_analysis` re-reads `settings.gemini_model` itself and
HARD-RAISES at `:2758` if it is not `gemini-*`, so flipping the dispatcher alone
raises instead of fixing.

**3. Rail-dead never raises.** Probe fails -> `rail_guard_disable` ->
`ClaudeCodeClient` returns `LLMResponse(text="", thoughts="rail_guard_skipped: ...")`
(`claude_code_client.py:737-741`) -> regex finds no JSON -> fabricated
`{"action":"HOLD","confidence":0,"score":5,"_parse_failed":True}` at
`autonomous_loop.py:2549`. "Find the raise" was the wrong search; the failure is
silent-empty.

**4. Vertex None-trap.** `make_client`'s Vertex leg is
`GeminiClient(model=vertex_model)` -- only `orchestrator.py:625-651` passes a real
bundle; every C-block service and `autonomous_loop.py:2764` pass `None`. Rewriting
the model string to gemini gives either AI-Studio-direct (if `GEMINI_API_KEY` is set,
NOT Vertex) or `GeminiClient(model=None)` -- the same $0 outcome with extra steps.
The seam must build its own bundle from ADC.

**5. Naming trap:** `settings.gemini_model` (`settings.py:31`) IS the standard-tier
selector and its default value is `"claude-sonnet-4-6"`. `EFFORT_DEFAULTS` is a
role->effort map, not a model-tier map -- do not route model selection through it.

**Why:** the step was written from a stale reading of `llm_client.py` and assumed
one chokepoint where there are two. Shipping Seam A alone would have produced a
green `ast.parse` verification command, a plausible-looking diff, and zero change
to the $0-book behaviour the step exists to fix.

**How to apply:** on any "change the provider order" step, first prove the failing
call site actually reaches `make_client` (grep for `import anthropic` /
`claude_code_invoke` / `genai.Client` in the failing module). Also verify the
verification command can discriminate -- here it is only `ast.parse`, which cannot
prove any of the three criteria.

Related: [[project_cc_rail_guard_66_1]], [[project_cc_rail_vs_claudeclient_78_1]],
[[project_decision_input_integrity_61_2]].
