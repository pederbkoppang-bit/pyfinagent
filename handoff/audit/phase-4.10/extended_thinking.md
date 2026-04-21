# Extended Thinking Audit — Claude Doc Alignment (phase-4.10.0)

## Documentation summary

Sources: `…/build-with-claude/extended-thinking` and `…/adaptive-thinking` (fetched 2026-04-18).

Canonical shape: `"thinking": {"type": "enabled"|"adaptive"|"disabled", "budget_tokens": N, "display": "summarized"|"omitted"}`.

- `type:"enabled"` + `budget_tokens` is **deprecated on Opus 4.6 / Sonnet 4.6** and **rejected 400 on Opus 4.7** — Opus 4.7 accepts only `"adaptive"`. Mythos defaults to adaptive when `thinking` is unset.
- `budget_tokens` caps: 128k (Opus 4.7/4.6), 64k (Sonnet 4.6/Haiku 4.5). Must be `< max_tokens` except with interleaved thinking.
- `display`: `"summarized"` (default ≤4.6) vs `"omitted"` (default Opus 4.7/Mythos). `"omitted"` reduces time-to-first-text, **not cost** — full thinking tokens are always billed.
- Adaptive pairs with `output_config.effort` of `low|medium|high|xhigh|max` (high=default; xhigh Opus 4.7 only). Adaptive auto-enables interleaved thinking; no beta header. Legacy 4.5/4 manual mode needs `anthropic-beta: interleaved-thinking-2025-05-14`.
- Tool use: `tool_choice` only `auto` or `none`; thinking blocks with their `signature` must be round-tripped unchanged when returning `tool_result`, else API strips them and silently disables thinking.
- Caching: changes to `thinking.type`/`budget_tokens` invalidate **message** cache breakpoints; system/tools caches survive. Adaptive↔enabled switches break message caches. Use 1-hour cache for thinking workflows.
- Signatures are opaque, cross-platform (API/Bedrock/Vertex). `usage.output_tokens` reflects full, not visible, thinking.

## Codebase audit

| Location | Behaviour |
|---|---|
| `llm_client.py:621-628` | `ClaudeClient` accepts `thinking.budget_tokens` and always emits `{"type":"enabled"}`. Forces `temperature=1`. **No adaptive/display/effort.** |
| `orchestrator.py:85-108, 394-397` | Judge configs hard-code `{"type":"enabled","budget_tokens":4096-8192}`. Injection **gated by `isinstance(model, GeminiClient)`** — Claude judges never get thinking even with `ENABLE_THINKING=true`. |
| `risk_debate.py:50-56, 257-263` | Same Gemini-only gate. |
| `config/settings.py:30-35` | `enable_thinking=False` default, budgets keyed to Gemini. No Claude flag. |
| `multi_agent_orchestrator.py:944-954` | MAS subagents call `thinking={"type":"enabled","budget_tokens":2048}` unconditionally in tool loop. Models are `claude-opus-4-6`/`claude-sonnet-4-6` (`model_tiers.py:46-54`). |
| `planner_agent.py:35, 115-122` | `claude-opus-4-6`, `max_tokens=1500`, **no thinking**. |
| `evaluator_agent.py` | Vertex/Gemini — no Anthropic path. |
| `ticket_queue_processor.py:164-213` | `claude-opus-4-6`, `max_tokens=1000`, no thinking. |
| `services/autonomous_loop.py:419-438` | Uses retired `claude-sonnet-4-20250514`; no thinking. |
| `slack_bot/mcp_tools.py:73, 243` | `client.beta.messages.create` with `claude-sonnet-4-20250514`; no thinking. |

Beta header `interleaved-thinking-2025-05-14` is **not present** anywhere; signature round-tripping in the MAS tool loop is implicit (response appended to `messages`) and not asserted.

## Findings

| Aspect | Status | Evidence | Notes |
|--------|--------|----------|-------|
| Claude extended thinking wired in `ClaudeClient` | Incorrect (deprecated mode) | `llm_client.py:626` emits `{"type":"enabled"}` only | Rejected on Opus 4.7; deprecated on 4.6 |
| Adaptive thinking (`type:"adaptive"` + `effort`) | Missing | grep shows zero hits for `adaptive` / `effort` in Claude paths | Opus 4.7 accepts only this mode |
| Judge-agent thinking on Claude backend | Missing | `orchestrator.py:396` `isinstance(model, GeminiClient)` gate | Switching judges to Claude silently disables extended thinking |
| MAS subagent interleaved thinking | Partially correct | `multi_agent_orchestrator.py:944-954` uses `type:"enabled", budget=2048` | Works on Opus 4.6 / Sonnet 4.6 (deprecated); will 400 if upgraded to 4.7 |
| Thinking-block round-trip in tool loop | Unverified | `_call_agent_with_tools` appends response to `messages` (must preserve `signature`) | Need explicit test; stripping thinking blocks silently disables mode |
| `thinking.display` control | Missing | No references | Opus 4.7 defaults to `"omitted"` — current code won't see thinking text |
| `temperature=1` forcing on thinking | Correct | `llm_client.py:628` | Matches doc requirement |
| Interleaved-thinking beta header | Missing | Not present | Needed only if staying on legacy 4.5/4 manual mode; N/A for 4.6/4.7 adaptive |
| Prompt-caching interaction awareness | Partial | Caching on in `ClaudeClient:602-609`, but no policy about invalidation when toggling thinking | Toggling modes mid-session will silently blow message cache |
| Planner / Evaluator / Researcher thinking | Missing | `planner_agent.py:115`, harness agents `.claude/agents/*.md` | Highest-value targets per doc guidance on multi-step reasoning |
| Signal pipeline / paper-trading thinking | Correctly absent | Layer-1 uses Gemini; latency-sensitive loops don't touch Claude thinking | Keep off to protect latency |

## Gaps & Opportunities

MUST FIX (correctness — blocks Opus 4.7 adoption):
- Extend `ClaudeClient.generate_content` (`backend/agents/llm_client.py:622`) to route `thinking.type` (accept `"adaptive"`), forward an optional `display`, and accept `output_config={"effort": ...}`. Gate manual vs adaptive on model name (`claude-opus-4-7` → force adaptive; older → allow manual with deprecation warning).
- Drop the `isinstance(model, GeminiClient)` gate in `orchestrator.py:396` and `risk_debate.py:55`. Refactor so each provider client decides whether thinking is supported; on Claude, always pass the `thinking` config through.
- Update `multi_agent_orchestrator.py:944-954` to use `{"type":"adaptive"}` with `effort: "high"` when model starts with `claude-opus-4-7`/`claude-opus-4-6`/`claude-sonnet-4-6`, and drop the hard-coded 2048 budget (adaptive manages it). Verify `signature` on thinking blocks is round-tripped unchanged in the tool loop — add an assertion.
- Retire the `claude-sonnet-4-20250514` reference in `backend/services/autonomous_loop.py:438` (legacy name; upgrade to `claude-sonnet-4-6` or adaptive-capable `claude-opus-4-7`).

NICE TO HAVE:
- Give the harness agents (`.claude/agents/researcher.md`, `qa-evaluator.md`, `harness-verifier.md`) adaptive thinking with `effort: "xhigh"` (Opus 4.7) — multi-step reasoning is exactly the doc's recommended use case, and the harness is already latency-tolerant.
- Add adaptive thinking with `effort: "high"` to `planner_agent.py` (strategy proposals are bimodal-complex, the doc's headline adaptive use case).
- Expose `effort` and `display` in `backend/config/settings.py` alongside `enable_thinking`; surface in the settings UI (Model Configuration BentoCard — mirrors existing pattern in `frontend/src/lib/types.ts:495-500`).
- Keep thinking **off** on `backend/agents/orchestrator.py` Layer-1 enrichment (latency-sensitive, Gemini anyway), MAS "Communication" routing agent (sub-second classification), and signal-pipeline/paper-trading loops — explicit doc advice: "respond directly" when reasoning won't improve quality.
- After migration, prune legacy manual-mode paths on Opus 4.7 (stress-test doctrine — scaffolding for deprecated API shape is dead weight).

## References

1. https://platform.claude.com/docs/en/build-with-claude/extended-thinking (accessed 2026-04-18)
2. https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking (accessed 2026-04-18)
3. https://platform.claude.com/docs/en/build-with-claude/effort (accessed 2026-04-18, referenced)
4. https://platform.claude.com/docs/en/build-with-claude/prompt-caching (accessed 2026-04-18, referenced)
5. https://platform.claude.com/docs/en/build-with-claude/streaming (accessed 2026-04-18, referenced)
6. https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview (accessed 2026-04-18, referenced)
7. https://platform.claude.com/docs/en/api/beta-headers (accessed 2026-04-18, referenced for `interleaved-thinking-2025-05-14`)
