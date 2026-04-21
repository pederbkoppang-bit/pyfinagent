# Adaptive Thinking Audit — Claude Doc Alignment (phase-4.10.1)

## Documentation summary

Source: [Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)
(plus linked sub-pages: [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
and [Effort parameter](https://platform.claude.com/docs/en/build-with-claude/effort).)

**What it is.** Instead of the operator manually setting a thinking
`budget_tokens`, adaptive thinking lets Claude "dynamically determine when and
how much to use extended thinking based on the complexity of each request."
It is activated by sending `thinking: {"type": "adaptive"}` in a Messages
request. No beta header is required. Interleaved thinking (reasoning between
tool calls) is auto-enabled in adaptive mode.

**Key parameters (verbatim from the doc):**
- `thinking: {"type": "adaptive"}` — turns on adaptive mode.
- `thinking: {"type": "enabled", "budget_tokens": N}` — manual mode,
  **deprecated on Opus 4.6 / Sonnet 4.6** and **rejected with 400 on Opus 4.7**.
- `thinking: {"type": "disabled"}` — off.
- `output_config: {"effort": ...}` — soft guidance. Levels: `max`, `xhigh`
  (Opus 4.7 only), `high` (default — "Claude always thinks"), `medium`
  ("may skip thinking for very simple queries"), `low` ("minimizes thinking").
- `thinking: {"display": "summarized" | "omitted"}` — on Opus 4.7 / Mythos
  Preview the default is `omitted`; must set `summarized` explicitly to keep
  reading thinking text.
- `max_tokens` acts as the hard cap across `thinking + response`.

**Model support matrix (verbatim):**
| Model | Adaptive | Manual `budget_tokens` |
|-------|----------|------------------------|
| Claude Mythos Preview (`claude-mythos-preview`) | default — applies if `thinking` unset | not supported (`disabled` also not supported) |
| Claude Opus 4.7 (`claude-opus-4-7`) | only supported mode | rejected with 400 |
| Claude Opus 4.6 (`claude-opus-4-6`) | supported | functional but deprecated |
| Claude Sonnet 4.6 (`claude-sonnet-4-6`) | supported | functional but deprecated |
| Older (Sonnet 4.5, Opus 4.5, Haiku, 3.x) | **not supported** | required |

**Cost tradeoffs.** Billed output tokens = full (un-summarized) thinking
tokens, regardless of `display`. Switching modes between turns breaks the
messages cache breakpoint (system + tools stay cached). Consecutive adaptive
requests preserve cache.

## Codebase audit

pyfinAgent has three distinct Claude-consuming surfaces.

**1. MAS orchestrator (Layer 2)** — `backend/agents/multi_agent_orchestrator.py`
- Direct `anthropic.Anthropic` client at
  `multi_agent_orchestrator.py:155-168`.
- Tool-loop subagent call at
  `multi_agent_orchestrator.py:944-954` sends `thinking={"type": "enabled",
  "budget_tokens": 2048}` — the deprecated manual form.
- No `output_config.effort`. No adaptive path. Same 2048 budget fires every
  turn regardless of complexity.

**2. `ClaudeClient` provider (Layer-1 & skill loops)** — `backend/agents/llm_client.py:581-672`
- `generate_content()` reads `config["thinking"]` and builds
  `{"type": "enabled", "budget_tokens": budget}` (`llm_client.py:622-628`). No
  `"adaptive"` branch. Comment at `:246` still claims this is
  "Gemini 2.5+ extended thinking config" — stale.
- Temperature is force-set to 1 whenever a budget is present, which is correct
  for manual mode but unnecessary for adaptive.

**3. Model routing** — `backend/config/model_tiers.py`
- Hardcoded role→model map:
  - `mas_main` / `mas_qa` → `claude-opus-4-6`
  - `mas_communication` / `mas_research` → `claude-sonnet-4-6`
  - `autoresearch_fast` → `claude-haiku-4-5`
- `resolve_model()` is a pure dict lookup — **no dynamic routing** based on
  query complexity, latency SLO, or spend.
- `QueryComplexity` enum (`agent_definitions.py:38-42`) classifies
  trivial/simple/moderate/complex, but the classification only picks WHICH
  subagent runs, not which model that subagent uses — a Communication agent
  always costs Sonnet 4.6, Main always Opus 4.6.

**4. Settings / budgets** — `backend/config/settings.py:31-35`
- `enable_thinking`, `thinking_budget_critic=8192`,
  `thinking_budget_moderator=8192`, `thinking_budget_risk_judge=4096`,
  `thinking_budget_synthesis=4096`. All are Gemini-targeted (see gate at
  `debate.py:55` / `risk_debate.py:55`: `if isinstance(model, GeminiClient)`)
  — Claude path never reads these.

**5. Cost observability** — `backend/agents/cost_tracker.py` is wired into
`orchestrator.py:914, 1307, 1463, 1485` with a per-analysis budget ceiling
(`settings.max_analysis_cost_usd`). Latency is not tracked per-model, and
there is no cost-aware router that downshifts on overspend.

## Findings

| Aspect | Status | Evidence | Notes |
|--------|--------|----------|-------|
| Adaptive mode used anywhere | **Missing** | `llm_client.py:622-628`, `multi_agent_orchestrator.py:950-953` | All Claude calls pass `type: "enabled"` only. |
| Using `claude-opus-4-6` where Opus 4.7 is newer | **Outdated** | `model_tiers.py:46,48` | Opus 4.7 requires adaptive — cannot migrate without the code change above. |
| Manual `budget_tokens` on Opus/Sonnet 4.6 | **Deprecated usage** | `multi_agent_orchestrator.py:952` | Doc: "deprecated … plan to migrate." |
| `output_config.effort` used | **Missing** | grep: no matches | Soft-guidance tier not exercised anywhere. |
| `thinking.display` controlled | **Missing** | grep: no matches | Opus 4.7 default (`omitted`) will silently blank `thoughts` field in `llm_client.py:637-638`. |
| Dynamic Opus↔Sonnet↔Haiku routing | **Missing** | `model_tiers.py:42-59` (pure dict) | `QueryComplexity` picks agent, not model. |
| Per-call cost tracking | **Correct** | `cost_tracker.py` + `orchestrator.py:1485` | Budget ceiling enforced. |
| Latency-awareness | **Missing** | no p95 tracker on LLM path (perf_tracker covers HTTP only) | — |
| Interleaved thinking between tool calls | **Correct (manual)** | `multi_agent_orchestrator.py:940-953` | Already matches Anthropic pattern; adaptive would enable it automatically. |

## Gaps & Opportunities

**MUST FIX:**
1. Extend `ClaudeClient.generate_content` in `llm_client.py:622-628` to accept
   `{"type": "adaptive"}` (and pass through `output_config.effort` +
   `thinking.display`). Without this, moving any role to `claude-opus-4-7`
   is a hard 400.
2. Swap `multi_agent_orchestrator.py:950-953` to adaptive
   (`{"type": "adaptive"}`) with effort = `high` for Main/QA, `medium` for
   Communication, `low` for Citation. Deprecation clock is ticking on Opus 4.6
   and Sonnet 4.6.
3. Add an `output_config.effort` field to `AgentConfig`
   (`agent_definitions.py:46-54`) so per-role guidance is explicit.

**NICE TO HAVE:**
4. Replace the static `_BUILD_TIER` map with a complexity-aware router:
   `QueryComplexity.TRIVIAL` → Haiku 4.5, `SIMPLE` → Sonnet 4.6 + `effort=low`,
   `MODERATE` → Sonnet 4.6 + `effort=medium`, `COMPLEX` → Opus 4.7 adaptive
   + `effort=high`. This is the pyfinAgent-native analog to adaptive
   thinking at the **model-selection** layer (adaptive thinking only flexes
   the reasoning budget within one model).
5. Set `thinking.display="summarized"` explicitly on Opus 4.7 calls where the
   Glass Box UI shows `thoughts` — otherwise `LLMResponse.thoughts` goes
   empty silently.
6. Drop the Gemini-only gating in `debate.py:55` /
   `risk_debate.py:55` once (1) lands; Claude judges could use adaptive for
   free and benefit from interleaved thinking.
7. Audit and prune `.claude/agents/*.md` — `qa-evaluator.md` (opus),
   `researcher.md` / `harness-verifier.md` (sonnet) are hardcoded and could
   instead rely on adaptive effort levels.

## References

1. https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
2. https://platform.claude.com/docs/en/build-with-claude/extended-thinking
3. https://platform.claude.com/docs/en/build-with-claude/effort
4. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
5. https://www.anthropic.com/engineering/multi-agent-research-system
6. https://www.anthropic.com/engineering/harness-design-long-running-apps
