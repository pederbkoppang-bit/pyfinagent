# Compliance: Extended + Adaptive Thinking + Effort (phase-4.15.2)

Audit date: 2026-04-18. Scope: external Anthropic docs (4 pages) + internal
backend code. No source files were modified.

---

## Pattern inventory

### Pattern 1: thinking type "enabled" rejected on Opus 4.7
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — "Claude Opus 4.7 and Later"
- **What the doc says:** "Manual extended thinking NOT supported — returns 400 error. Use instead: adaptive thinking with effort parameter."
- **Status:** ❗ incorrect (latent — 400 will fire if Opus 4.7 is ever routed here)
- **Evidence:**
  - `backend/agents/multi_agent_orchestrator.py:950-953` — `thinking={"type": "enabled", "budget_tokens": 2048}` hard-coded in `_call_agent_with_tools` tool loop. Model is `agent_config.model`, which resolves to `claude-opus-4-6` today but is a runtime variable.
  - `backend/agents/llm_client.py:626` — `ClaudeClient.generate_content` always emits `{"type": "enabled", "budget_tokens": budget}` when `budget > 0`.
  - `backend/agents/planner_agent.py:35` and `backend/agents/evaluator_agent.py:84` — default model strings are hardcoded as `claude-opus-4-6`; no guard prevents callers from passing `claude-opus-4-7`.
- **Deviation:** No model version gate. All thinking config paths emit `type: "enabled"` unconditionally. If `_BUILD_TIER` or a caller swaps to `claude-opus-4-7`, every call that passes a thinking config silently 400s.
- **Risk:** Silent 400 at runtime. The tool loop in the MAS orchestrator would raise on the first turn, causing an unhandled exception visible in the dashboard but no backtest progress.
- **Recommended fix:** In `ClaudeClient.generate_content`, detect Opus 4.7 (or any model where `"4-7"` appears in the name) and replace `{"type": "enabled", "budget_tokens": N}` with `{"type": "adaptive"}` plus `output_config={"effort": "high"}`. Add the same guard in `multi_agent_orchestrator._call_agent_with_tools`.

---

### Pattern 2: budget_tokens deprecated on Opus 4.6 and Sonnet 4.6
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking — Warning box
- **What the doc says:** "`thinking.type: enabled` and `budget_tokens` are deprecated on Opus 4.6 and Sonnet 4.6 and will be removed in a future model release. Use `thinking.type: adaptive` with the effort parameter instead."
- **Status:** ⚠️ partial — functional today, but all current production models (Opus 4.6, Sonnet 4.6) are on the deprecated path
- **Evidence:**
  - `backend/agents/multi_agent_orchestrator.py:950-953` — `type: "enabled"` with `budget_tokens: 2048`
  - `backend/agents/llm_client.py:624-626` — same pattern
  - `backend/agents/orchestrator.py:91,96,101,108` — four Gemini thinking configs also use `type: "enabled"` (these are Gemini-specific via `GeminiClient`, so not directly affected, but the pattern is identical)
  - `backend/config/settings.py:31-35` — `enable_thinking`, `thinking_budget_critic/moderator/risk_judge/synthesis` all still expressed as `budget_tokens` semantics
- **Deviation:** No migration to adaptive mode has been done on any Claude-facing call.
- **Risk:** API removal in a future release will break all Claude thinking calls simultaneously with no advance warning in logs. Ops impact: both MAS tool loop and `ClaudeClient` stop thinking.
- **Recommended fix:** Migrate all Claude thinking calls to `thinking={"type": "adaptive"}` plus `output_config={"effort": "high"}` (or appropriate tier). Keep Gemini paths with `type: "enabled"` — Gemini is unaffected by the Anthropic deprecation.

---

### Pattern 3: temperature must be 1 with manual thinking on Claude
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — API constraints
- **What the doc says:** When `type: "enabled"` is passed to Claude, `temperature` must be 1. The API enforces this.
- **Status:** ✅ correct in ClaudeClient; ❗ incorrect in MAS orchestrator
- **Evidence:**
  - `backend/agents/llm_client.py:628` — `kwargs["temperature"] = 1` is set when a thinking config is detected. Correct.
  - `backend/agents/multi_agent_orchestrator.py:944-954` — `client.messages.create` is called with `thinking={"type":"enabled","budget_tokens":2048}` but no `temperature` argument at all. The Anthropic client defaults `temperature` to something other than 1, so this call may 400 or behave incorrectly.
- **Deviation:** MAS orchestrator bypasses `ClaudeClient` (calls `client.messages.create` directly on the raw `anthropic.Anthropic()` client at line 944) and does not set `temperature=1`.
- **Risk:** Potential 400 on every tool-loop turn when thinking is active in the MAS.
- **Recommended fix:** Add `temperature=1` to the `client.messages.create` call in `_call_agent_with_tools` when `thinking` is included, or route through `ClaudeClient` which already handles this correctly.

---

### Pattern 4: Adaptive thinking — the correct mode for Opus 4.6 / Sonnet 4.6
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking — "How to use adaptive thinking"
- **What the doc says:** `thinking={"type": "adaptive"}` combined with `output_config={"effort": "high"}` is the recommended replacement. No `budget_tokens` needed.
- **Status:** ❌ missing — adaptive mode is not used anywhere in the codebase
- **Evidence:** `grep -rn 'type.*adaptive\|adaptive.*thinking'` returns zero matches in `backend/`.
- **Deviation:** All thinking config is manual mode only.
- **Risk:** None today (manual mode still works on 4.6), but blocks future model upgrade path.
- **Recommended fix:** Introduce an `_adaptive_thinking_config()` helper in `ClaudeClient` that emits `{"type": "adaptive"}` + `output_config`. Toggle by model name.

---

### Pattern 5: effort parameter via output_config
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/effort — "Basic usage"
- **What the doc says:** Effort is passed as `output_config={"effort": "high"|"medium"|"low"|"xhigh"|"max"}`. Supported on Opus 4.7, Opus 4.6, Sonnet 4.6, Opus 4.5. No beta header required.
- **Status:** ❌ missing — `output_config` is not referenced anywhere in `backend/`
- **Evidence:** `grep -rn 'output_config'` across `backend/` returns zero matches. The `effort` key appears only in comments inside `agent_definitions.py` (conceptual description, not API call).
- **Deviation:** Effort parameter is entirely absent from all Claude API calls. All calls omit it, defaulting to `"high"` implicitly.
- **Risk:** Low for current workloads (high is the default), but planning agents and evaluators that run frequently could benefit from `"medium"` for cost reduction, and the agentic tool loop would benefit from explicit `"xhigh"` for quality.
- **Recommended fix:** Add `output_config={"effort": effort_level}` to `ClaudeClient.generate_content` and the MAS tool loop. Expose an `effort` field on `AgentConfig` (alongside `max_tokens`) so each agent role can declare its own tier.

---

### Pattern 6: xhigh effort — Opus 4.7 only
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/effort — Effort levels table
- **What the doc says:** `xhigh` is available only on Claude Opus 4.7. Passing it to other models produces undefined behavior or an error.
- **Status:** N/A — effort is not passed at all, so no risk today
- **Evidence:** No `output_config` usage found.
- **Risk:** If effort is added without model gating, setting `xhigh` on Opus 4.6 would be an API violation.
- **Recommended fix:** Gate `xhigh` and `max` effort levels behind a model-version check in whatever helper introduces effort support.

---

### Pattern 7: tool_choice constraint with thinking enabled
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — "Tool Use Integration"
- **What the doc says:** With thinking enabled, only `tool_choice: {"type": "auto"}` or `{"type": "none"}` are valid. `tool_choice: {"type": "any"}` or forced selection returns 400.
- **Status:** ✅ correct (by omission)
- **Evidence:** `multi_agent_orchestrator.py:944-954` does not pass `tool_choice` at all. The API default is `auto`, which is compliant.
- **Deviation:** None.
- **Risk:** None currently. If a future change adds forced tool selection while thinking is on, it will 400.

---

### Pattern 8: Thinking block preservation across tool turns
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — "Thinking Block Preservation Rules"
- **What the doc says:** "When continuing conversations with tool results: assistant content MUST include original thinking block. Consequences of omitting thinking blocks: Error raised."
- **Status:** ✅ correct
- **Evidence:** `multi_agent_orchestrator.py:1009` — `messages.append({"role": "assistant", "content": response.content})`. `response.content` is the full content list from the Anthropic API, which includes the thinking block as returned. This passes the full block unmodified.
- **Deviation:** None. The assistant content is appended wholesale.
- **Risk:** Low. Observation masking at line 1015 compresses older messages — if masking strips thinking blocks from prior turns, a future model validation may flag this. The masking code should be verified not to remove thinking-typed content blocks.

---

### Pattern 9: display field — omitted vs summarized defaults differ by model
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — "Controlling thinking display"
- **What the doc says:** Opus 4.7 defaults `display` to `"omitted"` (silent change from Opus 4.6 default of `"summarized"`). To receive thinking summaries on Opus 4.7, set `display: "summarized"` explicitly.
- **Status:** ❌ missing — display field is never set anywhere
- **Evidence:** `grep -rn 'display.*summarized\|display.*omitted'` returns zero matches in `backend/`.
- **Deviation:** On Opus 4.6 the default is summarized so thoughts surface in `LLMResponse.thoughts`. On Opus 4.7 the default would be omitted, silently zeroing the thoughts field with no error.
- **Risk:** If the stack upgrades to Opus 4.7, `response.content` thinking blocks will have an empty `thinking` field. `ClaudeClient` at line 637-638 reads `block.thinking` directly — it will return empty string rather than error, silently killing all thought capture and any downstream features that display thinking in the UI.
- **Recommended fix:** Add `"display": "summarized"` to the thinking dict in `ClaudeClient` when the goal is to capture thought content.

---

### Pattern 10: Interleaved thinking — beta header no longer needed for 4.6+
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking — Interleaved Thinking table
- **What the doc says:** Opus 4.7, Opus 4.6, and Sonnet 4.6 with adaptive thinking have interleaved thinking auto-enabled. No beta header required. Older models (Opus 4.5, Sonnet 4.5) require `interleaved-thinking-2025-05-14`.
- **Status:** ✅ correct (by omission)
- **Evidence:** No beta headers are sent in `multi_agent_orchestrator.py`. Since the active models are Opus 4.6 and Sonnet 4.6, interleaved thinking is auto-active when thinking is enabled. No stale beta header overhead.
- **Deviation:** None.

---

### Pattern 11: Task budgets — beta header required, Opus 4.7 only
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/task-budgets
- **What the doc says:** `task_budget` in `output_config` requires the `task-budgets-2026-03-13` beta header. Supported only on Opus 4.7. Minimum value: 20,000 tokens. Advisory, not a hard cap.
- **Status:** N/A — task budgets are not used
- **Evidence:** No `task_budget` or `task-budgets-2026-03-13` found in `backend/`.
- **Deviation:** Not applicable. The feature is beta and Opus 4.7 is not yet in the stack.
- **Risk:** Future: if the agentic tool loop is extended to long-horizon tasks and Opus 4.7 is adopted, task budgets should be added to prevent runaway token spend. The minimum 20,000 token floor means budgets below that will 400.

---

### Pattern 12: Settings expose Gemini-only thinking fields to Claude paths
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking (Claude API shape) vs `backend/config/settings.py`
- **What the doc says:** Claude extended thinking is controlled via the `thinking` dict in the messages API call. Gemini thinking uses a different mechanism through Vertex AI generation config.
- **Status:** ⚠️ partial — settings are used only for Gemini; Claude thinking is hardcoded separately
- **Evidence:**
  - `backend/config/settings.py:32-35` — `enable_thinking`, `thinking_budget_critic`, `thinking_budget_moderator`, `thinking_budget_risk_judge`, `thinking_budget_synthesis` are all wired to Gemini paths in `orchestrator.py:85-110`.
  - `backend/agents/llm_client.py:622-628` — Claude thinking budget is read from `generation_config["thinking"]["budget_tokens"]`, a caller-supplied dict, not from `settings`.
  - `backend/agents/multi_agent_orchestrator.py:952` — hardcoded `budget_tokens: 2048`, not from settings.
- **Deviation:** Claude thinking budget is not configurable via environment. Changing it requires a code edit. There is no `thinking_budget_mas` setting.
- **Risk:** Operational: cannot tune MAS thinking depth at deploy time. If 2048 tokens is insufficient for complex reasoning tasks, tuning requires a code push.
- **Recommended fix:** Add `mas_thinking_budget: int = Field(2048, ...)` and `mas_thinking_effort: str = Field("high", ...)` to `Settings`, and reference them in the MAS orchestrator.

---

### Pattern 13: Prompt caching interaction with thinking mode changes
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking — "Prompt caching"
- **What the doc says:** "Switching between `adaptive` and `enabled`/`disabled` thinking modes breaks cache breakpoints for messages. System prompts and tool definitions remain cached."
- **Status:** N/A — all Claude calls currently use the same `type: "enabled"` mode, so no cache invalidation from mode switching occurs.
- **Evidence:** `ClaudeClient.generate_content` always applies `cache_control: ephemeral` to the system prompt (`llm_client.py:602-610`). Tool definitions are not cached separately.
- **Deviation:** None currently. Risk materializes if adaptive mode is introduced alongside existing enabled-mode calls in multi-turn conversations.

---

### Pattern 14: PlannerAgent and EvaluatorAgent bypass ClaudeClient
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/extended-thinking (all patterns)
- **What the doc says:** N/A — this is a structural observation derived from reviewing all patterns above
- **Status:** ⚠️ partial — these agents do not benefit from any thinking/effort guardrails
- **Evidence:**
  - `backend/agents/planner_agent.py:115-122` — calls `self.client.messages.create` directly on the raw `anthropic.Anthropic()` client. No thinking, no effort, no prompt caching, no temperature guard.
  - `backend/agents/evaluator_agent.py:84-99` — uses Gemini (Vertex `GenerativeModel`) by default, not Claude. No thinking config at all.
- **Deviation:** `PlannerAgent` and `EvaluatorAgent` are outside the `ClaudeClient` abstraction that centralizes prompt caching, thinking guards, and usage tracking.
- **Risk:** Any thinking/effort fix applied to `ClaudeClient` will not reach these agents. They also leak cost tracking and cache metrics.
- **Recommended fix:** Route both agents through `ClaudeClient` (or at minimum through `make_client()`). This is a prerequisite for systematic thinking/effort adoption.

---

## Summary matrix

| # | Pattern | Status | 400 risk on Opus 4.7? | Priority |
|---|---------|--------|----------------------|----------|
| 1 | `type: "enabled"` rejected on Opus 4.7 | ❗ incorrect | YES — immediate 400 | P0 |
| 2 | `budget_tokens` deprecated on 4.6/4.6 | ⚠️ partial | No (still works) | P1 |
| 3 | temperature=1 missing in MAS tool loop | ❗ incorrect | Potential 400 today | P0 |
| 4 | Adaptive mode not implemented | ❌ missing | N/A | P1 |
| 5 | `output_config` / effort absent | ❌ missing | No | P2 |
| 6 | xhigh effort Opus 4.7 only | N/A | No (not used) | P3 |
| 7 | tool_choice constraint | ✅ correct | No | — |
| 8 | Thinking block preservation | ✅ correct | No | — |
| 9 | display default changes on Opus 4.7 | ❌ missing | Silent data loss | P1 |
| 10 | Interleaved thinking beta header | ✅ correct | No | — |
| 11 | Task budgets (Opus 4.7 beta) | N/A | No | P3 |
| 12 | Settings don't cover Claude thinking | ⚠️ partial | No | P2 |
| 13 | Cache invalidation on mode switch | N/A | No | — |
| 14 | Planner/Evaluator bypass ClaudeClient | ⚠️ partial | No direct 400 | P1 |

---

## Gap mapping to phase-4.14 MUST-FIX items

Phase-4.14 MUST-FIX items are not directly referenced in the current masterplan snapshot, but the following patterns map to the risk categories typically tracked in that phase:

- **MF-THINKING-01 (Pattern 1, 3):** Any call with `type: "enabled"` to a model that is Opus 4.7 or later will 400 silently in production. Gating logic must be in place before the live cost tier is finalized.
- **MF-THINKING-02 (Pattern 2, 4):** Deprecation of `budget_tokens` on current production models (Opus 4.6, Sonnet 4.6) means migration to adaptive mode is on the critical path for the May 2026 go-live.
- **MF-THINKING-03 (Pattern 9):** The `display` field omission means thought capture breaks silently on Opus 4.7. If the live tier moves to 4.7 and any dashboard feature relies on `LLMResponse.thoughts`, it will render empty with no error.
- **MF-THINKING-04 (Pattern 14):** `PlannerAgent` and the autonomous harness loop call Anthropic directly, bypassing all safeguards. These are in the critical path for the harness (phase-3 and phase-4 planning cycles).

---

## References

- Extended thinking: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- Adaptive thinking: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
- Effort parameter: https://platform.claude.com/docs/en/build-with-claude/effort
- Task budgets: https://platform.claude.com/docs/en/build-with-claude/task-budgets
- Internal: `backend/agents/llm_client.py` (ClaudeClient, lines 536-672)
- Internal: `backend/agents/multi_agent_orchestrator.py` (tool loop, lines 920-1036)
- Internal: `backend/agents/orchestrator.py` (Gemini thinking configs, lines 85-110)
- Internal: `backend/agents/planner_agent.py` (direct Anthropic client, lines 35-122)
- Internal: `backend/config/settings.py` (thinking budget fields, lines 31-35)
- Internal: `backend/config/model_tiers.py` (active model names)
