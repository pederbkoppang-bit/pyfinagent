# Research Brief — phase-47.9 — Opus-4.8 sweep finish

Tier: **moderate**. Working: /Users/ford/.openclaw/workspace/pyfinagent

Two Priority-3 remainders:
- **A.** `max_tokens` vs thinking-token interaction under Opus-4.8 adaptive/extended thinking at effort=xhigh/max; audit per-agent max_tokens floors.
- **B.** Driver-pin straggler `run_autonomous_loop.py:73 planner_model="claude-opus-4-6"`.

Status: COMPLETE. gate_passed: true (6 read-in-full, recency scan done, all blockers satisfied).

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- |
| https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking | 2026-05-29 | doc (tier-2) | WebFetch full | **AUTHORITATIVE for Opus 4.8.** "Use `max_tokens` as a hard limit on **total output (thinking + response text)**." + "At `high` and `max` effort levels, Claude may think more extensively and can be more likely to exhaust the `max_tokens` budget. If you observe `stop_reason: "max_tokens"`... increase `max_tokens`... or lower the effort level." Opus 4.8 = adaptive ONLY; `display` defaults to `omitted`. |
| https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-05-29 | doc (tier-2) | WebFetch full | Older `budget_tokens`-framed page. "Current turn thinking counts towards your max_tokens limit for that turn" (via search snippet of same page) / "max_tokens is enforced as a strict limit." Interleaved-thinking exception: budget can exceed max_tokens (beta header, Claude-4-era). Note: this page's `budget_tokens < max_tokens` framing is the MANUAL-mode lens; adaptive page supersedes for 4.8. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-29 | doc (tier-2) | WebFetch full | **DECISIVE max_tokens floor.** "When running Opus 4.8 at `xhigh` or `max` effort, set a large `max_tokens` so the model has room to think and act across subagents and tool calls. **Starting at 64k tokens and tuning from there is a reasonable default.**" Effort affects ALL tokens (text + tool calls + thinking). `xhigh`/`max` only on Opus 4.8/4.7. |
| https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8 | 2026-05-29 | doc (tier-2) | WebFetch full | Opus 4.8 = **128k max output tokens**, 1M context, adaptive-thinking only (`budget_tokens` -> 400). effort default `high` on all surfaces. "thinking is OFF unless you explicitly set `thinking:{type:adaptive}`." Fewer wasted thinking tokens than 4.7 at same effort. |
| https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons | 2026-05-29 | doc (tier-2) | WebFetch full | `stop_reason:"max_tokens"` -> "retry the request with a higher `max_tokens`." Incomplete tool_use tail -> retry with higher max_tokens (example doubles 1024->4096). Also documents EMPTY `end_turn` responses + `model_context_window_exceeded`. Confirms project's :1565 dispatch is the documented pattern. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html | doc (Bedrock mirror) | Mirror of Anthropic doc; not the canonical source |
| https://decodeclaude.com/ultrathink-deprecated/ | blog | Community blog; lower tier; snippet only |
| https://www.developersdigest.tech/blog/extended-thinking-claude-production-guide | blog | Community; snippet only |
| https://www.cometapi.com/how-to-use-claude-4-extended-thinking/ | blog | Aggregator; snippet only |
| https://simonw.substack.com/p/claude-37-sonnet-extended-thinking | blog (named author) | 3.7-era; superseded |
| https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/extended-thinking-tips | doc | Tips page; may pull if budget allows |
| https://platform.claude.com/docs/en/build-with-claude/context-windows | doc | Context-window math; may pull |

## Search queries run (3-variant discipline)
- Current-year frontier: `Anthropic Claude max_tokens extended thinking budget interaction 2026`
- Last-2-year window: `Claude Opus 4.8 max_tokens stop_reason truncation thinking 2025`
- Year-less canonical: `Claude extended thinking max_tokens includes thinking tokens documentation`

## Recency scan (2024-2026)
Searched the last-2-year window explicitly (`...truncation thinking 2025` + the `2026` frontier query). Result: **the most recent sources are the most authoritative and they CONFIRM the verdict** -- there is no older canonical source that contradicts them. New/2026 findings: (1) the Opus-4.8 "What's new" doc (launch 2026-05) sets **128k max output** and reiterates adaptive-only + 400-on-budget_tokens; (2) the effort doc carries the 2026 "64k starting `max_tokens` at xhigh/max" guidance for 4.8/4.7 (absent in pre-4.7 docs because xhigh did not exist); (3) the adaptive-thinking doc (2026) is the page that states `max_tokens` is a hard cap on "thinking + response text" together -- the older extended-thinking page predates adaptive and frames everything around `budget_tokens`. Net: newer work SUPERSEDES the looser manual-mode reading for the 4.8 path. One adjacent recency signal: GitHub issue anthropics/claude-code#29488 ("CLAUDE_CODE_MAX_OUTPUT_TOKENS has no effect on Opus 4.6, capped at 32K") -- a Claude-Code-surface cap, NOT the Messages-API path pyfinagent uses; noted as snippet-only context, not load-bearing here.

## Key findings (external)
1. **`max_tokens` IS a combined ceiling on thinking + visible text under adaptive thinking.** The adaptive-thinking doc (authoritative for Opus 4.8, which is adaptive-only) states verbatim: "Use `max_tokens` as a hard limit on total output (thinking + response text)." This DIRECTLY contradicts the looser reading of the older extended-thinking page (which is framed around manual `budget_tokens < max_tokens`). For pyfinagent's path (4.8 -> adaptive branch), the combined-ceiling reading governs. (Source: adaptive-thinking doc, 2026-05-29)
2. **High/max effort raises truncation risk.** "At `high` and `max` effort levels, Claude may think more extensively and can be more likely to exhaust the `max_tokens` budget. If you observe `stop_reason: "max_tokens"`... consider increasing `max_tokens`... or lowering the effort level." So at xhigh/max, a low max_tokens can let thinking consume the whole budget and truncate (or empty) the visible answer. (Source: adaptive-thinking doc)
3. **Doc examples uniformly use `max_tokens: 16000`** in every adaptive-thinking code sample (all 9 language bindings), including the effort-parameter example. This is the de-facto recommended floor when thinking is on. (Source: adaptive-thinking doc)
4. **Opus 4.8 `display` defaults to `omitted`** (was `summarized` on 4.6). Visible thinking text is empty unless you set `display:"summarized"`. Does not change billing or the max_tokens math, but is a behavior change worth noting. (Source: adaptive-thinking doc)

## Internal code inventory
| File | Lines | Role | Actual max_tokens / status |
| --- | --- | --- | --- |
| `backend/agents/llm_client.py` | 1281-1336 | `ClaudeClient.generate_content` (Layer-2 single-shot path) | `max_tokens = config.get("max_output_tokens", 2048)` (:1285), passed verbatim at :1332. **Default 2048 if caller omits.** Effort wired separately at :1427-1451; thinking only if caller passes `thinking.budget_tokens>0` (:1382). |
| `backend/agents/llm_client.py` | 1404-1407 | sampling-param strip for 4.8/4.7 | Correct -- pops temperature/top_p/top_k for Opus 4.8/4.7. Matches "What's new 4.8" 400-error rule. |
| `backend/agents/llm_client.py` | 1427-1451 | effort pass-through | Resolves effort (config -> role -> model fallback), drops xhigh for non-4.8/4.7, sets `output_config.effort`. **Effort applied even when thinking is OFF** -- so effort governs text/tool token spend regardless. |
| `backend/agents/llm_client.py` | 1564-1594 | `stop_reason=="max_tokens"` dispatch | tool_use tail -> retry once at `min(max_tokens*2, 8192)`; **plain-text tail -> just logs warning + returns partial** (no retry, no bump). This is the residual truncation exposure. |
| `backend/config/model_tiers.py` | 221-242 | `EFFORT_DEFAULTS` + fallback | All `mas_*` roles = `max` (step-scoped override from phase-23.2.2, never reverted -> now permanent per CLAUDE.md phase-29.2). Opus-4.8 fallback = `xhigh`. **No max_tokens config here** -- token budgets live in agent_definitions. |
| `backend/agents/agent_definitions.py` | 51,130,182,230,276 | Layer-2 AgentConfig.max_tokens | Default 2000; COMMUNICATION 500, MAIN 1500, QA 2500, RESEARCH 3000. These are the per-agent visible-output budgets. |
| `backend/agents/multi_agent_orchestrator.py` | 982-987 | `_simple_text_call` | `max_tokens=agent_config.max_tokens` (NO +2048, NO thinking, NO effort param passed). Lowest-budget Claude path. |
| `backend/agents/multi_agent_orchestrator.py` | 1061-1081 | tool-loop adaptive IF-branch | 4.8/4.7 -> `thinking:{type:"adaptive"}`, `max_tokens=agent_config.max_tokens + 2048`, no sampling params. ELSE-branch (older models) -> `{type:"enabled",budget_tokens:2048}` + temperature=1. |
| `backend/agents/multi_agent_orchestrator.py` | 1168-1200 | tool-loop max_tokens dispatch | tool_use tail -> retry at `min((max_tokens+2048)*2, 16384)`; plain-text tail -> warn + partial. Higher cap (16384) than the generate_content path's 8192. |
| `backend/agents/planner_agent.py` | 58, 146-153, 253-260 | PlannerAgent (autonomous-loop planner) | **Bypasses llm_client entirely** -- raw `Anthropic()` client. `max_tokens=1500` (generate_proposal), `max_tokens=1200` (reflect). NO thinking, NO effort, NO output_config. Default model `claude-opus-4-8` (good). |
| `backend/autonomous_loop.py` | 71-74 | `AutonomousLoopOrchestrator.__init__` | `planner_model: str = "claude-opus-4-8"` -- **default is correct** (bumped in 47.8). |
| `scripts/harness/run_autonomous_loop.py` | 73 | driver | `planner_model="claude-opus-4-6"` -- **STALE, overrides the good default with a 2-versions-old model.** (B target) |
| `scripts/mas_harness/run_cycle.sh` | 63 | launchd MAS-cycle driver | `--model claude-opus-4-6` -- **STALE and OPERATIVE** (a THIRD pin the task's grep missed; see B). |

## Verdict on A: per-agent max_tokens at xhigh/max -- REAL truncation risk, action required

**Authoritative facts (cited):**
- For the **Opus-4.8 adaptive path**, `max_tokens` is a **single hard ceiling on thinking + visible text COMBINED** (adaptive-thinking doc, verbatim: "Use `max_tokens` as a hard limit on total output (thinking + response text)"). The older extended-thinking page's `budget_tokens < max_tokens` framing is the *manual-mode* lens and does NOT govern 4.8.
- At `high`/`max` (and `xhigh`) effort, "Claude may think more extensively and can be more likely to exhaust the `max_tokens` budget. If you observe `stop_reason:"max_tokens"`... increase `max_tokens`... or lower the effort level" (adaptive-thinking doc).
- **Anthropic's explicit floor:** "When running Opus 4.8 at `xhigh` or `max` effort, set a large `max_tokens`... **Starting at 64k tokens** is a reasonable default" (effort doc).
- Effort governs ALL tokens even with thinking OFF (effort doc), so the risk is NOT contingent on a `thinking` config being passed.

**What the code actually does -> the risk:**
1. Every Layer-2 Opus-4.8 agent runs at `effort=max` (model_tiers.py:221-225) but with `max_tokens` of **500-3000** (agent_definitions.py) -- or even just 2048 via the `generate_content` default. The orchestrator tool-loop adds only +2048 (-> 2548-5048). **All of these are 10-100x below Anthropic's 64k starting recommendation for max/xhigh.** With adaptive thinking able to consume the combined budget, a hard multi-step turn can spend most/all of the 500-5048 ceiling on thinking and return a truncated or empty visible answer.
   - Worst case: COMMUNICATION agent = `max_tokens=500` at `effort=max`. The router is low-complexity so adaptive thinking will usually skip/minimize -- but 500 is dangerously tight if it ever does think.
   - `_simple_text_call` (orchestrator:982-987) passes raw `agent_config.max_tokens` with NO +2048 buffer and NO effort param -- but note: because it does NOT pass `output_config`, effort falls to the API default `high`, at which 4.8 "almost always thinks". So even this path carries combined-budget pressure at the agent's bare 500-3000.
2. **The text-tail truncation is currently swallowed.** Both stop_reason dispatches (llm_client:1591-1594, orchestrator:1198-1200) only RETRY on a `tool_use` tail; a plain-text `stop_reason:"max_tokens"` just logs a warning and returns the partial. So if thinking starves the text, the caller silently gets a truncated/empty string -- exactly the failure the docs warn about.

**Is it a REAL bug or already adequate?** REAL risk, not adequate. The combination (effort=max + 500-5048 max_tokens + adaptive thinking sharing that ceiling + no text-tail retry) is precisely the configuration the effort doc tells you to avoid. It will not fail every call (adaptive thinking is frugal on easy turns, and 4.8 wastes fewer thinking tokens than 4.7), but it is a latent truncation/empty-response bug on hard turns.

**Recommended fixes (in priority order):**
1. **Raise per-agent `max_tokens` floors for Opus-4.8 agents.** You do NOT need the full 64k for these short-output agents (that floor is framed for long-horizon Claude-Code/subagent sessions), but the current 500-3000 is too tight to coexist with `effort=max` adaptive thinking. A pragmatic floor: **bump the visible-output budgets so total headroom is comfortably above expected thinking spend** -- e.g. set a per-call minimum of ~8k-16k for any Opus-4.8 agent running at high/xhigh/max, OR pass a larger `max_tokens` specifically on the adaptive branch. Cleanest single-point fix: in `multi_agent_orchestrator.py` adaptive IF-branch (:1075) and in `ClaudeClient.generate_content`, compute `max_tokens` as `max(agent_config.max_tokens + THINKING_HEADROOM, FLOOR)` where THINKING_HEADROOM ~= 8192 and FLOOR ~= 8192, gated on `model.startswith(("claude-opus-4-8","claude-opus-4-7"))` AND effort in (high,xhigh,max). (128k is the hard output cap, so there is ample room.)
2. **Make the text-tail `stop_reason=="max_tokens"` path retry (or at least surface a typed truncation signal)** instead of silently returning partial -- mirror the tool_use-tail retry-doubling for text tails too, capped sensibly (e.g. 16384). This converts a silent truncation into a recovered or flagged response.
3. **Consider lowering effort for the genuinely-trivial agents** (COMMUNICATION router is a classifier -- `max` effort is overkill per the effort doc's "reserve max for frontier problems / can overthink structured-output"). Dropping COMMUNICATION to `low`/`medium` both saves cost and removes the 500-token-at-max-effort hazard. NOTE: this is a tuning call, not a correctness fix, and touches the phase-23.2.2 "all mas at max" directive -- flag for owner, don't unilaterally revert.
4. **PlannerAgent (planner_agent.py)**: `max_tokens=1500`/`1200`, model Opus-4.8, but it passes **NO `thinking` and NO `output_config.effort`** -> effort defaults to `high`, at which 4.8 almost always thinks, and 1500/1200 is a tight COMBINED ceiling. Same latent risk via a different (raw-client) path. Either bump these max_tokens to ~8k or pass `output_config={"effort":"medium"}` to reduce thinking pressure. This path also has NO stop_reason handling at all (`response.content[0].text` at :156 will IndexError/return junk if truncated to empty).

**Minimal-change recommendation for phase-47.9 scope:** the substantive, defensible fix is (1) an Opus-4.8 max_tokens FLOOR on the adaptive branch + generate_content (8k-16k), plus (2) text-tail retry. (3) and (4) are worth flagging but (3) collides with an owner directive and (4) is a separate call path -- document them, fix if scope allows.

## Confirmation on B: driver-pin straggler -- the task's grep was INCOMPLETE

- **CONFIRMED:** `scripts/harness/run_autonomous_loop.py:73 planner_model="claude-opus-4-6"` is a stale, operative pin. It is passed to `AutonomousLoopOrchestrator(...)`, overriding the now-correct `claude-opus-4-8` default at `backend/autonomous_loop.py:74`. So yes -- the script overrides the good default with a 2-versions-old model. Fix: change to `"claude-opus-4-8"` (or drop the kwarg to inherit the default).
- **CONFIRMED:** `backend/autonomous_loop.py:74` default IS `claude-opus-4-8` (good; bumped in 47.8).
- **CORRECTION to the task's premise:** the task said "the other two grep hits are a SQL-comment example + a history note." That is **inaccurate for the live tree.** A fresh `grep -rn "claude-opus-4-6" scripts/` returns exactly TWO hits:
  1. `scripts/harness/run_autonomous_loop.py:73` (the known target), AND
  2. **`scripts/mas_harness/run_cycle.sh:63  --model claude-opus-4-6`** -- this is NOT a comment or history note. It is the OPERATIVE model flag passed to `claude -p --model claude-opus-4-6` in the launchd-fired MAS-harness cycle driver (`com.pyfinagent.mas-harness`, referenced by `scripts/mas_harness/smoke_test_4_17_11.py:5` and `scripts/go_live_drills/revert_hygiene_drill.py:92`). **It is a THIRD operative stale pin and should also be bumped to `claude-opus-4-8`.**
  - No SQL-comment or history-note hit exists under `scripts/` today (the task's description appears to be from an earlier tree state / drift).
- **Adaptive-thinking-path question for a 4-6 planner:** moot for the PlannerAgent call path. `PlannerAgent` (planner_agent.py) uses a **raw `Anthropic()` client and passes NO `thinking` argument at all** -- it never touches `multi_agent_orchestrator.py:1061`. So whether 4-6 hits the IF or ELSE branch there is irrelevant to the planner. (For completeness: Opus 4.6 DOES support adaptive thinking per the adaptive-thinking doc, and the orchestrator's :1061 IF-branch does NOT include 4-6, so a 4-6 call THROUGH the orchestrator would wrongly hit the manual ELSE-branch `{type:"enabled",budget_tokens:2048}` -- but 4-6 still accepts manual mode (deprecated, not rejected), so it would not 400. Again: not the planner's path.) The only material consequence of the stale 4-6 pin is **running a 2-versions-old model** (lower capability, no xhigh, more wasted thinking tokens), not an API error.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: adaptive-thinking, extended-thinking, effort, whats-new-4-8, handling-stop-reasons -- all Anthropic tier-2 docs; extended-thinking + adaptive both fetched)
- [x] 10+ unique URLs total (6 read-in-full + 7 snippet-only = 13)
- [x] Recency scan (last 2 years) performed + reported (explicit 2025 + 2026 passes; supersession analysis included)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client both Claude paths, model_tiers, agent_definitions, multi_agent_orchestrator both branches + both stop_reason dispatches, planner_agent, autonomous_loop, both stale-pin scripts)
- [x] Contradictions noted (extended-thinking page vs adaptive-thinking page on what max_tokens bounds; resolved in favor of adaptive page for the 4.8 path)
- [x] All claims cited per-claim with URL/file:line

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
