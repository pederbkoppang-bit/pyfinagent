# Claude Remote / Max Programmatic Handoff -- Feasibility Study

**Phase:** 19.0
**Date:** 2026-04-26
**Operator question:** "Can pyfinagent push heavy-lifting work to Claude remote
agents using my Max ($200/mo) flat-fee subscription so it doesn't cost
additional cash?"
**Status:** Decision document. NO implementation this phase.
**Research brief:** `handoff/current/phase-19.0-research-brief.md` (7
external sources read in full, 17 URLs, 8 internal files inspected,
gate_passed: true).

---

## Recommendation

**REJECT** the literal hypothesis (use Max OAuth for backend dispatch) -- it
violates Anthropic's Terms of Service as of April 4, 2026.

**ACCEPT** the underlying intent (offload long-context work to Claude) via
the standard Anthropic API with the existing `ANTHROPIC_API_KEY`. 1M-context
calls on Sonnet 4.6 / Opus 4.7 are now standard-priced (no surcharge), so
the cost objection from the prior turn was overstated -- a 300K-token
synthesis call costs ~$0.90, fitting comfortably inside the existing
`$5/day` provider budget.

The right next step is a 0.5-cycle spike adding an `enable_1m_context` flag
to `ClaudeClient` and removing the Layer-1 compaction guard, then measuring
whether the un-truncated synthesis output is materially better.

---

## TL;DR

| Question | Answer |
|----------|--------|
| Can we use Max OAuth from the FastAPI backend? | NO. ToS violation since 2026-04-04. |
| Does Claude Max include programmatic API quota? | NO. Max covers interactive Claude Code / Claude.ai / Claude Desktop only. |
| Is 1M context still expensive? | NO -- standard API rates apply on Sonnet 4.6 / Opus 4.7 as of 2026. No `extended-context-1m-2025-08-07` beta header surcharge. |
| Should we add long-context routing? | YES, via direct `ClaudeClient` with `ANTHROPIC_API_KEY`. |
| What's the spike cost? | 0.5 cycles for the flag + dispatch. 3 cycles for full integration across synthesis + skill_optimizer + directive_rewriter. |
| What's the budget impact? | Marginal. ~$2-5/day at expected duty cycle, well within the existing `$5/day` cap. |

---

## What Anthropic's ToS Now Says (cited)

The Register, 2026-02-20: "Anthropic Clarifies Third-Party Tool Ban". Per
that piece + the official `code.claude.com/docs/en/agent-sdk/overview`
page (accessed 2026-04-26), enforcement against using `CLAUDE_CODE_OAUTH_TOKEN`
in any third-party app began January 2026 and was fully applied April 4, 2026.
The Agent SDK doc explicitly says programmatic use requires
`ANTHROPIC_API_KEY`, not subscription OAuth.

**pyfinagent's FastAPI backend IS a third-party app.** There is no
ToS-compliant path for routing automated dispatches through the operator's
Max subscription. Doing so risks account suspension.

## What Claude Max Actually Includes

| Surface | Covered by Max ($200) | Per-token billing |
|---------|------------------------|----|
| Claude Code (desktop CLI) | YES | NO (counts against 5h window) |
| Claude.ai (web) | YES | NO |
| Claude Desktop app | YES | NO |
| Anthropic API direct (`api.anthropic.com`) | NO | YES (per-token) |
| Claude Agent SDK (programmatic) | NO | YES (per-token; uses API key) |
| `claude` CLI in headless `--bare` mode | NO | YES (requires `ANTHROPIC_API_KEY`) |

**Max rate limits on the covered surfaces (current 2026-04-26):** 200-800
prompts per 5-hour rolling window depending on tier; shared across all
first-party interfaces. Heavy programmatic use of Claude Code (even if
allowed) would cannibalize the operator's interactive budget.

## What 1M Context Now Costs on the Anthropic API

Per `platform.claude.com/docs/en/about-claude/pricing` (2026-04-26):

- Sonnet 4.6: $3 / Mtok input, $15 / Mtok output. Includes 1M context. **No surcharge.**
- Opus 4.7: $5 / Mtok input, $25 / Mtok output. Includes 1M context. **No surcharge.**
- Batch API: 50% discount on either model for non-urgent jobs.

Working examples for pyfinagent:
- 28-skill synthesis call, ~300K tokens input, ~5K output:
  - Sonnet 4.6: 0.300 * $3 + 0.005 * $15 = **$0.975**
  - Batch: **$0.49**
- Cross-cycle directive rewrite, ~500K tokens (60 raw briefs):
  - Sonnet 4.6: 0.500 * $3 + 0.010 * $15 = **$1.65**
  - Batch: **$0.83**

Daily duty cycle of 5 such calls = ~$5/day. Fits the existing daily cap
exactly; operator may want to bump to $7-10/day.

**Latency:** 1M-context calls take 15-90 seconds. Existing 120s timeout
in `llm_client.py` accommodates.

---

## pyfinagent Jobs that Benefit (ranked by ROI)

ROI = (impact-of-1M-context) / (engineering-effort).

| Rank | Job | Current input | Win with 1M | Engineering | Impact |
|------|-----|---------------|-------------|-------------|--------|
| 1 | Layer-1 synthesis pipeline (`backend/agents/orchestrator.py:806-870`) | Compacted to ~30K chars | All 28 raw enrichments + full RAG memory verbatim | Remove compaction guard; conditional on `enable_1m_context` | HIGH (synthesis quality is the system's bottleneck) |
| 2 | Skill optimizer cross-skill (`backend/agents/skill_optimizer.py`) | One skill at a time | All 28 skill prompts + recent outcomes in one call -> propose system-wide refactors | Add `optimize_global()` method | HIGH (catches cross-skill patterns) |
| 3 | Directive rewriter (`backend/meta_evolution/directive_rewriter.py:100-122`) | 6 scalar aggregates | Full text of last 60 research briefs | Replace `_summarize_brief_signals()` aggregation with raw-text concat | MEDIUM (directive proposals are weekly, not daily) |
| 4 | Outcome tracker reflection (`backend/services/outcome_tracker.py`) | Price/return scalars only | Full decision trace + bull/bear debate transcripts in agent_memories writes | Plumb decision traces through to LLM call | MEDIUM (better long-term memory) |
| 5 | Deep-dive agent (`backend/agents/skills/deep_dive_agent.md`) | Per-ticker truncated | 10-K + earnings call + 5y filings + 100 news headlines in one shot | Update prompt + remove input truncation | MEDIUM (already passable at 200K) |

## Jobs that Do NOT Benefit

Explicit anti-recommendations:

1. **Debate / risk debate** -- dialectic flow; not context-limited. 200K is plenty.
2. **Cron allocator / provider rebalancer / kelly allocator** -- pure arithmetic, no LLM at all.
3. **Paper trader / portfolio manager** -- deterministic Kelly + risk judge; no LLM in hot path.
4. **Perf optimizer** -- regression analysis, no LLM.
5. **Morning/evening digest** -- Gemini 2.0 Flash already has 1M context for free; no benefit from Anthropic 1M.
6. **Per-ticker analysis (28 Layer-1 calls)** -- each fits in 200K easily; switching to 1M would just waste tokens.
7. **Backtest parameter optimization** -- compute-bound, not context-bound.

---

## Recommended Architecture

Single-file addition: extend the existing `make_client()` factory in
`backend/agents/llm_client.py` rather than introduce a new module.

```python
# Sketch only -- DO NOT IMPLEMENT THIS PHASE.
def make_client(
    provider: str = "anthropic",
    *,
    model: str | None = None,
    enable_1m_context: bool = False,
) -> LLMClient:
    if provider == "anthropic" and enable_1m_context:
        # Claude Sonnet 4.6 / Opus 4.7 -- 1M context standard-priced.
        # No special header required as of 2026.
        return ClaudeClient(
            model=model or "claude-sonnet-4-6",
            max_input_tokens=1_000_000,
        )
    ...
```

**Auth:** Existing `ANTHROPIC_API_KEY` from `backend/.env`. No OAuth, no
subprocess, no Max credential.

**Sync vs async:** Sync wrapped in `asyncio.to_thread` from FastAPI handlers
(matches existing pattern in `_call_llm_for_rewrite`).

**Retries / fallback:** Existing 3-attempt retry + Gemini fallback chain
already in `llm_client.py` works unchanged. If a 1M call fails, fallback to
Gemini Pro (also 1M context, free-tier-friendly).

## Recommended Budget Tracker

Add a row to `.claude/provider_budget.yaml`:

```yaml
# Sketch only -- DO NOT IMPLEMENT THIS PHASE.
- name: anthropic_long_context
  priority_weight: 5
  min_floor_usd: 0.50
  max_ceiling_usd: 2.00
  enabled: false   # operator must opt-in per cycle
  notes: 1M-context Claude calls (Sonnet 4.6 / Opus 4.7). Disabled by default.
```

Tracking stays in `cost_tracker.py` (USD per call). NO new "prompts per 5h"
tracker -- that's a Max-subscription concept, irrelevant here.

---

## Engineering Cost Estimate

| Cycle | Scope | Effort |
|-------|-------|--------|
| Spike (0.5) | Add `enable_1m_context` flag to `make_client()`. Smoke-call from a script. Measure latency + token count. | ~0.5 day |
| Cycle A | Wire into Layer-1 synthesis (`orchestrator.py`); A/B compare against compacted baseline. | 1 day |
| Cycle B | Wire into skill optimizer global-pass; produce one cross-skill-refactor proposal. | 1 day |
| Cycle C | Wire into directive rewriter; first weekly Sunday cron run with raw briefs. | 1 day |
| Cycle D | Observability + budget tracking + ceiling clamp. | 0.5 day |

Total: ~3.5 days for full integration. The spike (0.5 day) is the gate to decide whether to proceed.

## Risks

1. **Rate limits on the existing API key (MEDIUM)** -- 1M-context calls count against Tier 1 RPM/TPM. Mitigation: reuse existing Anthropic API key (already past warm-up); spread calls via existing 5-hour cron buckets.
2. **Latency variance (MEDIUM)** -- 15-90s per call. Mitigation: existing 120s timeout in `llm_client.py:_call_anthropic()` already covers; for synchronous flows, frontend shows progress spinner.
3. **Budget overrun (LOW)** -- ceiling clamp in `provider_budget.yaml` + daily cap in `cost_budget_api.py` already prevent runaway spend. Worst case = $2/day extra.
4. **Tokenizer change on Opus 4.7 (LOW)** -- ~35% more tokens than Sonnet 4.6 for same input. Mitigation: prefer Sonnet 4.6 for cost-controlled paths; reserve Opus 4.7 for high-value monthly jobs.
5. **ToS revocation risk (LOW)** -- Anthropic could change pricing on 1M context at any time. Mitigation: monitor pricing page in a quarterly audit cron; switch to Gemini 2.0 Pro 1M (free) as fallback if Anthropic raises.
6. **NOT a risk: Max ToS violation** -- this design uses API key, not OAuth. No ToS exposure.

---

## Decision

**Proceed** with the 0.5-cycle spike. Gates the full 3.5-day integration on
the spike's measurable A/B win.

**Do not** pursue any approach that uses Max OAuth, `CLAUDE_CODE_OAUTH_TOKEN`,
or `claude -p` subprocess invocation from the backend -- all would violate ToS.

The Claude Max subscription remains valuable for the operator's
interactive Claude Code session (where this very harness runs) but cannot
be the cost-saving mechanism for backend long-context dispatch. Budget the
$2-5/day API cost explicitly.

## Cross-references

- `handoff/current/phase-19.0-research-brief.md` (7 external sources, 17 URLs, 8 internal files)
- `backend/agents/llm_client.py:1155` (existing ClaudeClient)
- `backend/agents/orchestrator.py:806-870` (compaction guard to remove conditionally)
- `backend/meta_evolution/directive_rewriter.py:100-122` (`_summarize_brief_signals` aggregation)
- `.claude/provider_budget.yaml` (where the new provider row would go)
- `backend/services/cost_budget_api.py` (existing $5/day cap)
- The Register 2026-02-20 -- ToS clarification source
- `code.claude.com/docs/en/agent-sdk/overview` -- Agent SDK auth requirement
- `platform.claude.com/docs/en/about-claude/pricing` -- 2026 1M-context pricing

## Out of scope for THIS phase

- Any code changes (this is a decision document only).
- The spike itself -- if approved, opens as a NEW masterplan step (phase-19.1).
- Migration of existing Layer-1 synthesis to 1M (the spike must prove the win first).
- Gemini 2.0 Pro 1M evaluation as alternative provider (separate study if Anthropic pricing changes).
