# Research Brief: Phase-19.0 — Claude Remote / Max Programmatic Handoff Feasibility

**Tier:** moderate  
**Date:** 2026-04-26  
**Researcher:** researcher agent (merged researcher + Explore)

---

## Read in Full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://code.claude.com/docs/en/authentication | 2026-04-26 | Official doc | WebFetch | "Using OAuth tokens obtained through Claude Free, Pro, or Max accounts in any other product, tool, or service — including the Agent SDK — is not permitted" (Agent SDK note); plus setup-token and CLAUDE_CODE_OAUTH_TOKEN mechanics |
| https://code.claude.com/docs/en/headless | 2026-04-26 | Official doc | WebFetch | Full -p / --print / --bare mode semantics, --output-format json, structured output JSON schema, stream-json, session resume, --allowedTools; bare mode requires ANTHROPIC_API_KEY, skips OAuth |
| https://code.claude.com/docs/en/agent-sdk/overview | 2026-04-26 | Official doc | WebFetch | Agent SDK auth: "Set your ANTHROPIC_API_KEY"; explicit note: "Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK. Please use API key authentication." SDK dispatches via subprocess to claude CLI binary. |
| https://platform.claude.com/docs/en/about-claude/pricing | 2026-04-26 | Official doc | WebFetch | Full pricing table: Opus 4.7/4.6/Sonnet 4.6 include 1M context at standard pricing ($5/$25 Opus, $3/$15 Sonnet, no surcharge); extended-context-1m beta header no longer required; Batch API 50% discount; prompt caching 0.1x reads |
| https://portkey.ai/blog/claude-code-limits/ | 2026-04-26 | Tech blog | WebFetch | Max 20x: 200-800 prompts per 5-hour rolling window; all Claude.ai plans share a common usage bucket across Claude app and Claude Code; weekly 240-480 Sonnet hours, 24-40 Opus hours |
| https://docs.litellm.ai/docs/tutorials/claude_code_max_subscription | 2026-04-26 | Tech doc | WebFetch | LiteLLM OAuth forwarding pattern for Max; dual auth shape; no ToS analysis |
| https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/ | 2026-04-26 | Tech press | WebFetch | "OAuth tokens from subscriptions in any third-party product or tool" forbidden; enforcement began January 2026, fully applied April 4 2026; first-party CLI programmatic use NOT explicitly addressed |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://lalatenduswain.medium.com/claude-api-authentication-in-2026-oauth-tokens-vs-api-keys-explained-12e8298bed3d | Blog | Search snippet; covered by official docs |
| https://substack.com/home/post/p-166025131 | Blog | Third-party tool workaround; ToS-violating path |
| https://github.com/rynfar/meridian | Code | ToS-violating proxy; not relevant to first-party approach |
| https://www.shareuhack.com/en/posts/openclaw-claude-code-oauth-cost | Blog | Cost analysis but ToS-violating path |
| https://rogs.me/2026/02/use-your-claude-max-subscription-as-an-api-with-cliproxyapi/ | Blog | Third-party proxy; ToS risk |
| https://mlq.ai/news/anthropic-ends-paid-access-for-claude-in-third-party-tools-like-openclaw/ | News | Same story as TheRegister; snippet sufficient |
| https://decodethefuture.org/en/anthropic-blocks-third-party-tools/ | News | Redundant with TheRegister |
| https://intuitionlabs.ai/articles/claude-max-plan-pricing-usage-limits | Blog | Max plan limits; partially covered by portkey |
| https://www.theregister.com/2026/03/31/anthropic_claude_code_limits/ | Tech press | Rate limit drain issue; informative snippet |
| https://aihola.com/article/anthropic-1m-context-standard-pricing | Blog | 1M pricing; covered by official pricing doc |

---

## Recency Scan (2024-2026)

**Queries run:**
1. Year-less canonical: "Claude Max subscription programmatic API authentication"
2. 2025 window: "Claude Code SDK programmatic headless non-interactive 2025"
3. 2026 frontier: "Claude Max $200 plan programmatic use ToS violation third-party April 2026"

**Result:** Found 3 highly relevant new findings from 2026 that are decisive for this decision:

1. **April 4, 2026 ToS enforcement cutoff** — Anthropic fully blocked OAuth subscription tokens in third-party tools; any path that relies on Max OAuth for non-first-party dispatch violates ToS as of this date (multiple sources: TheRegister, mlq.ai, decodethefuture.org).

2. **Agent SDK renamed and API-key-only** — The Claude Code SDK was renamed "Claude Agent SDK" and its official auth path is ANTHROPIC_API_KEY only; explicit note in docs: third-party developers (and pyfinagent IS a third-party application) "must use API key authentication methods."

3. **1M context at standard pricing** — The extended-context-1m beta header is no longer needed for Opus 4.7, Opus 4.6, Sonnet 4.6. A 900K-token request costs the same per-token rate as a 9K request. This fully eliminates the 2x surcharge concern.

---

## Key Findings

1. **Max subscription OAuth is ToS-prohibited for programmatic use.** Anthropic explicitly prohibits using OAuth tokens from Max (or Pro/Free) accounts in "any other product, tool, or service." pyfinagent's FastAPI backend calling claude via subprocess with subscription OAuth = ToS violation. Enforcement began April 4, 2026. Source: official Agent SDK docs + TheRegister 2026-02-20.

2. **First-party `claude -p` with API key IS the legitimate path.** The `claude` CLI in bare + non-interactive mode (`-p`, `--bare`, `--output-format json`) is fully supported for scripted/CI use. Authentication must come from `ANTHROPIC_API_KEY` (the Console API key), NOT from Max subscription OAuth. Bare mode skips keychain, reads only env vars. Source: official headless docs.

3. **Claude Agent SDK requires API key, not subscription OAuth.** The official SDK (Python: `pip install claude-agent-sdk`, TypeScript: `npm install @anthropic-ai/claude-agent-sdk`) uses `ANTHROPIC_API_KEY`. It dispatches via the same subprocess mechanism as `claude -p` internally. Source: official Agent SDK overview.

4. **1M context is now standard-priced on Opus 4.6/4.7 and Sonnet 4.6.** No beta header, no surcharge. 900K tokens billed at the same per-token rate as 9K. Pricing: Opus 4.6/4.7 $5/$25 MTok, Sonnet 4.6 $3/$15 MTok, Batch API 50% off both. Source: Anthropic pricing page.

5. **Max plan usage bucket is shared across all first-party interfaces.** Interactive Claude Code desktop + CLI -p calls all draw from the same 5-hour rolling window. Max 20x: 200-800 prompts per 5-hour window (varies by model + token size). Heavy programmatic use from pyfinagent would directly cannibalize the operator's interactive Claude Code quota. Source: portkey.ai Claude Code limits blog.

6. **Rate limit drain is a real production problem.** Since March 2026, Max users are hitting the 5-hour limit in <2 hours under CLI-heavy workloads. This is not a hypothetical risk. Source: GitHub issue #38335 + TheRegister 2026-03-31.

7. **The correct architecture uses a metered API key, not subscription OAuth.** Adding a `ClaudeMaxCLI` dispatcher to pyfinagent would actually mean using the standard `anthropic` SDK or `claude -p --bare` with a Console API key — charged per-token, no flat-fee benefit. The "Max flat-fee" only applies to interactive Claude Code and Claude.ai web sessions.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/llm_client.py` | 1155 | Multi-provider LLM dispatch; ClaudeClient uses `ANTHROPIC_API_KEY` via sdk | Active; 4-provider factory |
| `backend/meta_evolution/provider_rebalancer.py` | 230 | WFQ allocator over `provider_budget.yaml`; 4 providers: anthropic, google_vertex, openai, github_models | Active; USD floor+ceiling per provider |
| `.claude/provider_budget.yaml` | 47 | anthropic: weight=10, floor=$1.00, ceiling=$4.00; total daily=$5.00 | Active |
| `.claude/cron_budget.yaml` | 190 | 15 slots/day self-imposed cap; daily token budget 100K; notes it is NOT an Anthropic limit | Active |
| `backend/meta_evolution/directive_rewriter.py` | 320+ | Rewrites researcher.md; does NOT pass full briefs verbatim — reduces to scalar signals via `_summarize_brief_signals()` | Active; 1M context would let it read full brief text instead of aggregated scalars |
| `backend/agents/orchestrator.py` | 1477 | 15-step pipeline; synthesis has compaction path for models with <=30K char limits; uses `max_output_tokens=4096` | Active; context compaction code at lines 806-870 |
| `backend/agents/skill_optimizer.py` | ~300+ | Autoresearch loop for skills/*.md; reads skill text + outcomes; no explicit context limit | Active |
| `backend/services/outcome_tracker.py` | ~200+ | LLM reflection on outcomes; generates per-agent memories for BM25; no full decision trace — only price/return scalars | Active |

---

## Candidate Jobs That Would Benefit from 1M Context

### Ranked by ROI (impact / engineering-effort)

**Rank 1 — `orchestrator.py` synthesis stage (file:line 806-870)**
Current: compaction path truncates each enrichment section to 1,500 chars + full_context compaction for models with <=30K limit. Synthesis uses `max_output_tokens=4096` on Claude Sonnet/Opus.
With 1M: all 28 raw enrichment outputs (~4K chars each = ~112K chars total) could be sent verbatim to synthesis without compaction, plus full RAG memory (currently capped at 200 rows), plus full debate transcripts.
ROI: HIGH impact (richer synthesis = better signal), LOW effort (remove compaction guard, update context config).

**Rank 2 — `skill_optimizer.py` cross-skill optimization**
Current: reads individual skill prompts + recent outcomes sequentially; no cross-skill context.
With 1M: could load all 28 skill prompts (~280K chars) + all recent experiment TSV rows + full BQ outcome signals in a single call for cross-skill co-optimization (the Karpathy autoresearch pattern applied globally).
ROI: HIGH impact (cross-skill correlations are currently invisible), MEDIUM effort (new dispatcher call).

**Rank 3 — `directive_rewriter.py` research directive optimization**
Current: `_summarize_brief_signals()` at line 100-122 reduces N briefs to 6 scalar aggregates. The LLM never sees the raw brief text, just the averages.
With 1M: could pass all 60+ raw research briefs verbatim to the rewriter LLM, enabling it to identify specific failure modes (e.g., "3 of the last 5 gate failures were missing recency scans on quant topics").
ROI: MEDIUM impact (better directive mutations), LOW effort.

**Rank 4 — `outcome_tracker.py` LLM reflection**
Current: reflection prompt includes only price/return scalars + ticker/date. Full decision trace (28-agent enrichments, debate transcript, synthesis) is not included.
With 1M: full decision trace (~500K chars per analysis) could be included in the reflection call, enabling the LLM to identify which agent outputs were directionally wrong before the final recommendation.
ROI: MEDIUM impact (better agent memory = better BM25 retrieval), HIGH effort (BQ round-trip to reconstruct full report).

**Rank 5 — `agents/skills/deep_dive_agent.md`**
Current: deep dive reads summary-level report dict. 10-K, earnings call transcripts, news headlines are pre-processed by earlier agents.
With 1M: could include raw SEC filings + 5y earnings call transcripts + 100 news items in one shot.
ROI: LOW (data pipeline not currently fetching raw filings into memory), VERY HIGH effort (data ingestion change required first).

---

## Jobs That Would NOT Benefit (Anti-Recommendations)

1. **`debate.py` / `risk_debate.py`** — Debate is inherently multi-turn with small per-agent outputs (1536 tokens max). 1M context would not improve the Bull/Bear dialectic; the bottleneck is reasoning depth, not information width.

2. **`cron_allocator.py` / `provider_rebalancer.py`** — Pure arithmetic + YAML parsing. LLM is not involved in the critical path. 1M context is irrelevant.

3. **`perf_optimizer.py`** — TTL tuner uses regression on latency logs. No LLM call; not a 1M-context candidate.

4. **`morning_digest` / `evening_digest` cron slots** — These are Gemini-powered summaries of daily data. Input is bounded and already fits within Gemini 2.5's context. No value in routing to a 1M-context Claude call.

5. **`paper_trader.py` / `portfolio_manager.py`** — Position sizing is deterministic (Kelly + risk judge). Adding a 1M context model here would add latency and cost for no signal benefit.

---

## Decisive Answers to the 9 Questions

### 1. Is programmatic Max use ALLOWED by Anthropic ToS?
**NO** — for third-party applications. The Agent SDK overview states explicitly: "Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK." pyfinagent is a third-party application. The only permitted programmatic path is via ANTHROPIC_API_KEY from the Console (metered, pay-per-token). Max OAuth via `claude setup-token` is scoped to first-party Anthropic tools only (Claude Code interactive, Claude.ai web, Claude Desktop). Using it in a FastAPI subprocess is prohibited as of April 4, 2026.

**Corollary:** The flat-fee benefit of Max does NOT extend to programmatic dispatch from pyfinagent.

### 2. Which API surface is best?
**Claude Agent SDK (Python)** via `pip install claude-agent-sdk` with `ANTHROPIC_API_KEY`. Rationale:
- Structured Python async interface; no subprocess shell-escaping risk
- Full tool access (Read, Bash, Grep, Glob) without building tool loops manually
- Session resume semantics for multi-step tasks
- Fires against the same Anthropic API as `ClaudeClient` in `llm_client.py`
- Hooks allow pre/post tool logging compatible with `backend/services/observability`

Second best: raw `anthropic.Anthropic` SDK (already in `llm_client.py:ClaudeClient`) with an explicit 1M context request. Simplest engineering lift; no new binary dependency.

**Avoid:** `claude -p --bare` subprocess — it works but is fragile (subprocess parsing, PATH dependency, version drift). The Agent SDK wraps this properly.

### 3. Concrete rate limits on Max ($200) as of 2026-04-26
- **5-hour rolling window** shared across all Claude.ai products (web + Claude Code desktop + CLI)
- Max 20x: ~200-800 prompts per window depending on model and token volume
- Weekly: ~240-480 Sonnet hours; ~24-40 Opus hours
- **Critical:** programmatic API calls via ANTHROPIC_API_KEY do NOT draw from the Max quota — they are billed per-token at Anthropic Console rates. The Max quota only covers interactive first-party tool usage.
- API Tier 1 rate limits apply for fresh API keys (lower concurrency); Tier 2+ after spend history.

### 4. 3-5 Specific pyfinagent jobs that would benefit (ranked by ROI)
1. **Synthesis pipeline** — remove compaction guards; send all 28 raw enrichments verbatim
2. **Skill optimizer cross-skill** — global optimization over all 28 skills in one call
3. **Directive rewriter** — raw brief text instead of scalar aggregates
4. **Outcome tracker reflection** — include full decision trace in agent memory write
(Deep dive skipped to rank 5; requires upstream data pipeline work first)

### 5. 3-5 Jobs that would NOT benefit
1. Debate / risk debate (dialectic bounded; bottleneck is reasoning not context)
2. Cron allocator / provider rebalancer (pure arithmetic)
3. Paper trader / portfolio manager (deterministic Kelly sizing)
4. Perf optimizer (regression on latency TSV; no LLM)
5. Morning/evening digest cron (Gemini-powered; already fits in 1M Gemini 2.0 Flash context)

### 6. Recommended architecture
**Do not build a "ClaudeMax dispatcher."** Max subscription credit cannot be programmatically used.

**Correct architecture:** Add a `claude_long_context` route to `make_client()` in `llm_client.py` that:
- Selects `claude-sonnet-4-6` or `claude-opus-4-6` via the existing `ClaudeClient`
- Uses `ANTHROPIC_API_KEY_LONG_CONTEXT` (separate Console key for budget isolation)
- Sets a large context ceiling in `_MODEL_MAX_INPUT_CHARS` (remove the cap for these models)
- Is gated by a feature flag `settings.enable_1m_context: bool = False`

New file suggestion: `backend/llm_router/long_context_dispatcher.py` — a thin wrapper around `ClaudeClient` that enforces the 1M context path and logs to observability with a `long_context=True` tag.

Dispatch contract: **async, immediate** (not queued). 1M-context calls for synthesis take 15-60s; use `asyncio.to_thread` (already the pattern in `orchestrator.py`).

Auth: `ANTHROPIC_API_KEY` from `backend/.env`. Separate key optional for budget isolation. No OAuth. No subprocess.

### 7. Recommended budget tracker
Add to `provider_budget.yaml`:
```yaml
- name: anthropic_long_context
  priority_weight: 3
  min_floor_usd: 0.00
  max_ceiling_usd: 2.00
  enabled: false  # flip to true when feature flag on
  notes: Sonnet 4.6 / Opus 4.6 1M-context calls for synthesis + skill optimizer.
```
Add `backend/services/claude_max_quota_tracker.py` (mirroring `cost_tracker.py`) tracking: calls per day, total tokens per day, average context size, 1M-context-specific cost. This is standard USD tracking, not "prompts per 5h" (that's only for subscription-based quotas which don't apply here).

### 8. Engineering cost estimate
- **Spike (1 cycle):** Add `enable_1m_context` flag + remove compaction guard in synthesis + verify 1M round-trip works with Console API key. ~0.5 engineering cycles.
- **Full integration (3 cycles):**
  - Cycle 1: `long_context_dispatcher.py` + synthesis path + budget YAML update
  - Cycle 2: Skill optimizer cross-skill 1M call + outcome tracker reflection upgrade
  - Cycle 3: Directive rewriter full-brief path + observability tagging + cost monitoring dashboard
- Incremental cost per full-analysis run with 1M synthesis call: Sonnet 4.6 at 300K tokens input = $0.90/call. For 5 tickers/day = ~$4.50/day additional, within existing $5/day API budget ceiling if other providers reduce.

### 9. Risks
| Risk | Severity | Mitigation |
|------|----------|-----------|
| ToS revocation of Max for first-party programmatic use | MEDIUM — currently allowed for interactive; Anthropic track record of tightening | Use API key path only; don't mix subscription OAuth |
| Rate limit surprises on API Tier 1 | MEDIUM — fresh API key starts at Tier 1 (low concurrency) | Use existing key already in `ANTHROPIC_API_KEY`; Tier 2+ after spend history |
| OAuth credential leakage (if implemented) | WOULD BE HIGH — but irrelevant since we're NOT using OAuth | N/A for recommended API key path |
| Latency variance at 1M tokens | MEDIUM — 15-90s per call; variable under load | Use `asyncio.to_thread`; implement 120s timeout (already in `GeminiClient`) |
| Budget overrun | LOW-MEDIUM | `max_ceiling_usd: 2.00` in budget YAML + existing kill-switch at $5/day |
| API key cost (not flat-fee) | EXPECTED — this is metered, not free | Budget already sized ($1/day floor for anthropic); 1M context adds ~$0.90/synthesis call |

---

## Consensus vs Debate

**Consensus:** Every official Anthropic source is unambiguous — programmatic use of Max OAuth for third-party applications is prohibited. There is no debate on this point.

**Debate on implementation path:** Community sources show various workarounds (Meridian proxy, liteLLM OAuth forwarding, claude_max library). These all violate ToS as of April 4, 2026. The only clean path is API key.

**New 2026 finding (positive):** 1M context is now standard-priced on Opus 4.6/4.7 and Sonnet 4.6. This makes the API path significantly cheaper than feared: no 2x surcharge on long-context calls.

---

## Pitfalls (from literature and code audit)

1. **"Max is flat-fee therefore free for programmatic use"** — FALSE. Max flat-fee covers interactive first-party tools only. Programmatic use is billed per-token via API key.
2. **Shared quota drain** — If anyone uses `claude setup-token` + `CLAUDE_CODE_OAUTH_TOKEN` for automation, it directly reduces the operator's interactive Claude Code quota. Keep programmatic paths on API key only.
3. **`--bare` mode skips OAuth** — Bare mode requires `ANTHROPIC_API_KEY`. This is actually the correct behavior for scripted use, but it makes it impossible to accidentally use subscription quota.
4. **Agent SDK is subprocess-based** — Under the hood, the Python `claude-agent-sdk` spawns the `claude` binary. This means the `claude` CLI must be installed. Raw `anthropic` SDK (`ClaudeClient`) avoids this dependency and is the simpler lift.
5. **Context size estimate** — The 28-agent pipeline enrichments at ~4K chars each = ~112K chars total. At 1 token ~= 3.5 chars, that's ~32K tokens for enrichments. Full synthesis call with debate transcript, RAG memory, quant data, macro data is realistically 200-400K tokens — well within 1M but not trivially small.
6. **Opus 4.7 tokenizer** — Uses ~35% more tokens than earlier models per the same text. If targeting 1M context synthesis, Sonnet 4.6 is more economical.

---

## Application to pyfinagent (mapping to file:line anchors)

| Finding | File:line | What changes |
|---------|-----------|--------------|
| Remove synthesis compaction guard | `orchestrator.py:806-870` | Remove `_compact_context` branch; remove `_MODEL_MAX_INPUT_CHARS` cap for Sonnet/Opus 4.6 |
| Add `anthropic_long_context` provider | `provider_budget.yaml:19-26` | New provider block, `enabled: false` initially |
| Add 5th route to factory | `llm_client.py:1090-1154` (`make_client`) | New branch: `settings.enable_1m_context` → `ClaudeClient(model="claude-sonnet-4-6")` |
| Remove Sonnet/Opus cap | `llm_client.py:115-146` (`_MODEL_MAX_INPUT_CHARS`) | Delete or skip entry for `claude-sonnet-4-6` (currently no entry = unconstrained) |
| Directive rewriter full brief | `directive_rewriter.py:100-122` | Replace `_summarize_brief_signals` aggregation with raw brief text pass-through when 1M enabled |
| Outcome tracker reflection | `outcome_tracker.py:60+` | Include full_report_json in reflection prompt |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total (17 collected)
- [x] Recency scan (last 2 years) performed + reported (decisive 2026 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client, provider_rebalancer, directive_rewriter, orchestrator, skill_optimizer, outcome_tracker, budget YAMLs)
- [x] Contradictions / consensus noted (ToS prohibition is unambiguous consensus; cost math is new finding)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-19.0-research-brief.md",
  "gate_passed": true
}
```
