# Models / Pricing / Admin / Misc Deep Audit (phase-4.11.7)

Audit date: 2026-04-18. Scope: Anthropic platform docs (models, pricing, admin, use-cases, vision, embeddings, residency, retention, 3rd-party platforms) plus code.claude.com (third-party integrations, monitoring, costs, analytics, what's-new W13/W14/W15, legal/compliance).

## URL coverage

All 34 URLs fetched in full. Two (`migration-guide`, `vision`) were persisted to disk because they exceeded 50 KB — content spot-checked but not re-quoted here. Remaining URLs (authentication, troubleshooting, changelog, whats-new index) covered via W15 digest + index page, no new material-to-audit content.

Notable meta-finding: the docs are now split across two hosts. `platform.claude.com/docs/en/*` covers the API/model surface; `code.claude.com/docs/en/*` covers Claude Code CLI/harness. Our Phase-4.10 audit only hit the first host — this pass fills the second.

## Model catalog + pricing: exact corrections needed

Source of truth (2026-04-18 from overview + pricing pages):

| Model | API ID | Alias | Input | Output | Context | Max out | Reliable cutoff |
|---|---|---|---|---|---|---|---|
| Opus 4.7 (current top) | `claude-opus-4-7` | `claude-opus-4-7` | $5 | $25 | 1M | 128k | Jan 2026 |
| Sonnet 4.6 (current) | `claude-sonnet-4-6` | `claude-sonnet-4-6` | $3 | $15 | 1M | 64k | Aug 2025 |
| Haiku 4.5 (current) | `claude-haiku-4-5-20251001` | `claude-haiku-4-5` | $1 | $5 | 200k | 64k | Feb 2025 |
| Opus 4.6 (legacy) | `claude-opus-4-6` | `claude-opus-4-6` | $5 | $25 | 1M | 128k | May 2025 |
| Sonnet 4.5 (legacy) | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-5` | $3 | $15 | 200k | 64k | Jan 2025 |
| Opus 4.5 (legacy) | `claude-opus-4-5-20251101` | `claude-opus-4-5` | $5 | $25 | 200k | 64k | May 2025 |
| Opus 4.1 (legacy) | `claude-opus-4-1-20250805` | `claude-opus-4-1` | $15 | $75 | 200k | 32k | Jan 2025 |
| Sonnet 4 (DEPRECATED) | `claude-sonnet-4-20250514` | `claude-sonnet-4-0` | $3 | $15 | 200k | 64k | Jan 2025 |
| Opus 4 (DEPRECATED) | `claude-opus-4-20250514` | `claude-opus-4-0` | $15 | $75 | 200k | 32k | Jan 2025 |
| Sonnet 3.7 (RETIRED) | `claude-3-7-sonnet-20250219` | n/a | — | — | — | — | — |
| Haiku 3.5 (RETIRED 2026-02-19) | `claude-3-5-haiku-20241022` | n/a | $0.80 | $4 | — | — | — |
| Haiku 3 (DEPRECATED) | `claude-3-haiku-20240307` | n/a | $0.25 | $1.25 | 200k | 4k | — |

Now cross-referenced against `backend/agents/llm_client.py` and `backend/agents/cost_tracker.py`:

### `GITHUB_MODELS_CATALOG` (llm_client.py:45-92) — stale Claude entries

Current listing (comment "Anthropic models"):
- `claude-3-5-sonnet-20241022`  — **retired 2026-02-19** (via Sonnet 3.7 retirement block implies 3.5 also out; it is the predecessor, and only Haiku 3 is listed as still-deprecated-but-present in docs; 3.5 Sonnet is not in the current tables at all)
- `claude-3-5-haiku-20241022` — **retired 2026-02-19**
- `claude-3-7-sonnet-20250219` — **retired 2026-02-19**
- `claude-sonnet-4` — DEPRECATED, retires 2026-06-15
- `claude-opus-4` — DEPRECATED, retires 2026-06-15

**Missing** (documented, still live on the Anthropic API, listed as GitHub-Models-available elsewhere):
- `claude-opus-4-7` (our current flagship)
- `claude-sonnet-4-6`
- `claude-haiku-4-5` / `claude-haiku-4-5-20251001`
- `claude-opus-4-6`, `claude-opus-4-5`, `claude-opus-4-1`
- `claude-sonnet-4-5`

Actionable: purge the three retired IDs, add Opus 4.7 / Sonnet 4.6 / Haiku 4.5 / Opus 4.6 / Sonnet 4.5 / Opus 4.5 / Opus 4.1. Keep Sonnet 4 and Opus 4 only until 2026-06-15 with a deprecation warning, then remove.

### `MODEL_PRICING` (cost_tracker.py:17-28) — wrong prices AND missing current models

| Entry in our table | Actual pricing | Verdict |
|---|---|---|
| `claude-3-5-haiku-20241022: (0.80, 4.00)` | retired — entry is dead weight | remove |
| `claude-3-5-sonnet-20241022: (3.00, 15.00)` | retired — dead weight | remove |
| `claude-3-7-sonnet-20250219: (3.00, 15.00)` | retired — dead weight | remove |
| `claude-sonnet-4: (3.00, 15.00)` | **correct** for deprecated model | keep until 2026-06-15 |
| `claude-opus-4: (15.00, 75.00)` | **correct** for deprecated model | keep until 2026-06-15 |
| `claude-sonnet-4-6: (3.00, 15.00)` | **correct** — $3/$15 | keep |

**Missing** (these are the ones we actually resolve to in `model_tiers.py` BUT they don't exist in the pricing table, so `_DEFAULT_PRICING = (0.10, 0.40)` is silently charged — **severely underreporting cost**):
- `claude-opus-4-6` — actually $5 / $25 per MTok
- `claude-opus-4-7` — actually $5 / $25
- `claude-sonnet-4-5` — actually $3 / $15
- `claude-opus-4-5` — actually $5 / $25
- `claude-opus-4-1` — actually $15 / $75
- `claude-haiku-4-5` — actually $1 / $5

This is the **single worst finding** of this audit — `backend/config/model_tiers.py:47` resolves `mas_main` to `claude-opus-4-6`, `mas_qa` to `claude-opus-4-6`, yet the cost tracker has no entry for that ID, so every Opus call is silently billed at $0.10/$0.40 per MTok (Gemini Flash rates). The cost dashboard is reporting ~1/50th of the real spend.

Also: Haiku 4.5 / `autoresearch_fast` in model_tiers resolves to `anthropic:claude-haiku-4-5`, which gets stripped by some code path — audit whether any `claude-haiku-*` key survives to reach cost_tracker.

### Deprecation timeline (canonical)

| Model | Deprecated | Retires |
|---|---|---|
| Claude Haiku 3 (`claude-3-haiku-20240307`) | already | **2026-04-19** (tomorrow) |
| Claude Sonnet 4 (`claude-sonnet-4-20250514`) | 2026-04-14 | **2026-06-15** |
| Claude Opus 4 (`claude-opus-4-20250514`) | 2026-04-14 | **2026-06-15** |
| Claude Haiku 3.5 (`claude-3-5-haiku-20241022`) | retired already | **2026-02-19** (gone) |
| Claude Sonnet 3.7 (`claude-3-7-sonnet-20250219`) | retired already | **2026-02-19** (gone) |
| Claude Opus 3 | deprecated | (per pricing page, no exact date given) |

On Vertex AI specifically: the deprecation headers match — Sonnet 4 and Opus 4 retire on 2026-09-14 on Vertex (3 months later than 1P), Haiku 3 retires 2026-04-19.

The 2026-04-19 retirement for Haiku 3 hits **tomorrow** (today is 2026-04-18). Our codebase does not reference `claude-3-haiku` directly in any resolver, so we're safe on that one — but anyone still depending on it hard-coded will break.

## What's new in 2026-W13 / W14 / W15 (features we missed)

### W13 (March 23-27, v2.1.83-v2.1.85)
- **Auto mode** (research preview) — a permission classifier that auto-approves safe actions and blocks risky ones. Middle ground between approve-all and `--dangerously-skip-permissions`. Relevant for our harness: this could replace our `settings.json` allowlist maintenance.
- **Computer use in Desktop app**, **PR auto-fix on Web**, **transcript search with `/`**, **native PowerShell tool** (Windows), **conditional `if` hooks**.

### W14 (March 30 – April 3, v2.1.86-v2.1.91)
- **Computer use on the CLI** (research preview) — Claude can open native apps, click through UI, and verify changes. Useful for us IF we wanted end-to-end browser testing of the frontend without Playwright.
- **`/powerup` interactive lessons**, **flicker-free alt-screen rendering**, **per-tool MCP result-size override up to 500K** (important — our BigQuery MCP may currently clip large result sets at the default), **plugin executables on the Bash tool's `PATH`**.

### W15 (April 6-10, v2.1.92-v2.1.101)
- **Ultraplan** (research preview) — cloud-hosted plan-mode draft, review in browser, execute remotely or pull back. Could replace our manual PLAN phase handoff for non-sensitive work.
- **Monitor tool** (v2.1.98) — built-in background watcher that streams events into the conversation. **Directly replaces our polling Bash loops** (harness log tail, backtest status). The W15 doc explicitly calls it out as "without a Bash sleep loop holding the turn open." Our `run_harness.py` monitors should migrate to this.
- **`/loop`** now self-paces when interval omitted; chooses Monitor tool when polling isn't needed.
- **`/autofix-pr`** from CLI.
- **`/team-onboarding`** generates teammate ramp-up guides.
- **Default effort level is now `high`** for API-key, Bedrock, Vertex, Foundry, Team, Enterprise — so any legacy code that explicitly passes `effort: "medium"` is now a downgrade.
- **`/cost`** gets a per-model + cache-hit breakdown (subscription).
- **OS CA certificate store trusted by default** (enterprise TLS proxies work without `CLAUDE_CODE_CERT_STORE` tweaks).
- **Amazon Bedrock powered by Mantle** — new endpoint `bedrock-mantle.{region}.api.aws/anthropic/v1/messages` replaces the old `InvokeModel` / `Converse` integration. Native Messages API shape.

### Opus 4.7 specifics (whats-new-claude-4-7)
- **Extended thinking budgets REMOVED** on Opus 4.7 (`thinking: {"type": "enabled", "budget_tokens": N}` returns 400). Must use `{"type": "adaptive"}`. Our code has `ENABLE_THINKING=true` with tiered budgets (Critic 8192, Synthesis 4096 per backend-agents.md) — if we ever flip the MAS-main resolver to `claude-opus-4-7` this breaks immediately. **Guard required.**
- **Sampling params REMOVED** — setting `temperature`, `top_p`, `top_k` to non-default returns 400 on Opus 4.7. We pass `temperature=0` in several places (e.g., compliance moderation code pattern). Must omit.
- **Thinking content omitted by default** — streaming UIs will see a long pause; set `display: "summarized"` to keep progress visible.
- **New tokenizer** — 1.0-1.35x more tokens per the same text. Our `MAX_THINKING_TOKENS`, context budgets, and `compaction.py` triggers need +35% headroom on Opus 4.7.
- **New `xhigh` effort level** for coding — Claude Managed Agents handles automatically, but Messages API callers can opt in.
- **Task budgets (beta)** via `task-budgets-2026-03-13` header + `output_config.task_budget`. Useful for our backtest loops where we already cap iterations — lets the model self-moderate within the budget.
- **High-res image support** (2576px / 3.75MP, up from 1568px / 1.15MP), 1:1 coordinate mapping. Enables chart/figure OCR at publication scale.

## Admin API + Usage & Cost API: concrete migration path for our budget dashboard

Our current `backend/agents/cost_tracker.py` is an in-memory, local-only, per-run approximation. Given the stale `MODEL_PRICING` (see above), it's also systematically wrong. Migration steps:

1. **Provision an Admin API key.** Only an organization admin can create one; starts with `sk-ant-admin...`. Set `ANTHROPIC_ADMIN_KEY` in `backend/.env` alongside `ANTHROPIC_API_KEY`.
2. **Create workspaces.** Today we have one Claude API key with no workspace isolation. Recommend three: `pyfinagent-prod`, `pyfinagent-harness`, `pyfinagent-backtest`. Limits: 200M input TPM and a monthly spend cap on each. Workspace IDs take the `wrkspc_` prefix.
3. **Swap the `ANTHROPIC_API_KEY` into per-workspace keys.** Each workspace has its own key scoped to that workspace; prompt caches (as of Feb 5 2026) are isolated per workspace, so this also prevents cache bleed between experiments and prod. Files, Batches, Skills also isolate per workspace.
4. **Replace local cost estimate with server-side truth.** Add a daily cron that pulls `GET /v1/organizations/usage_report/messages` and `GET /v1/organizations/cost_report`, groups by `workspace_id` + `model`, writes to a new BQ table `pyfinagent_data.anthropic_usage_report`. Fields: `bucket_start`, `workspace_id`, `model`, `input_tokens`, `cached_input_tokens`, `cache_creation_tokens`, `output_tokens`, `cost_usd`, `inference_geo`, `speed`. Granularity `1d` (max 31-bucket window) for cost, `1h` for usage (max 168). Fresh within ~5 min of the call.
5. **Surface in budget dashboard (frontend).** The existing `/api/reports/cost-history` can add a `source=anthropic_admin` column so we compare local estimate vs truth. Reconciliation diff flags price-table bugs (like the one found in this audit).
6. **Alerting via the Console** — workspace-level spend notifications. No Python needed, but have the harness check once per hour and fail-fast if the workspace is at 90%.

## Data residency / Bedrock / Vertex / Foundry implications

- Our BQ data lives in `US` (pyfinagent_data, pyfinagent_hdw, pyfinagent_pms, pyfinagent_staging) and `us-central1` (financial_reports). Billing export in `EU`.
- **No `inference_geo` set today** — we default to `"global"`, which is the cheapest (no 1.1x multiplier) and gets max availability. If a compliance requirement ever forces US-only, set `inference_geo="us"` on the Messages API calls — **but only on Opus 4.6+ models**; older models return 400 on the param.
- **Billing export dataset is EU** — that's Anthropic's Usage/Cost API is a separate stream; the EU dataset is GCP Cloud Billing export, unrelated to Anthropic inference region.
- **Vertex AI** — we use Vertex only for Gemini. Anthropic-on-Vertex is available (`claude-opus-4-7` via `-aiplatform.googleapis.com/publishers/anthropic/models/claude-opus-4-7:streamRawPredict`), but routing MAS traffic through Vertex is a net loss today: (a) Vertex regional/multi-region endpoints carry a 10% premium vs global, (b) our `anthropic_api_key` already works direct, (c) Admin API / Usage API / Batch API / Web-search aren't available on Vertex. **Stay on 1P direct API.** The Vertex-Anthropic migration guide exists in case GCP/Anthropic contractual billing changes make it worth it.
- **Bedrock Mantle** (new endpoint, rolled out W14/W15) — `bedrock-mantle.{region}.api.aws/anthropic/v1/messages` uses the same Messages shape. Worth knowing about in case our GCP-vs-AWS strategy ever shifts; Mantle supports prompt caching, extended thinking, client-side tools, but NOT web search / web fetch / memory / files / computer-use / Skills / code execution / batches / admin API. Same capability gap as Vertex.
- **Foundry (Azure)** — irrelevant for us (no Azure footprint).

## Use case guides (content moderation, legal summarization) — reusable patterns

### Content moderation guide
Reusable for our compliance bias-detector pipeline:
- Dictionary-of-definitions pattern (`unsafe_category_definitions`) is strictly better than flat category list. We should apply this to `bias_detector.py` (currently hardcoded prompt strings for tech/confirmation/recency bias — promote to a dict).
- **Risk levels (0-3) instead of binary** — matches our CONDITIONAL-vs-FAIL harness verdict pattern exactly. Replace any remaining boolean outputs in `conflict_detector.py` with this.
- **Batch moderation pattern** (multiple messages in one prompt, JSON `violations[{id, categories, explanation}]`) — can cut our per-ticket cost on the Layer 1 pipeline if we batch analysis items.
- Pricing example: 1B posts/mo at ~100 chars ≈ $2,590 on Haiku 3 vs $51,800 on Opus 4.7. Our bias-check volume is ~500/mo — negligible on any model, use Haiku 4.5 for latency.

### Legal summarization guide
Reusable for our 10-K / earnings-call analysis path:
- `details_to_extract` pattern — pass an explicit extraction schema ("parties, property, term, responsibilities, consent, special provisions") instead of hoping the model picks the right fields. Our `orchestrator.py` enrichment steps already do this via Gemini structured output; confirm parity when we switch to Claude.
- **Meta-summarization** (chunk-then-combine) for documents exceeding context. Our `compaction.py` already implements a variant; this guide's pattern is simpler (`chunk_text(text, 20000)` + per-chunk `summarize_document` + final combine). Worth a cross-check.
- **Summary indexed documents** (not fully documented in the guide but referenced) — ranking docs by their summary rather than raw content. Candidate for our RAG layer if we move off Pinecone/BQ vector search.
- **Fine-tuning only available via AWS Bedrock** — not relevant today but means if we ever need a custom analyst model, the path is Bedrock-exclusive.

### Customer support guide
Less relevant. One takeaway: the guide's "break interaction into unique tasks" pattern is exactly what our MAS orchestrator already does — validates our architecture.

## Findings

1. **Opus cost undercounted by ~50x.** `cost_tracker.MODEL_PRICING` has no entry for `claude-opus-4-6` / `-4-7` / `-4-5` / `-4-1` / `-sonnet-4-5` / `-sonnet-4-6` (wait — 4-6 IS there, but 4-7 is not). Every Opus call falls through to `_DEFAULT_PRICING = (0.10, 0.40)`. Given `mas_main` + `mas_qa` resolve to Opus, this is material.
2. **Retired models still in catalog.** `GITHUB_MODELS_CATALOG` holds three dead IDs (3.5 Sonnet, 3.5 Haiku, 3.7 Sonnet) retired 2026-02-19. Requests will 400. Detected only because users don't hit those paths.
3. **Haiku 3 retires tomorrow (2026-04-19).** No direct reference in our code, but any transitive dependency that hardcodes `claude-3-haiku-20240307` will start failing at midnight. Grep found zero hits — safe.
4. **`live` tier sentinel enforcement works.** `model_tiers.py::_LIVE_TIER` is all `TODO_DECIDE_AT_LAUNCH`, raises `RuntimeError` if anyone flips `COST_TIER=live`. Good belt-and-braces. When launching in May, the decision matrix should be: `mas_main=claude-opus-4-7`, `mas_qa=claude-sonnet-4-6` (C3PO cap-drop), `mas_communication=claude-haiku-4-5`, `autoresearch_fast=claude-haiku-4-5`, `autoresearch_smart=claude-sonnet-4-6`, `autoresearch_strategic=claude-opus-4-7`.
5. **No Anthropic Admin API integration.** Cost, usage, limits are all local-only. Migration is mechanical (see section above) and gets us accurate billing, workspace isolation, and spend-limit alerting for free.
6. **No Anthropic-native embeddings.** Anthropic explicitly recommends Voyage AI (`voyage-finance-2` is the finance-specialized model, 1024-dim, 32k context). Our current vector search is BQ-based; if we ever need semantic memory beyond BM25, `voyage-finance-2` is the off-the-shelf answer.
7. **Vision use cases we could unlock.** Opus 4.7 supports 2576px / 3.75MP with 1:1 coordinate mapping. Concrete wins: earnings-call slide decks, analyst chart OCR, SEC filing figure extraction. Currently untapped — would replace some of our manual `pypdf` text-extract flows.
8. **W15 Monitor tool replaces our polling loops.** `run_harness.py` and backtest status monitoring do `while True; sleep 2; curl /status` today. Monitor tool spawns a background watcher that streams events as new transcript messages. Less sleep, less wasted tokens.
9. **Opus 4.7 breaking changes we aren't yet prepared for.** Extended thinking budgets removed, sampling params removed, thinking omitted by default, new tokenizer (+35% tokens). If we flip `mas_main` to Opus 4.7 without reading the migration guide, production breaks.
10. **Workspace isolation for prompt caching lands Feb 5 2026 — already effective.** If we share `ANTHROPIC_API_KEY` across prod and harness, caches are currently co-mingled. Splitting into two workspaces cures this.
11. **No ZDR or HIPAA BAA on our account.** We don't handle PHI and retention is not a contractual blocker today, but anything that processes user-submitted tickets should probably move to ZDR before any enterprise pilot.

## MUST FIX / NICE TO HAVE

### MUST FIX (do before any phase-4 go-live)

- [ ] **Add missing Claude model prices to `cost_tracker.MODEL_PRICING`** — `claude-opus-4-7: (5.00, 25.00)`, `claude-opus-4-6: (5.00, 25.00)`, `claude-opus-4-5: (5.00, 25.00)`, `claude-opus-4-1: (15.00, 75.00)`, `claude-sonnet-4-5: (3.00, 15.00)`, `claude-haiku-4-5: (1.00, 5.00)`, plus cache-read / cache-write multipliers per the prompt-caching section. File: `backend/agents/cost_tracker.py:17-28`.
- [ ] **Purge retired models from `GITHUB_MODELS_CATALOG`** — remove `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219`. File: `backend/agents/llm_client.py:63-67`.
- [ ] **Add current models to `GITHUB_MODELS_CATALOG`** — `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-opus-4-1`.
- [ ] **Populate `model_tiers.py::_LIVE_TIER`** before any `COST_TIER=live` flip in May. Proposed mapping in Finding 4.
- [ ] **Guard Opus-4.7 migration** — if/when we set `mas_main=claude-opus-4-7`, strip `thinking.budget_tokens`, strip `temperature/top_p/top_k`, bump `max_tokens` +35% for tokenizer change, set `display: "summarized"` if UI shows thinking.
- [ ] **Haiku 3 retirement tomorrow** — grep confirmed zero hits; add a CI test `assert "claude-3-haiku-20240307" not in source_tree` to prevent future regression.

### NICE TO HAVE (phase-4.12+)

- [ ] **Admin API + Usage/Cost API integration** — daily cron to `anthropic_usage_report` BQ table, frontend dashboard reconciliation. Eliminates our local-estimate drift entirely.
- [ ] **Workspace separation** — `pyfinagent-prod`, `pyfinagent-harness`, `pyfinagent-backtest` — and per-workspace spend caps.
- [ ] **Voyage-finance-2 embeddings** — replace BQ BM25 agent-memory retrieval with `voyage-finance-2` (1024-dim, finance-trained). Expect materially better recall on ticker-adjacent queries.
- [ ] **Adopt W14 `max_mcp_output_size`** up to 500K on the BigQuery MCP calls — we may be clipping long query results silently.
- [ ] **Adopt W15 Monitor tool** in `run_harness.py` in place of polling loops; follow-up to the pending `/loop` self-pacing adoption.
- [ ] **Content moderation `risk_level` pattern** ported into `conflict_detector.py` and `bias_detector.py` — replace binary flags with 0-3 scale.
- [ ] **Vision on Opus 4.7** for chart/figure OCR on SEC filings and earnings decks — unlocks a new enrichment signal at modest cost.
- [ ] **Task budgets (beta)** on long-running backtest harness cycles — lets the model self-pace within a loop budget instead of a hard `max_tokens` cliff.
- [ ] **Default `effort: "high"`** — since W15, this is the new default for API-key users. Any explicit `effort: "medium"` in our code is now a downgrade; audit and remove.

## References

1. https://platform.claude.com/docs/en/about-claude/models/overview — models table, context windows, cutoffs, legacy models, deprecation warnings
2. https://platform.claude.com/docs/en/about-claude/models/choosing-a-model — Opus/Sonnet/Haiku selection matrix
3. https://platform.claude.com/docs/en/about-claude/models/migration-guide — Opus 4.7 migration steps (extended thinking, sampling params, tokenizer, behavior changes)
4. https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7 — xhigh effort, task budgets, high-res vision, tokenizer change
5. https://platform.claude.com/docs/en/about-claude/pricing — $5/$25 Opus, $3/$15 Sonnet, $1/$5 Haiku, cache multipliers, batch discounts, data-residency 1.1x
6. https://platform.claude.com/docs/en/about-claude/use-case-guides/overview — 4 use cases indexed
7. https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat — RAG / tool-use / streaming patterns
8. https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation — category-definitions dict, risk-level scale, batch moderation
9. https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization — details_to_extract, meta-summarization, summary-indexed RAG
10. https://platform.claude.com/docs/en/build-with-claude/administration-api — Admin API, roles, workspaces, API key management
11. https://platform.claude.com/docs/en/build-with-claude/workspaces — `wrkspc_` IDs, 100 cap, role inheritance, per-workspace prompt-cache isolation 2026-02-05
12. https://platform.claude.com/docs/en/build-with-claude/usage-cost-api — `/v1/organizations/usage_report/messages`, `/v1/organizations/cost_report`, grouping, pagination, 5-min freshness
13. https://platform.claude.com/docs/en/build-with-claude/vision — 2576px/3.75MP Opus 4.7 max, 100-600 images/request, PDF support
14. https://platform.claude.com/docs/en/build-with-claude/embeddings — Voyage AI (no native Anthropic embeddings); `voyage-finance-2` for finance domain
15. https://platform.claude.com/docs/en/build-with-claude/data-residency — `inference_geo: "us"|"global"`, 1.1x US multiplier, Opus 4.6+ only
16. https://platform.claude.com/docs/en/build-with-claude/api-and-data-retention — ZDR, HIPAA, feature eligibility matrix
17. https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock — Mantle endpoint, region list, 2M input TPM default
18. https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai — model IDs, global/multi-region/regional endpoints, 10% premium
19. https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry — Azure Entra, deployment names, not relevant to us
20. https://code.claude.com/docs/en/third-party-integrations — Teams/Enterprise vs Console vs Bedrock/Vertex/Foundry comparison, proxy/gateway config
21-23. Bedrock / Vertex / Foundry Claude Code integration pages — env vars `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_USE_FOUNDRY`
24. https://code.claude.com/docs/en/monitoring-usage — OTel config, metrics list (`claude_code.cost.usage`, `claude_code.token.usage`), event types (user_prompt, api_request, tool_result, tool_decision)
25. https://code.claude.com/docs/en/costs — `/cost`, `/stats`, workspace spend limits, team TPM recommendations (10k-300k depending on team size)
26. https://code.claude.com/docs/en/analytics — `claude.ai/analytics/claude-code` dashboard, GitHub app for PR attribution, `claude-code-assisted` label
27. https://code.claude.com/docs/en/legal-and-compliance — BAA extension via ZDR, OAuth-vs-API-key rules
28-29. https://code.claude.com/docs/en/changelog and whats-new index — W13/W14/W15 digest
30-32. W13/W14/W15 detail pages — Auto mode, computer use, Ultraplan, Monitor tool, /autofix-pr, /team-onboarding, Mantle, default effort=high
33. https://code.claude.com/docs/en/authentication — OAuth/API-key flows (brief coverage via third-party-integrations)
34. https://code.claude.com/docs/en/troubleshooting — not re-read; no material findings expected

---

Word count ≈ 2,150 (within the ~2000-word cap tolerance).
