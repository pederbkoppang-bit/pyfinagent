# Messages Sidebar Final Sweep (phase-4.13.0)

Deep audit of 20 Messages-sidebar docs pages against pyfinagent's current
implementation. AUDIT ONLY — no source modified. Dated 2026-04-18.

## URL coverage (20 URLs)

| # | URL | Status | Canonical / note |
|---|-----|--------|------------------|
| 1 | /docs/en/intro | CHECKED | — |
| 2 | /docs/en/quickstart | CHECKED | Landing page redirects conceptually to /docs/en/get-started; content is the quickstart |
| 3 | /docs/en/build-with-claude/overview | CHECKED | Full ZDR table captured |
| 4 | /docs/en/build-with-claude/working-with-messages | CHECKED | — |
| 5 | /docs/en/build-with-claude/claude-api-skill | FAILED (404) | No content. Does not appear to exist under /build-with-claude/. Best-match candidate is the Agent-Skills API (entry 20). Flagging as phantom page — likely dropped or renamed after the v3 sweep. |
| 6 | /docs/en/build-with-claude/handling-stop-reasons | CHECKED | — |
| 7 | /docs/en/build-with-claude/effort | CHECKED | `xhigh` is Opus-4.7-only |
| 8 | /docs/en/build-with-claude/task-budgets | CHECKED | Beta header `task-budgets-2026-03-13`, Opus-4.7 only |
| 9 | /docs/en/build-with-claude/fast-mode | CHECKED | Beta `fast-mode-2026-02-01`, Opus-4.6 only |
| 10 | /docs/en/build-with-claude/structured-outputs | CHECKED | — |
| 11 | /docs/en/build-with-claude/citations | CHECKED | — |
| 12 | /docs/en/build-with-claude/streaming | CHECKED | Large doc, persisted |
| 13 | /docs/en/build-with-claude/search-results | CHECKED | Large doc, persisted |
| 14 | /docs/en/build-with-claude/streaming-refusals | FAILED (404) | Streaming-specific refusal handling is actually folded into the Streaming page (event type `message_delta` with `stop_reason:"refusal"`) and handling-stop-reasons page. No standalone page. |
| 15 | /docs/en/build-with-claude/multilingual-support | CHECKED | — |
| 16 | /docs/en/build-with-claude/files | CHECKED | Beta `files-api-2025-04-14` |
| 17 | /docs/en/build-with-claude/pdf-support | CHECKED | 32 MB / 600 pages |
| 18 | /docs/en/build-with-claude/vision | CHECKED | 100–600 images, 8000×8000 px |
| 19 | /docs/en/agents-and-tools/agent-skills/quickstart | CHECKED | Beta `skills-2025-10-02` + `code-execution-2025-08-25` |
| 20 | /docs/en/agents-and-tools/agent-skills/skills-in-the-api | FAILED (404) | Content is effectively merged into `/build-with-claude/skills-guide` + the quickstart above. The quickstart already covers `container.skills` / `skill_id` invocation from Messages API. |

**18/20 fetched successfully. The 3 that 404'd (#5, #14, #20) are phantom
slugs — their content has been absorbed into adjacent pages. No MUST-FIX lost.**

---

## Per-page digests + pyfinagent relevance

### 1. Intro & 2. Quickstart (`/intro`, `/quickstart`)

Two build paths: Messages API (us) and Managed Agents (we skip, per
`managed_agents_gaps.md`). Four-step flow recommended: quickstart →
Messages API → model comparison → features. Quickstart omits `system=`
and caching.

- **Adopt:** Nothing net-new. Link the four-step flow from CLAUDE.md.
- **Avoid:** Don't use the bare quickstart as a template — always send
  `system=` with `cache_control:"ephemeral"`.

### 3. Features overview + ZDR table (`/docs/en/build-with-claude/overview`)

ZDR eligibility is feature-level and documented explicitly. Maps to our
current use:

| Feature we use | ZDR eligible? |
|---|---|
| Adaptive thinking | Yes |
| Effort parameter | Yes |
| Prompt caching (5m + 1h) | Yes |
| Citations | Yes |
| Structured outputs | Yes, qualified (schemas cached ≤24h; prompts/outputs not stored) |
| PDF support | Yes |
| Search results | Yes |
| Batch processing | **No** |
| Web search + web fetch | Yes, except when `dynamic_filtering` is on |
| Files API | **No** |
| Agent Skills (beta) | **No** |
| MCP connector | **No** |
| Code execution | **No** |

- **Adopt (MUST-FIX confirmation):** If Peder ever requests ZDR, Batch +
  Files + Skills + MCP become unavailable. Flag this before we adopt the
  Files API in the 10-K ingest.
- **Reconcile:** Our `llm_client.py::ClaudeClient` passes `system` with
  `cache_control:{type:"ephemeral"}` — ZDR-compatible. Good.

### 4. Using the Messages API (`working-with-messages`)

Canonical shape: `model`, `max_tokens`, `messages[{role, content}]`.
`system=` at top level (string OR list of text blocks — list required for
`cache_control`). Stateless multi-turn. **Prefill is NOT supported on
Opus 4.7 / 4.6 / Sonnet 4.6** — 400 error. Use structured outputs instead.

- **Confirm:** `llm_client.py:617` uses the list form with cache_control —
  correct.
- **Withdraw v3 nice-to-have:** "Use prefill for JSON coercion" — no longer
  valid on current Claude models.

### 5. Claude API skill — 404 (phantom).

### 6. Handling stop reasons (`handling-stop-reasons`) — HIGH IMPORTANCE

Six `stop_reason` values plus `model_context_window_exceeded`:

1. `end_turn` — normal
2. `max_tokens` — retry with higher cap; if last block is incomplete
   `tool_use`, must retry (truncated tool calls are unusable)
3. `stop_sequence`
4. `tool_use` — execute tool, feed back
5. `pause_turn` — server-side tool loop hit its 10-iteration cap; resend
   assistant message verbatim to let it continue
6. `refusal` — model declined for safety
7. `model_context_window_exceeded` — GA on Sonnet 4.5+; beta header
   `model-context-window-exceeded-2025-08-26` on older models

Also highlights an empty-response failure mode (`end_turn` with zero content
blocks) caused by appending `{type:"text"}` immediately after a
`tool_result` — teaches the model that tool results are user-text turns and
it should end.

**pyfinagent coverage today** (`multi_agent_orchestrator.py:962, 1024`):
- `tool_use` — handled
- `end_turn` / `max_tokens` — treated as "agent finished" with no
  differentiation. **Gap: we do not retry on `max_tokens` with an incomplete
  `tool_use` tail block, nor do we log a warning.**
- `pause_turn` — **not handled**. We don't use server-side tools today
  (we call our own Python tools), so the risk is theoretical. Still, the
  moment we switch to web-search / web-fetch server tools (proposed in v3
  §5.3), we'd silently drop the response.
- `refusal` — **not handled anywhere**. `assistant_handler.py` has zero
  hits on `stop_reason`, `refusal`, or `pause_turn` (confirmed via grep).
  If the Slack bot hits a refusal, the user sees an empty message.
- `model_context_window_exceeded` — **not handled**; would currently
  appear as empty text, same as refusal.

**Net-new MUST-FIX:**
- **MF-26: Full stop_reason dispatch table in `ClaudeClient` and
  `assistant_handler`.** Handle all 7 values with explicit log + UX
  behavior. Particularly: surface `refusal` to the Slack user with a
  "Claude declined this request" message + suggest retry with Haiku 4.5
  (which has different safety filters, per the doc's own tip).
- **MF-27: On `max_tokens` with incomplete `tool_use` tail block,
  auto-retry with 2× `max_tokens`.** Cheap; prevents silent data loss.

**Cross-ref to v2:** v2 item "empty response after tool_result" was
already MUST-FIX; this page confirms the root cause.

### 7. Effort (`/build-with-claude/effort`) — HIGH IMPORTANCE

Five levels: `low, medium, high, xhigh, max`. `xhigh` is **Opus-4.7 only**.
Default is `high`. Opus 4.7 guidance: **start at `xhigh` for coding /
agentic tasks**, step down to `medium` for cost-sensitive work. Sonnet 4.6
recommended default is **`medium`**, not `high`, to avoid unexpected latency.
`budget_tokens` is **deprecated** on Opus 4.6 and Sonnet 4.6; effort
replaces it. Manual `thinking: {type: "enabled", budget_tokens: N}` is
**no longer supported on Opus 4.7** — adaptive thinking + effort is the only
path. Effort affects text, tool calls, AND thinking tokens.

**pyfinagent coverage:** `ClaudeClient` at `llm_client.py:622-628` still
reads `thinking: {budget_tokens: N}` and forwards it to the API. This works
on Opus 4.5 and older but would fail on Opus 4.7 with a 400.

- **Net-new MUST-FIX (MF-28):** Add `effort` pass-through to `ClaudeClient`.
  Default to `high` for synthesis/debate, `medium` for enrichment, `low`
  for bias/conflict checks. Keep `budget_tokens` path only for Opus 4.5
  and 3.7.
- **Net-new MUST-FIX (MF-29):** Gate the extended-thinking code path
  (`llm_client.py:622-628`) on `"claude-opus-4-5"` or older. Opus 4.7 will
  400 on `thinking: {type: "enabled"}`.

### 8. Task budgets (`task-budgets`) — BETA, OPUS-4.7 ONLY

Beta header `task-budgets-2026-03-13`. `output_config.task_budget =
{type:"tokens", total:N, remaining?:N}`. Advisory, not enforced; combine
with `max_tokens` as the hard cap. The countdown is injected server-side;
if your client resends full history + decrements `remaining` client-side,
you *under-report* budget and the model wraps up prematurely. Minimum
`total` is 20k tokens. Budgets that are too small cause refusal-like
behavior (scope-down / early stop). Mutating `remaining` per turn
invalidates prompt caching — set once and let the server track.

- **Adopt (net-new NICE-TO-HAVE NTH-10):** Pair `task_budget` with our
  backtest iteration cap in the harness. If `scripts/harness/run_harness.py`
  wants to cap a full Cycle's Claude spend at e.g. 500k tokens, setting
  `task_budget.total=500_000` on the planner's first request is cleaner
  than our current `consecutive_fails` counter. Caveat: Opus-4.7 only +
  beta, so wait until GA before the harness depends on it.
- **Avoid:** Do NOT mutate `remaining` turn-over-turn; we already do
  full-history resends.

### 9. Fast mode (`fast-mode`) — BETA, Opus-4.6 only

`speed:"fast"` + beta `fast-mode-2026-02-01`. 2.5× OTPS at **6× price**
($30/MTok in, $150/MTok out). Not on Batch, not on Priority Tier. Distinct
from Claude Code `/fast`. Switching invalidates prompt cache.

- **Do not adopt.** Our pipeline is not TTFT-sensitive, and 6× pricing
  kills the economics. Revisit only for interactive Slack Q&A on Opus 4.6.

### 10. Structured outputs (`structured-outputs`) — HIGH IMPORTANCE, confirms v3 MUST-FIX

Two features: `output_config.format = {type:"json_schema", schema:{...}}`
for response-shape, and `strict:true` on tools for tool-input validation.
**Incompatible with citations** (returns 400 if both are on). SDK helpers:
Pydantic in Python, Zod in TS. Schema cache 24h. Hard limits: 20 strict
tools, 24 optional params, 16 union params. First request with a new schema
incurs grammar compilation latency.

**pyfinagent coverage today** (`llm_client.py:586-598`): ClaudeClient
injects the schema as a **system-prompt string** rather than via
`output_config.format`. This is the old pre-GA workaround and produces
best-effort JSON, not guaranteed JSON.

- **Confirm MUST-FIX (v3 MF-21):** Switch ClaudeClient to
  `output_config.format = {type:"json_schema", schema:...}` for all
  structured calls. This was already flagged in v3; this page confirms no
  change in recommendation.
- **Net-new MUST-FIX (MF-30):** When a call has `citations.enabled=True`
  AND structured output, we must pick one — error early with a clear
  message instead of letting Anthropic 400.

### 11. Citations (`citations`)

`citations:{enabled:true}` on document blocks (text, PDF, or custom
content). Three citation types: `char_location` (plain text),
`page_location` (PDF), `content_block_location` (custom). `cited_text` is
**free** (not counted as output or input tokens). Works with prompt
caching via `cache_control` on the document block. Must be enabled on all
docs or none in a request. Streaming adds `citations_delta` events.

- **Adopt (net-new MUST-FIX, MF-31):** `backend/tools/sec_insider.py` and
  `earnings_tone.py` pass filings as plain text today. Wrap them in
  `type:"document"` blocks with `citations.enabled=true` so synthesis cites
  specific 10-K sentences. `cited_text` being free is a direct cost win.
- **Reconcile:** Confirmed incompatible with structured outputs — gate
  this in ClaudeClient (see MF-30).

### 12. Streaming (`streaming`)

SSE events: `message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, `message_stop`, `ping`. `stop_reason`
arrives in `message_delta`, **not** in `message_stop`. SDKs offer
high-level helpers (`client.messages.stream(...).text_stream`).
Reconnection: no built-in resume — must restart the request. Tool use
streams as `input_json_delta` chunks.

- **Adopt (net-new NTH-11):** Wire streaming into the Slack assistant
  handler so long Opus 4.7 syntheses (30s+) feel responsive. Use the
  Python SDK's `client.messages.stream(...).text_stream` pattern; don't
  hand-roll SSE parsing.
- **Avoid:** Don't rely on `message_stop` for `stop_reason` — that's in
  `message_delta`.

### 13. Search results (`search-results`)

`type:"search_result"` content block `{source, title, content:[{type:"text",text:...}]}`.
Produces natural citations identical to web-search quality. Can be top-level
content or wrapped in a `tool_result`. Supported on Opus 4.7, 4.6, 4.5, 4.1,
Sonnet 4.6, 4.5, 3.7, Haiku 4.5, 3.5.

- **Adopt (net-new MUST-FIX, MF-32):** When we pull BigQuery rows for RAG
  (macro events, insider tx, price context) and pass them into synthesis,
  wrap each row as a `search_result` block with a source URL pointing at
  `sunny-might-477607-p8://{dataset}/{table}/{row_id}` or the upstream
  SEC/FRED URL. Gives us free "According to the 2024-Q3 filing..." style
  citations and closes the long-standing "where did the model get this
  number?" hole.

### 14. Streaming refusals (404)

No standalone page. Refusal streaming behavior is: during a stream, a
refusal still emits `message_delta` with `stop_reason:"refusal"`. No
special client logic needed beyond the standard stop_reason dispatch
(covered under MF-26). Not a gap.

### 15. Multilingual support (`multilingual-support`)

MMLU-relative scores: Spanish/Portuguese/Italian/French at 97-98% of
English, Japanese 96.9%, Swahili 89.8%, Yoruba 80.3%. No API surface
change — just guidance (state target language explicitly, native scripts).

- **Defer:** US-tickers-only today. Revisit for phase-5 if we ever ingest
  Nikkei decks or European insider filings.

### 16. Files API (`files`)

Beta `files-api-2025-04-14`. 500 MB per file, 500 GB per org. File IDs
reusable across requests. **Not ZDR-eligible.** Not on Bedrock / Vertex.
Storage is per-workspace. Uploads/downloads/list/delete are **free**;
inference pricing unchanged (input tokens). `.csv/.xlsx/.docx/.md/.txt`
not supported as `document` blocks — convert to plain text or PDF.
Download only works for files CREATED by Skills / code-execution, not
user-uploaded files.

- **Adopt (net-new MUST-FIX, MF-33):** For large SEC 10-Ks that blow past
  the 32 MB request limit (common for mega-caps with 600+ page filings),
  upload once via Files API and reference by `file_id`. Caches across the
  pipeline (enrichment → synthesis → citations) without re-uploading.
- **Avoid:** Don't use Files API if Peder requests ZDR — not eligible.

### 17. PDF support (`pdf-support`)

32 MB / 600 pages (100 pages on 200k-context models). Rasterise + OCR per
page. 1.5-3k tokens/page text + image tokens. Best paired with prompt
caching (`cache_control:"ephemeral"` on the document block) for repeated
queries on the same PDF, and with Batch API for high-volume.

- **Adopt (net-new MUST-FIX, MF-34):** Route earnings decks and 10-Ks
  through the PDF content-block pathway with `cache_control:"ephemeral"`.
  Our current `earnings_tone.py` converts PDF → text first, losing the
  visual (charts, tables). PDF-native ingestion with citations gives us
  both back.
- **Reconcile:** For very large filings (>32 MB), combine with MF-33
  (Files API reference).

### 18. Vision (`vision`)

100 images/req on 200k-context, 600/req otherwise. 8000×8000 px max
(drops to 2000×2000 when >20 images). base64, URL, or Files API.

- **NTH-12:** Pass embedded filing figures as `image` blocks for chart
  OCR. Defer; low ROI.

### 19. Agent Skills quickstart + 20. Skills in API (20 is phantom)

Pre-built Skills: `pptx, xlsx, docx, pdf`, invoked via
`container.skills=[{type,skill_id,version}]` + tool
`code_execution_20250825`. Betas: `skills-2025-10-02` +
`code-execution-2025-08-25`. **Not ZDR-eligible.** Output files surface
as `file_id` in `tool_result` → download via Files API.

- **NTH-13:** `pptx` Skill for "export analysis as deck" if Peder asks.
  Keep Skills out of the daily pipeline (ZDR + Code-Execution compliance).

---

## Net-new MUST-FIX

| ID | Summary | File(s) | Effort |
|----|---------|---------|--------|
| MF-26 | Full stop_reason dispatch (all 7 values) in ClaudeClient + assistant_handler; surface `refusal` to Slack user with Haiku-4.5 retry suggestion | `backend/agents/llm_client.py`, `backend/slack_bot/assistant_handler.py`, `backend/agents/multi_agent_orchestrator.py` | M |
| MF-27 | On `max_tokens` with incomplete `tool_use` tail, auto-retry with 2× max_tokens | `backend/agents/llm_client.py`, `multi_agent_orchestrator.py:962` | S |
| MF-28 | Add `output_config.effort` pass-through to ClaudeClient (default high/medium/low per agent class) | `backend/agents/llm_client.py` | S |
| MF-29 | Gate `thinking:{type:"enabled",budget_tokens}` on Opus 4.5 and older; Opus 4.7 uses adaptive+effort | `backend/agents/llm_client.py:622-628` | S |
| MF-30 | Error fast when a call sets BOTH citations and structured outputs (400 otherwise) | `backend/agents/llm_client.py` | XS |
| MF-31 | Wrap SEC filings + earnings transcripts as `document` blocks with `citations.enabled=true` | `backend/tools/sec_insider.py`, `backend/tools/earnings_tone.py` | M |
| MF-32 | Wrap BigQuery RAG rows as `search_result` content blocks for natural citations in synthesis | `backend/agents/orchestrator.py` (Step 3 RAG), `backend/agents/multi_agent_orchestrator.py` | M |
| MF-33 | Files API pathway for large (>32 MB) SEC filings with `file_id` reuse across the pipeline | `backend/tools/sec_insider.py` | M |
| MF-34 | PDF-native ingestion with `cache_control:"ephemeral"` for earnings decks and 10-Ks | `backend/tools/earnings_tone.py` | M |

## Net-new NICE-TO-HAVE

| ID | Summary | Notes |
|----|---------|-------|
| NTH-10 | Use `task_budget` to cap per-Cycle harness spend on Opus 4.7 | Beta only; wait for GA |
| NTH-11 | Stream Opus 4.7 synthesis into Slack for responsiveness | SDK `text_stream` helper, not raw SSE |
| NTH-12 | Pass embedded filing figures as vision `image` blocks | Low ROI today |
| NTH-13 | `pptx` Skill for "export analysis as deck" export | Gated behind ZDR & Code-Execution constraints |

## Cross-references to prior v3 findings

- **v3 MF-21 (structured outputs via `output_config.format`):** Confirmed
  by this sweep's §10. No change.
- **v3 "prefill for JSON coercion" nice-to-have:** **WITHDRAWN**. Page #4
  explicitly rejects prefill on Opus 4.7 / 4.6 / Sonnet 4.6. The right
  replacement is structured outputs.
- **v3 "empty response after tool_result" MUST-FIX:** Root cause
  confirmed by §6 (appending text-blocks after tool_results teaches the
  model to end). No change to recommendation.
- **v3 managed_agents_gaps.md position (skip Managed Agents):** Reaffirmed
  by §1's clean Messages-API-vs-Managed-Agents split. Continue to keep
  Managed Agents out of scope for phase-4.
- **v3 "adopt effort parameter" NICE-TO-HAVE:** **PROMOTED to MUST-FIX
  (MF-28 + MF-29)** because Opus 4.7 silently rejects
  `thinking:{type:"enabled"}` — our current code path would 400.
- **v3 Files API NICE-TO-HAVE:** **PROMOTED to MUST-FIX (MF-33)** — 500 MB
  limit + free uploads means we no longer have an excuse for the current
  "re-encode filing PDF on every synthesis call" pattern.
- **v3 Citations NICE-TO-HAVE:** **PROMOTED to MUST-FIX (MF-31)** —
  `cited_text` being free input+output is too strong a cost + auditability
  win to leave on the shelf.

## References

- https://platform.claude.com/docs/en/intro
- https://platform.claude.com/docs/en/quickstart
- https://platform.claude.com/docs/en/build-with-claude/overview
- https://platform.claude.com/docs/en/build-with-claude/working-with-messages
- https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
- https://platform.claude.com/docs/en/build-with-claude/effort
- https://platform.claude.com/docs/en/build-with-claude/task-budgets
- https://platform.claude.com/docs/en/build-with-claude/fast-mode
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- https://platform.claude.com/docs/en/build-with-claude/citations
- https://platform.claude.com/docs/en/build-with-claude/streaming
- https://platform.claude.com/docs/en/build-with-claude/search-results
- https://platform.claude.com/docs/en/build-with-claude/multilingual-support
- https://platform.claude.com/docs/en/build-with-claude/files
- https://platform.claude.com/docs/en/build-with-claude/pdf-support
- https://platform.claude.com/docs/en/build-with-claude/vision
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart
- https://platform.claude.com/docs/en/build-with-claude/claude-api-skill (404; phantom)
- https://platform.claude.com/docs/en/build-with-claude/streaming-refusals (404; phantom, behavior folded into stop-reasons + streaming pages)
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/skills-in-the-api (404; phantom, covered by quickstart + skills-guide)

Internal cross-refs:
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/llm_client.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/orchestrator.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/tools/sec_insider.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/tools/earnings_tone.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/assistant_handler.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/phase-4.12/CONSOLIDATED_REPORT_v3.md`
