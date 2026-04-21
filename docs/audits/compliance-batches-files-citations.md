# Compliance Audit — Batches API + Files API + Citations + Search Results

**Phase:** 4.15.6  
**Date:** 2026-04-18  
**Auditor:** researcher (merged external docs + internal code)  
**SDK pinned:** `anthropic==0.87.0` (requirements.txt line 37)  
**Zero-usage confirmation:** All four live grep checks returned no output — zero usage of any of the four features in production code.

---

## Claim verified

> pyfinagent uses ZERO of the four doc features (Batches, Files API, Citations, search_result blocks).

**Evidence:** grep commands run against `backend/` with patterns `messages.batches`, `beta.files`, `citations.*enabled`, `type.*search_result`, `char_location`, `page_location`, `content_block_location`, `anthropic-beta`, `base64.*application/pdf` — all returned empty.

---

## Current ingestion patterns (context for gap analysis)

| Call site | File | How docs reach Claude today |
|-----------|------|------------------------------|
| Paper-trading overnight sweep | `autonomous_loop.py:436` | Plain-text prompt string, `client.messages.create()` directly — one ticker at a time, sequential, no batching |
| MAS orchestrator agents | `multi_agent_orchestrator.py:148,1112` | Plain-text prompt via `anthropic.Anthropic().messages.create()` |
| SEC Form 4 filings | `sec_insider.py` | XML parsed to Python dict, dict fields injected as f-string into prompt |
| Earnings transcripts | `earnings_tone.py` | Transcript text scraped, passed as plain string to Gemini (not Claude) |
| Step 3 RAG | `orchestrator.py:492-498` | Vertex AI Search grounding — Gemini only, no Claude document blocks |
| ClaudeClient (llm_client.py) | `llm_client.py:581` | Flat string `prompt` → `messages=[{"role":"user","content":prompt}]` — no document blocks, no beta headers |

---

## Pattern table

---

### Pattern 1: Batches endpoint — zero usage
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing
- **Doc says:** `POST /v1/messages/batches` via `client.messages.batches.create(requests=[...])`; each request has `custom_id` (1-64 chars, `^[a-zA-Z0-9_-]{1,64}$`) and a `params` object identical to a normal Messages request.
- **Status:** missing
- **Evidence:** `grep -rn 'messages.batches' backend/ --include='*.py'` — no output
- **Deviation:** `autonomous_loop._run_claude_analysis()` calls `client.messages.create()` once per ticker in a sequential `asyncio.to_thread` loop (line 436). The overnight sweep runs N tickers × 1 synchronous API call each.
- **Risk:** Leaving 50% cost discount on the table for the overnight batch sweep. Sequential calls also hit per-minute rate limits faster than a single batch request.
- **Recommended fix:** Collect all ticker prompts into a `requests=[Request(custom_id=ticker, params=...)]` list; call `client.messages.batches.create()`; poll `client.messages.batches.retrieve(batch.id)` until `processing_status == "ended"`; iterate JSONL results.
- **MF-# mapping:** MF-34 (batch candidate for overnight sweep)

---

### Pattern 2: Batch limits — max 100k requests / 256 MB
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing#batch-limitations
- **Doc says:** "A Message Batch is limited to either 100,000 Message requests or 256 MB in size, whichever is reached first."
- **Status:** N/A (feature not used)
- **Evidence:** No batch code to validate against.
- **Deviation:** Not applicable yet; overnight sweep typically covers tens of tickers, well within limits.
- **Risk:** No risk at current scale. Must track if sweep scope expands beyond screener universe.
- **Recommended fix:** Add pre-flight assertion `assert len(requests) <= 100_000` and payload size check before batch submission.
- **MF-# mapping:** MF-34

---

### Pattern 3: Batch 50% discount + async results
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing#pricing
- **Doc says:** "All usage is charged at 50% of the standard API prices." Batches complete within 1 hour in most cases; results available for 29 days; expire after 24 hours if not complete.
- **Status:** missing
- **Evidence:** `autonomous_loop.py:436` uses synchronous `client.messages.create()` at full price.
- **Deviation:** Overnight sweep (non-urgent, results needed before market open) is a textbook batch candidate — low urgency + high volume = 50% discount eligible.
- **Risk:** Sustained overpayment on overnight analysis; at 50+ tickers nightly the discount is material.
- **Recommended fix:** Wrap overnight sweep in `client.messages.batches.create()`; poll with exponential backoff; stream JSONL results via `client.messages.batches.results(batch_id)`.
- **MF-# mapping:** MF-34

---

### Pattern 4: Batch JSONL result format + retry semantics
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing
- **Doc says:** Results streamed as JSONL; each line has `custom_id` and either `result.type == "succeeded"` with a full Messages response, or `result.type == "errored"` with an error object. No built-in retry — caller must re-submit errored requests.
- **Status:** N/A (feature not used)
- **Evidence:** No batch result parsing code exists.
- **Deviation:** Current sequential loop retries at call-site; batch model requires per-`custom_id` error inspection.
- **Risk:** If batch adopted without error handling, errored tickers silently drop.
- **Recommended fix:** After polling completion, filter `result.type == "errored"` lines and re-queue those `custom_id`s as a second batch or individual calls.
- **MF-# mapping:** MF-34

---

### Pattern 5: Batch ZDR ineligibility
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing (top Note block)
- **Doc says:** "This feature is **not** eligible for Zero Data Retention (ZDR)."
- **Status:** missing (risk not documented anywhere in codebase)
- **Evidence:** No ZDR flag in `autonomous_loop.py` or any batch-related code.
- **Deviation:** If org has ZDR and adopts Batches API, ZDR guarantee is broken for those requests.
- **Risk:** Compliance failure if ZDR is contractually required. Batches API retains data according to standard retention policy even under ZDR agreements.
- **Recommended fix:** Document ZDR ineligibility in `autonomous_loop.py` module docstring before enabling. Confirm with Peder whether ZDR applies to this workspace.
- **MF-# mapping:** MF-34, cross-cutting ZDR

---

### Pattern 6: Batch not on Bedrock/Vertex
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/batch-processing
- **Doc says:** Message Batches API is Anthropic first-party only; not available on Amazon Bedrock or Google Vertex AI.
- **Status:** N/A (consistent — code uses direct `anthropic.Anthropic()` for Claude calls)
- **Evidence:** `autonomous_loop.py:419` — `client = anthropic.Anthropic(api_key=api_key)` — direct Anthropic, not Vertex. Compatible.
- **Deviation:** None; however `llm_client.py` routes some Claude model calls through GitHub Models (OpenAI-compat endpoint) — those cannot use Batches API.
- **Risk:** If GitHub Models route is chosen for a batch-eligible model, batches silently unavailable.
- **Recommended fix:** Gate batch usage to calls that use `ClaudeClient` (direct Anthropic), not `OpenAIClient` with GitHub Models base_url.
- **MF-# mapping:** MF-34

---

### Pattern 7: Files API beta header requirement
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files
- **Doc says:** "To use the Files API, you'll need to include the beta feature header: `anthropic-beta: files-api-2025-04-14`." Required on both upload calls (`/v1/files`) and message calls that reference a `file_id`.
- **Status:** missing
- **Evidence:** `grep -rn 'files-api-2025-04-14\|beta.files' backend/ --include='*.py'` — no output. `ClaudeClient.generate_content()` sends no `betas` parameter (llm_client.py:613-630).
- **Deviation:** No beta header wiring in `ClaudeClient`. To use Files API, `client.beta.messages.create(..., betas=["files-api-2025-04-14"])` must replace `client.messages.create()`.
- **Risk:** Any attempt to pass `{type:"document",source:{type:"file",file_id:"..."}}` without the header returns a 400 error.
- **Recommended fix:** Add `betas: list[str]` parameter to `ClaudeClient.generate_content()`; pass through to SDK call when non-empty. Default to `[]`.
- **MF-# mapping:** MF-33

---

### Pattern 8: Files API — size limits (500 MB/file, 500 GB/org)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files#file-storage-and-limits
- **Doc says:** "Maximum file size: 500 MB per file. Total storage: 500 GB per organization."
- **Status:** N/A (feature not used)
- **Evidence:** No file upload code exists.
- **Deviation:** Earnings transcripts (GCS JSON) and SEC Form 4 XMLs are small (< 1 MB each) — well within per-file limit. 10-K PDFs from GCS could be 10-50 MB — also within limit.
- **Risk:** No immediate size risk at current data volume. Must gate 500 MB hard limit if large 10-K bundles adopted.
- **Recommended fix:** Add pre-upload check: `assert os.path.getsize(path) < 500 * 1024 * 1024` before `client.beta.files.upload()`.
- **MF-# mapping:** MF-33

---

### Pattern 9: Files API — MIME whitelist and content block type mapping
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files#file-types-and-content-blocks
- **Doc says:** PDF → `application/pdf` → `document` block; plain text → `text/plain` → `document` block; images → `image/jpeg|png|gif|webp` → `image` block. CSV/XLSX/DOCX/MD not supported as document blocks — must convert to plain text.
- **Status:** N/A (feature not used)
- **Evidence:** `earnings_tone.py` passes transcript as plain JSON string — not a document block. `sec_insider.py` passes parsed XML fields as f-string — not a document block.
- **Deviation:** Both call sites could benefit from `document` blocks (MF-33 scope), but currently bypass the Files API entirely.
- **Risk:** SEC Form 4 XML and earnings JSONs that contain embedded HTML/special chars may corrupt f-string prompts. Document blocks would sanitize this.
- **Recommended fix:** For MF-33: convert SEC Form 4 XML to plain text or PDF; upload via Files API; reference `file_id` in `{type:"document",source:{type:"file",file_id:...}}` block. Earnings transcripts: `text/plain` upload.
- **MF-# mapping:** MF-33, MF-34

---

### Pattern 10: Files API — file_id reuse and lifecycle
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files#file-lifecycle
- **Doc says:** "Files are scoped to the workspace of the API key. Other API keys can use files created by any other API key associated with the same workspace. Files persist until you delete them."
- **Status:** N/A (feature not used)
- **Evidence:** No `file_id` caching or persistence layer exists. Each analysis re-serializes documents from scratch.
- **Deviation:** Major missed opportunity: same 10-K PDF or earnings transcript re-uploaded (as inline base64 or string) on every analysis cycle. Files API would allow upload-once, reference-many.
- **Risk:** Token cost inflation — inline base64 PDFs are charged as input tokens on every call. A 10-MB 10-K at ~1.25M tokens × $15/MTok = $18.75 per call, versus upload-once free + reuse at token cost only.
- **Recommended fix:** Add a `file_id_cache` dict (or BQ table) keyed by `{ticker}_{filing_date}_{doc_hash}`. Upload on first use, cache `file_id`, reuse on subsequent calls. Evict via Files API delete when stale.
- **MF-# mapping:** MF-33

---

### Pattern 11: Files API — download restriction
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files#downloading-a-file
- **Doc says:** "You can only download files that were created by skills or the code execution tool. Files that you uploaded cannot be downloaded."
- **Status:** N/A (feature not used)
- **Evidence:** No download code exists.
- **Deviation:** None — no download requirement in current architecture.
- **Risk:** If future code execution tool generates output files (charts, tables), those CAN be downloaded. User-uploaded PDFs/transcripts cannot — original source must be retained separately.
- **Recommended fix:** When implementing MF-33, retain original document in GCS as source of truth; treat Files API as a token-efficiency cache, not a document store.
- **MF-# mapping:** MF-33

---

### Pattern 12: Files API — ZDR ineligibility
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files (top Note block)
- **Doc says:** "This feature is **not** eligible for Zero Data Retention (ZDR)."
- **Status:** missing (risk not documented)
- **Evidence:** No ZDR flag anywhere in codebase.
- **Deviation:** Adopting Files API for SEC filings or earnings transcripts would store those documents on Anthropic infrastructure beyond response time.
- **Risk:** SEC filings contain MNPI-adjacent data. Storing them in Anthropic's file storage under a non-ZDR policy may conflict with data governance requirements.
- **Recommended fix:** Confirm ZDR posture with Peder before enabling Files API for any regulated document type. Consider using inline base64 (ZDR-eligible) for sensitive filings even if it costs more tokens.
- **MF-# mapping:** MF-33, cross-cutting ZDR

---

### Pattern 13: Files API — not on Bedrock/Vertex
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/files
- **Doc says:** "The Files API is currently not supported on Amazon Bedrock or Google Vertex AI."
- **Status:** N/A (consistent — direct Anthropic used for Claude calls)
- **Evidence:** `autonomous_loop.py:419`, `multi_agent_orchestrator.py:165` — both use `anthropic.Anthropic()` directly. No Vertex routing for Claude calls.
- **Deviation:** None.
- **Risk:** None currently. Same caveat as Pattern 6 — GitHub Models route cannot use Files API.
- **MF-# mapping:** MF-33

---

### Pattern 14: Citations — enabled flag on document blocks
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations
- **Doc says:** `"citations": {"enabled": true}` on each document block. "Currently, citations must be enabled on all or none of the documents within a request."
- **Status:** missing
- **Evidence:** `grep -rn "citations.*enabled" backend/ --include='*.py'` — no output. No document blocks exist (Pattern 7 confirms zero Files API usage; all docs passed as plain strings).
- **Deviation:** No `citations` key anywhere. All ingestion passes text as plain string content in user message — cannot be cited.
- **Risk:** MF-31 (citations on SEC filings) requires document block wrapping AND `citations.enabled=true` as prerequisites. Neither precondition is met.
- **Recommended fix:** Wrap SEC filing text in `{type:"document",source:{type:"text",media_type:"text/plain",data:text},citations:{enabled:True}}`. Note: all-or-none constraint means every document in the same request must also have citations enabled.
- **MF-# mapping:** MF-31

---

### Pattern 15: Citations — location types (char_location vs page_location vs content_block_location)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations
- **Doc says:** Plain text docs → `char_location` (0-indexed, exclusive end). PDF docs → `page_location` (1-indexed, exclusive end). Custom content docs → `content_block_location` (0-indexed, exclusive end from content list).
- **Status:** missing
- **Evidence:** `grep -rn 'char_location\|page_location\|content_block_location' backend/ --include='*.py'` — no output.
- **Deviation:** No citation location parsing exists. MF-31 implementation will need to branch on document type to render correct location type in UI.
- **Risk:** If plain text is used (as expected for Form 4 XML), responses return `char_location` — UI must be able to highlight character ranges. If PDF native (MF-34) is used, responses return `page_location` — UI needs page number rendering.
- **Recommended fix:** Build a `render_citation(citation: dict) -> str` helper that dispatches on `citation["type"]` before implementing MF-31.
- **MF-# mapping:** MF-31, MF-34

---

### Pattern 16: Citations — cited_text is free (no output tokens)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations#token-costs
- **Doc says:** "The `cited_text` field is provided for convenience and does not count towards output tokens. When passed back in subsequent conversation turns, `cited_text` is also not counted towards input tokens."
- **Status:** N/A (feature not used)
- **Evidence:** No citation code exists.
- **Deviation:** Current prompt-based evidence extraction emits full quotes in output text — charged as output tokens. Citations feature eliminates this cost for extracted evidence.
- **Risk:** Leaving output token savings on the table. For SEC filing analysis with multi-sentence quotes, this is measurable.
- **Recommended fix:** Replace prompt instructions like "quote the relevant passage" with `citations:{enabled:true}` on document block. Let the API extract `cited_text` for free.
- **MF-# mapping:** MF-31

---

### Pattern 17: Citations — incompatibility with structured outputs
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations (Warning block)
- **Doc says:** "Citations cannot be used together with Structured Outputs. If you enable citations on any user-provided document and also include the `output_config.format` parameter (or deprecated `output_format`), the API will return a 400 error."
- **Status:** N/A (feature not used — but HIGH risk when MF-31 lands)
- **Evidence:** `ClaudeClient.generate_content()` (llm_client.py:581-630) injects JSON schema as system prompt when `response_schema` is provided, but does NOT use Anthropic's native `output_config.format`. The incompatibility does not apply to prompt-injected JSON mode. Confirmed: no `output_config` or `output_format` in `llm_client.py`.
- **Deviation:** No incompatibility with current implementation. However, if native structured outputs are added in the future alongside citations, 400 errors will occur.
- **Risk:** Low current risk. Future risk: any developer adding `output_config.format` to a call that also uses document blocks with citations will get a silent 400.
- **Recommended fix:** Add comment in `ClaudeClient.generate_content()` — "NOTE: if citations are enabled on document blocks, do not pass output_config.format — see MF-31 constraint."
- **MF-# mapping:** MF-31

---

### Pattern 18: Citations — compatible with prompt caching
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations#using-prompt-caching-with-citations
- **Doc says:** "Citations and prompt caching can be used together effectively. Apply `cache_control` to your top-level document content blocks." Citation response blocks themselves cannot be cached, but source documents can.
- **Status:** partial
- **Evidence:** `ClaudeClient` already wires `cache_control: {type: ephemeral}` on the system prompt (llm_client.py:603-611). No document blocks exist yet, so document-level caching is not tested. System-level caching is correctly implemented.
- **Deviation:** When MF-31 document blocks are added, `cache_control` must be placed on the document block itself, not just the system prompt, to cache document content across calls for the same ticker.
- **Risk:** Without document-level cache_control, each Claude call re-encodes and re-ingests the entire SEC filing, negating the Files API efficiency gains from Pattern 10.
- **Recommended fix:** When wrapping SEC filings in document blocks, add `cache_control: {type: ephemeral}` on the document block, not only on the system prompt.
- **MF-# mapping:** MF-31, MF-33

---

### Pattern 19: Citations — streaming (citations_delta event)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations#streaming-support
- **Doc says:** Streaming adds a `citations_delta` event type containing a single citation to be appended to the current text block's `citations` list.
- **Status:** N/A (ClaudeClient does not use streaming)
- **Evidence:** `ClaudeClient.generate_content()` uses `client.messages.create()` (non-streaming). No `stream=True` or SSE handling anywhere.
- **Deviation:** None — citations_delta only relevant if streaming is added.
- **Risk:** No risk. If streaming is added in a future phase, citation delta handling must be wired alongside text_delta handling.
- **MF-# mapping:** MF-31

---

### Pattern 20: Search results — content block schema
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/search-results
- **Doc says:** `{type:"search_result", source:"<url or id>", title:"<title>", content:[{type:"text",text:"..."}], citations:{enabled:true}}`. Fields `type`, `source`, `title`, `content` are required. `citations` and `cache_control` are optional.
- **Status:** missing
- **Evidence:** `grep -rn 'type.*search_result' backend/ --include='*.py'` — no output. Step 3 RAG uses Vertex AI Search grounding (Gemini-only) and returns Gemini grounding metadata, not `search_result` blocks.
- **Deviation:** The RAG pipeline (`orchestrator.py:492-498`) runs through `self.rag_client` (a `GeminiClient`) against a Vertex AI Search data store. This is architecturally separate from Anthropic search_result blocks. Claude calls never see document retrieval results as structured content blocks.
- **Risk:** MF-32 scope: if BQ RAG rows or Vertex Search results are surfaced to Claude (for synthesis or MAS planning), they arrive as `json.dumps()`-stringified text, not citable `search_result` blocks. Citation quality is degraded.
- **Recommended fix:** For BQ RAG rows passed to Claude: wrap each row in `{type:"search_result",source:"bigquery://pyfinagent_data.{table}",title:"{ticker} {date}",content:[{type:"text",text:json.dumps(row)}]}`. Claude will then cite specific rows.
- **MF-# mapping:** MF-32

---

### Pattern 21: Search results — supported models
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/search-results
- **Doc says:** Supported: Opus 4.7, 4.6, 4.5, 4.1; Sonnet 4.6, 4.5, 3.7; Haiku 4.5, 3.5.
- **Status:** N/A (feature not used — but architecture check)
- **Evidence:** `autonomous_loop.py:438` uses `claude-sonnet-4-20250514` (deprecated). `multi_agent_orchestrator.py:148` uses `claude-opus-4-6`, line 1112 uses `claude-sonnet-4-6`. Both `claude-opus-4-6` and `claude-sonnet-4-6` are on the supported list.
- **Deviation:** `claude-sonnet-4-20250514` in `autonomous_loop.py` is deprecated — should be updated to `claude-sonnet-4-6` or `claude-sonnet-4-5-20250929` regardless of search_result adoption.
- **Risk:** Deprecated model may be sunset during go-live window.
- **Recommended fix:** Update `autonomous_loop.py:438` to `claude-sonnet-4-6` (already used elsewhere in the codebase).
- **MF-# mapping:** MF-32, general model hygiene

---

### Pattern 22: Search results — placement (top-level vs tool_result)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/search-results
- **Doc says:** Search result blocks can appear (1) directly in user message content array (pre-fetched data), or (2) wrapped in a `tool_result` block (dynamic RAG via tool calls).
- **Status:** N/A (feature not used)
- **Evidence:** No tool_result blocks or search_result blocks in any Claude call site.
- **Deviation:** Current MAS orchestrator tool use (if any) passes results as plain text. BQ rows are stringified JSON in the prompt.
- **Risk:** Tool-result-wrapped search_result blocks enable dynamic citation of retrieved data — a significant RAG quality upgrade for the MAS planning agents.
- **Recommended fix:** For MF-32: use top-level placement for pre-fetched BQ signal rows; use tool_result wrapping for real-time retrieval during MAS reasoning loops.
- **MF-# mapping:** MF-32

---

### Pattern 23: Search results — ZDR eligible
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/search-results (top Note block)
- **Doc says:** "This feature is eligible for Zero Data Retention (ZDR)."
- **Status:** N/A (consistent with ZDR preservation — feature not used but would be safe)
- **Evidence:** Feature not implemented.
- **Deviation:** None — search_result blocks are ZDR-safe unlike the other three features.
- **Risk:** None. This is the only one of the four audited features that does NOT break ZDR.
- **Recommended fix:** If ZDR is required, prefer `search_result` blocks (MF-32) over Files API document blocks (MF-33) for surfacing retrieved content to Claude.
- **MF-# mapping:** MF-32, cross-cutting ZDR

---

### Pattern 24: PDF native document block (base64 inline)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations#pdf-documents
- **Doc says:** `{type:"document",source:{type:"base64",media_type:"application/pdf",data:<b64>},citations:{enabled:True}}`. Citation format: `page_location` with 1-indexed `start_page_number`/`end_page_number` (exclusive end).
- **Status:** missing
- **Evidence:** `grep -rn 'base64.*application/pdf\|media_type.*pdf' backend/ --include='*.py'` — no output. `fpdf2` is in requirements.txt (for PDF generation only, not ingestion).
- **Deviation:** 10-K and 10-Q filings (referenced in RAG via Vertex AI Search) are never passed to Claude as PDF document blocks. `earnings_tone.py` fetches transcripts as HTML-scraped text. No base64 PDF encoding anywhere.
- **Risk:** MF-34 requires PDF native ingestion. Without it, multi-page citations (page_location) are unavailable. Scanned PDFs without extractable text will not be citable even with MF-34.
- **Recommended fix:** For MF-34: fetch 10-K PDF from GCS, `base64.b64encode(pdf_bytes).decode()`, pass in document block with `citations:{enabled:True}`. Fallback: if PDF is scanned-only, extract text layer first (pdfminer/pymupdf), pass as plain text document for `char_location` citations.
- **MF-# mapping:** MF-34

---

### Pattern 25: Document block — title/context fields (not citable)
- **Doc source:** https://platform.claude.com/docs/en/build-with-claude/citations#citable-vs-non-citable-content
- **Doc says:** "Text found within a document's `source` content can be cited from. `title` and `context` are optional fields that will be passed to the model but not used towards cited content."
- **Status:** N/A (feature not used)
- **Evidence:** No document blocks exist.
- **Deviation:** Current prompt templates inject ticker, sector, date as f-string preamble in the prompt body — these would become citable if moved to `source.data`. Metadata (filing date, CIK) should be in `context` (not citable) rather than embedded in `source.data`.
- **Risk:** If metadata (CIK, date, filing type) is placed in `source.data`, Claude may incorrectly cite it as a claim-bearing statement.
- **Recommended fix:** Place document content only in `source.data`; move filing metadata to `context` field as stringified JSON.
- **MF-# mapping:** MF-31, MF-33, MF-34

---

## ZDR impact summary

| Feature | ZDR eligible? | Impact if adopted |
|---------|---------------|-------------------|
| Batches API | NO | Org loses ZDR for all batch requests |
| Files API | NO | Org loses ZDR for all requests referencing a file_id |
| Citations (document blocks) | YES | ZDR preserved |
| Search results (search_result blocks) | YES | ZDR preserved |

**Key takeaway:** MF-31 (citations on document blocks) and MF-32 (search_result blocks) are ZDR-safe. MF-33 (Files API) and MF-34 (Batches API for overnight sweep) break ZDR. If the org has a ZDR arrangement, MF-33 and MF-34 require explicit approval from Peder before enabling. As a ZDR-preserving alternative for MF-33, documents can be passed inline as base64 (PDF) or plain text in each request rather than uploaded to Files API — this costs more tokens per call but maintains ZDR.

---

## Prerequisites before MF-31/MF-33/MF-34 land

1. **MF-31 (Citations):** Requires document block wrapping first (Pattern 14). Requires confirming no native `output_config.format` usage (Pattern 17). Requires `cache_control` on document blocks (Pattern 18).
2. **MF-32 (search_result):** Requires BQ row retrieval to be decoupled from prompt f-string injection. Update deprecated model in `autonomous_loop.py:438` (Pattern 21).
3. **MF-33 (Files API):** Requires `betas` parameter in `ClaudeClient` (Pattern 7). Requires ZDR sign-off (Pattern 12). Requires file_id cache layer (Pattern 10).
4. **MF-34 (PDF native + Batches):** Requires base64 PDF ingestion path (Pattern 24). Requires Batches API wiring in `autonomous_loop.py` (Pattern 1). ZDR sign-off required (Pattern 5).

---

## Sources

- https://platform.claude.com/docs/en/build-with-claude/batch-processing
- https://platform.claude.com/docs/en/api/messages/batches
- https://platform.claude.com/docs/en/api/creating-message-batches
- https://platform.claude.com/docs/en/build-with-claude/files
- https://platform.claude.com/docs/en/api/files-create
- https://platform.claude.com/docs/en/build-with-claude/citations
- https://platform.claude.com/docs/en/build-with-claude/search-results
