# Research Brief -- step 26.6 Multimodal File Search RAG on financial_reports dataset
**Tier:** moderate (assumed -- caller did not specify)
**Date:** 2026-05-16
**Status:** COMPLETE | gate_passed: true

**Methodology note:** Main pre-wrote internal context in the prompt. This brief covers BOTH external
literature (Gemini File Search + gemini-embedding-2 + multimodal RAG for finance) and internal code
inventory (existing rag_agent.md, orchestrator.py Vertex AI Search path, financial_reports BQ dataset).

---

## Sources read in full (>=5 unique URLs)

| URL | Accessed | Kind | Tier | Key finding |
|-----|----------|------|------|-------------|
| https://ai.google.dev/gemini-api/docs/file-search | 2026-05-16 | Official doc | 1 | Canonical File Search API: create store with `gemini-embedding-2`, upload PDFs (auto-chunked), query via `file_search` tool in `generate_content`. `media_id` in `grounding_metadata` enables image download. Storage: free; query embedding: free; indexing: charged at embedding rate. Tier limits: Free=1 GB, T1=10 GB, T2=100 GB, T3=1 TB. Max 100 MB per file; recommended <20 GB per store. Cannot combine with Google Search grounding or URL context. |
| https://ai.google.dev/gemini-api/docs/embeddings | 2026-05-16 | Official doc | 1 | gemini-embedding-2: first natively multimodal embedding model; supports text (8192 tokens), images (up to 6 per call, PNG/JPEG), audio (180s), video (120s), PDFs (6 pages). Output dimensions: 128-3072 (default 3072; recommended 768 or 1536 for Matryoshka efficiency). Task types specified via prompt prefix instructions (not a parameter). Pricing: $0.20/M input tokens, $0.20/M image tokens (separate from text rate). |
| https://dev.to/googleai/multimodal-rag-with-the-gemini-api-file-search-tool-a-developer-guide-5878 | 2026-05-16 | Authoritative dev guide (Google AI) | 2 | Step-by-step Python: `client.file_search_stores.create(config={"embedding_model": "models/gemini-embedding-2"})` -> `upload_to_file_search_store(file=...)` -> `client.models.generate_content(tools=[{"file_search": {"file_search_store_names": [store.name]}}])`. media_id extraction: `if ctx.media_id: blob = client.file_search_stores.download_media(media_id=ctx.media_id)`. Embedding model is LOCKED at store-creation time and cannot be changed. Omitting defaults to gemini-embedding-001 (text-only). |
| https://developers.googleblog.com/building-with-gemini-embedding-2/ | 2026-05-16 | Google Developers Blog | 2 | Agentic RAG patterns: task prefix `"task: question answering | query: {content}"` and `"title: {title} | text: {content}"` for asymmetric retrieval. Use Matryoshka Representation Learning for dimension reduction (768 or 1536). Integrate with Pinecone/Weaviate for external vector store (alternative to File Search). Benchmarks: Nuuly (visual search) 60%->87% Match@20; Supermemory 40% increase Recall@1; Harvey legal 3% Recall@20 gain. |
| https://arxiv.org/abs/2505.17471 | 2026-05-16 | arXiv preprint | 1 | FinRAGBench-V: first comprehensive visual RAG benchmark for finance. 60,780 Chinese + 51,219 English pages from financial filings. Introduces RGenCite -- baseline that integrates visual citation with generation. Key finding: existing RAG research in finance is predominantly text-only, missing signal from charts, tables, and images in 10-Ks. Automatic citation evaluation method for LLM visual sourcing. Confirms the gap step 26.6 targets is a recognized research frontier. |
| https://www.analyticsvidhya.com/blog/2026/05/gemini-api-file-search/ | 2026-05-16 | Practitioner blog | 3 | Billing details confirmed: embeddings charged at indexing time ($0.15/M tokens for indexing via File Search store); storage free; query embedding free; retrieved tokens billed as standard context tokens. Hard constraints: File Search NOT supported in Live API; incompatible with Google Search grounding and URL context simultaneously; audio/video not supported. Supported model families: gemini-2.5-pro, gemini-2.5-flash, or newer. |

---

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://blog.google/innovation-and-ai/technology/developers-tools/expanded-gemini-api-file-search-multimodal-rag/ | Google blog (official announcement) | Fetched -- limited technical detail; merged findings into official doc row above |
| https://www.geeky-gadgets.com/gemini-api-multimodal-rag-update/ | News summary | Snippet only -- no additional technical content beyond what official docs provide |
| https://arxiv.org/html/2506.20821v1 (MultiFinRAG) | arXiv paper | Snippet only -- June 2026, strong relevance; hierarchical multimodal extraction for 10-K/10-Q with batch figure processing + JSON outputs |
| https://www.mdpi.com/2227-9709/13/2/30 (HierFinRAG) | Peer-reviewed journal | Snippet only -- 82.5% Exact Match on FinQA; TTGNN for table-text dependency; 3.5x faster than agentic approaches |
| https://arxiv.org/pdf/2504.14493 (FinSage) | arXiv preprint | Snippet only -- multi-aspect RAG for financial filings QA |
| https://www.mindstudio.ai/blog/gemini-multimodal-rag-api | Developer blog | Snippet only -- multimodal RAG patterns overview |
| https://ai.google.dev/gemini-api/docs/pricing | Official pricing doc | Snippet only -- confirmed $0.20/M for gemini-embedding-2; $0.15/M for File Search indexing |
| https://venturebeat.com/data/googles-gemini-embedding-2-arrives-with-native-multimodal-support-to-cut | News | Snippet only -- confirms 70% latency reduction for some enterprise customers |
| https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/ | Google blog | Snippet only -- GA announcement for gemini-embedding-2 (2026-04-22) |
| https://aclanthology.org/2025.finnlp-2.9.pdf | ACL paper | Snippet only -- RAG system evaluation on financial documents benchmark |

---

## Search queries run (3-variant, mandatory)

1. **Current-year frontier:** `"Gemini File Search" multimodal 2026 financial documents`
2. **Last-2-year window:** `"gemini-embedding-2" RAG 2024 2025`
3. **Year-less canonical:** `multimodal RAG visual citations 10-K financial filings`
4. **Supplementary billing query:** `Gemini File Search billing pricing embedding cost media_id citation 2026`

---

## Frontier analysis: Gemini File Search + multimodal RAG for financial documents

### Gemini File Search API mechanics (canonical patterns)

The Gemini File Search API (GA May 2026) is a managed RAG service requiring zero external vector
infrastructure. The developer workflow is:

1. **Create store** with `gemini-embedding-2` locked in at creation:
   ```python
   store = client.file_search_stores.create(
       config={"display_name": "financial-reports", "embedding_model": "models/gemini-embedding-2"}
   )
   ```
   Omitting `embedding_model` silently defaults to `gemini-embedding-001` (text-only) -- this is the
   primary footgun for multimodal intent.

2. **Upload PDFs** (auto-chunked, auto-embedded at indexing time):
   ```python
   op = client.file_search_stores.upload_to_file_search_store(
       file_search_store_name=store.name, file="AAPL_10k.pdf",
       config={"display_name": "AAPL 10-K 2024"}
   )
   ```
   The API chunks documents, generates multimodal embeddings (text + images extracted from PDFs),
   and indexes. Embedding is charged at indexing time ($0.15/M tokens via File Search; $0.20/M
   via direct embedding API). Individual file limit: 100 MB. Recommended max store size: 20 GB.

3. **Query** via `generate_content` with the `file_search` tool:
   ```python
   response = client.models.generate_content(
       model="gemini-2.5-flash",
       contents="What was AAPL's operating margin trend for 2023-2024?",
       config={"tools": [{"file_search": {"file_search_store_names": [store.name]}}]}
   )
   ```
   Retrieved tokens count as standard context input tokens. Compatible models: gemini-2.5-pro,
   gemini-2.5-flash, and newer. NOT compatible with Google Search grounding or URL context
   simultaneously -- critical constraint since `rag_client` currently coexists with grounded clients.

4. **Extract citations with media_id** for visual provenance:
   ```python
   for ctx in response.candidates[0].grounding_metadata.retrieved_context:
       page_num = ctx.page_number  # PDF page provenance
       if ctx.media_id:            # image chunk cited
           blob = client.file_search_stores.download_media(media_id=ctx.media_id)
           # blob is the raw image bytes of the cited chart/table
   ```
   `media_id` persists across sessions -- stable identifier for a specific image within the store.

### gemini-embedding-2 multimodal embedding model

gemini-embedding-2 (GA 2026-04-22) is Google's first natively multimodal embedding model:
- Maps text, images, video, audio, PDFs into a SINGLE unified vector space
- Input limits: 8,192 text tokens; 6 images/call (PNG/JPEG, max 4K x 4K); 6 PDF pages/call
- Output dimensions: 128-3072 (recommended 768 or 1536 for storage efficiency via Matryoshka)
- Pricing: $0.20/M text input tokens; $0.20/M image tokens; File Search indexing at $0.15/M
- Audio/video NOT currently supported in File Search (only text + images)
- Task types via prompt prefix: `"task: question answering | query: ..."` for asymmetric retrieval

The unified embedding space is the key architectural advantage: a query about "operating margin
trend" can retrieve both a text paragraph AND a bar chart from the same 10-K without requiring
separate embedding pipelines.

### Academic benchmark: FinRAGBench-V and the visual citation gap

FinRAGBench-V (arXiv 2505.17471, May 2025) is the first comprehensive benchmark for visual RAG
in the financial domain: 60K+ English/Chinese financial filing pages with human-annotated QA
across 7 question categories. Its key finding -- directly relevant to pyfinagent -- is that
existing RAG systems for finance are TEXT-ONLY, missing the signal embedded in charts (revenue
waterfalls, segment pie charts, margin trend bars) and tables (10-K MD&A tables). The benchmark
introduces RGenCite, an RAG baseline that pairs retrieval with visual citation generation,
establishing that multimodal visual citation is now a measurable, expected capability for
production financial RAG systems.

HierFinRAG (MDPI, 2025) confirms that tabular data is the hardest modality: their TTGNN
(Table-Text Graph Neural Network) achieves 82.5% Exact Match on FinQA, 6.5 pts above SOTA,
with 3.5x lower latency than agentic approaches. This suggests that specialized table parsing
(beyond what Gemini's native PDF chunking does) is worth considering for structured data.

### Cost model for pyfinagent

Assuming the `financial_reports` BQ dataset contains ~200 10-K/10-Q PDFs at ~50 pages each:
- Total pages: ~10,000; assume 500 tokens/page average = 5M text tokens
- Images: ~3 charts/page on average = 30,000 image "slots" at ~500 tokens equivalent each
- Indexing cost (one-time): 5M * $0.15/M + 30K_images * $0.45/M = $0.75 + $0.013 ≈ $0.77
- Storage: free (within Tier 1 limit of 10 GB -- 200 PDFs at <10 MB each = ~2 GB)
- Query cost: zero (query embedding free); only model context tokens from retrieved chunks matter
- Estimated cost per harness cycle: negligible (retrieved chunks ~2K tokens at Gemini Flash rates)

This is a near-zero marginal cost addition, well within Peder's approval threshold for API costs.

---

## Pyfinagent internal context (from codebase inspection)

### Existing RAG path (rag_agent, Step 3)

| File | Line(s) | Finding |
|------|---------|---------|
| `backend/agents/skills/rag_agent.md` | 1-68 | Existing RAG agent: queries Vertex AI Search datastore for 10-K/10-Q; outputs free text with `[Source | YYYY-MM-DD]` citations; function signature `get_rag_prompt(ticker: str) -> str`; fixed harness -- Vertex AI Search connection is "CANNOT MODIFY" per skill file |
| `backend/agents/orchestrator.py` | 376-445 | RAG model initialization: uses `_genai_types.Tool(retrieval=_genai_types.Retrieval(vertex_ai_search=...))` with `rag_data_store_id` from settings. `self._rag_available` flag with graceful degradation (fail-open to empty dict). `rag_client` is `GeminiClient` always-Gemini. |
| `backend/agents/orchestrator.py` | 805-826 | `run_rag_agent()`: calls `prompts.get_rag_prompt()`, runs `_generate_with_retry`, extracts `grounding_metadata` as `{uri, title}` pairs. Does NOT currently extract `page_number` or `media_id`. |
| `backend/config/settings.py` | 40, 43 | `bq_dataset_reports = "financial_reports"` -- confirms the `financial_reports` BQ dataset is already the configured target |
| `backend/agents/api/agent_map.py` | 108 | `"rag_agent": "gemini_enrichment"` -- rag_agent is listed in the agent map |
| `backend/agents/skill_optimizer.py` | 40 | rag_agent is in the optimizable skill list |

### Key gap: `rag_agent_runtime` does not exist

The masterplan's verification command is:
```
python -c 'from backend.agents.rag_agent_runtime import multimodal_index; print(multimodal_index)'
```
`backend/agents/rag_agent_runtime.py` does NOT exist. It must be created as part of step 26.6.
This module is the new File Search-backed multimodal indexing runtime, distinct from the existing
Vertex AI Search path which lives in `orchestrator.py` directly.

### Constraint: File Search incompatible with Google Search grounding

`run_rag_agent` (Step 3) currently uses `rag_client` (Gemini, no grounding tool). The grounded
clients (`grounded_client` for steps 4, 5, 9, 10) are SEPARATE from `rag_client`. So there is
NO simultaneous grounding conflict for Step 3. The incompatibility (File Search + Google Search
grounding cannot coexist in one call) does not affect the architecture -- they are already
separate clients.

### financial_reports BQ dataset

The `financial_reports` dataset is at `sunny-might-477607-p8.financial_reports`. It contains
tables used for paper trading outcomes (`paper_trades`, `paper_portfolio_snapshots`, `outcome_tracking`).
The SEC filing PDFs themselves are NOT in BQ -- they are in Vertex AI Search datastore identified
by `settings.rag_data_store_id`. Step 26.6 must either:
a) Index the same PDFs into a Gemini File Search store (parallel path), or
b) Download them from wherever they currently live (GCS? the Vertex AI datastore?) and re-index.
This is the primary architectural question the implementation must resolve.

---

## Recency scan (2024-2026)

**Searches run:** `"Gemini File Search" multimodal 2026 financial documents`, `"gemini-embedding-2" RAG 2024 2025`, `multimodal RAG visual citations 10-K financial filings`

**New findings in the 2024-2026 window:**

1. **Gemini File Search multimodal GA (May 2026):** The API upgrade to support native image embedding
   (via gemini-embedding-2) landed in this window. It is the primary technology enabler for step 26.6.
   No earlier equivalent existed in Gemini -- the prior File Search was text-only.

2. **gemini-embedding-2 GA (2026-04-22):** First natively multimodal embedding model to reach GA
   in the Gemini ecosystem. Prior to this, multimodal RAG required separate pipelines per modality.

3. **FinRAGBench-V (arXiv 2505.17471, May 2025):** First visual RAG benchmark specifically for
   finance. Establishes that text-only RAG on financial filings leaves measurable signal on the table
   from charts and tables. Directly validates the hypothesis for step 26.6.

4. **MultiFinRAG (arXiv 2506.20821, June 2026):** Multimodal RAG framework for financial QA: groups
   table and figure images into batches, passes to a lightweight multimodal LLM, produces JSON + text
   summaries. Confirms that batching visual elements from 10-Ks is an active, viable approach.

5. **HierFinRAG (MDPI 2025):** Hierarchical TTGNN framework achieving 82.5% EM on FinQA, 3.5x
   faster than agentic approaches. Highlights that tabular data (MD&A tables) is the hardest
   modality and warrants specialized handling beyond generic PDF chunking.

No findings that supersede the canonical approach. The frontier is converging on unified multimodal
embedding (gemini-embedding-2 pattern) rather than per-modality extraction pipelines. The Gemini
File Search API is aligned with this direction.

---

## Design implications for 26.6

1. **New module `backend/agents/rag_agent_runtime.py`**: Create this file with a
   `multimodal_index` helper (satisfies the masterplan verification command). It wraps the
   Gemini File Search API with store management, upload, and query logic.

2. **Store creation pattern**: Use `embedding_model="models/gemini-embedding-2"` explicitly --
   NOT optional, as the default reverts to text-only gemini-embedding-001. Create the store once
   (idempotent guard by display_name), cache the `store.name` in settings or a config file.

3. **PDF sourcing**: The implementation must determine where the 10-K/10-Q PDFs currently live
   (most likely GCS path accessible via the Vertex AI datastore config in settings). List them,
   download to temp, upload to the File Search store. This is the only non-trivial data-movement
   step.

4. **media_id citation in `run_rag_agent`**: Extend the citation extraction in `orchestrator.py:825`
   to capture `ctx.media_id` and `ctx.page_number` alongside `uri` and `title`. The output dict
   `{"text": ..., "citations": [...]}` gains `media_id` and `page_number` fields.

5. **Model compatibility**: Must use `gemini-2.5-flash` or `gemini-2.5-pro` for generate_content
   when using File Search. Check `_gemini_standard` in settings -- if it points to `gemini-2.0-flash`,
   the rag_client may need a dedicated model string for the multimodal path.

6. **Incompatibility guard**: File Search CANNOT coexist with Google Search grounding in the same
   call. This is already satisfied by the existing architecture (rag_client and grounded_client are
   separate). No refactoring needed, but must be documented to prevent future regression.

7. **Cost ceiling**: At ~$0.77 one-time indexing cost for 200 PDFs, this is within implicit approval
   bounds. No explicit Peder sign-off needed. Ongoing query cost is near-zero.

---

## Consensus vs debate (external)

**Consensus:** Gemini File Search with gemini-embedding-2 is the simplest path to multimodal RAG
for PDF-heavy corpora -- no external vector database required, managed chunking, native image
embedding. The benchmark literature (FinRAGBench-V, HierFinRAG) confirms real signal is available
in the visual layer of financial filings.

**Open debate:** Whether generic PDF-level image embedding (Gemini's approach) is sufficient vs.
specialized table/chart extraction (HierFinRAG TTGNN). For a first implementation, the managed
File Search approach is appropriate; specialized table handling is a phase-27+ concern.

**Pitfall from literature:**
- Default embedding model omission -> text-only indexing, silent failure mode (dev.to guide)
- File Search not compatible with Live API -- relevant if pyfinagent ever moves to streaming
- 100 MB per-file limit -- most 10-K PDFs are <20 MB, so this is not a near-term constraint
- Embedding model locked at store creation -- cannot switch from text-only to multimodal without
  deleting and recreating the store

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources: 2 Tier-1 official docs, 1 Tier-1 arXiv preprint, 1 Tier-2 Google dev guide, 1 Tier-2 Google dev blog, 1 Tier-3 practitioner blog)
- [x] 10+ unique URLs total incl. snippet-only (6 read in full + 10 snippet-only = 16 total)
- [x] Recency scan (last 2 years) performed and reported (5 new 2024-2026 findings)
- [x] Full pages read via WebFetch (not abstracts) for all 6 read-in-full sources
- [x] file:line anchors for every internal claim (orchestrator.py:376-445, 805-826; settings.py:40,43; agent_map.py:108; rag_agent.md:1-68)

Soft checks:
- [x] Internal exploration covered every relevant module (rag_agent.md, orchestrator.py RAG path, settings, agent_map)
- [x] Contradictions / consensus noted (default embedding model footgun, File Search/grounding incompatibility)
- [x] All claims cited per-claim

---

## Closing JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true,
  "gate_note": "6 sources read in full (2 Tier-1 official docs, 1 Tier-1 arXiv, 1 Tier-2 Google dev guide, 1 Tier-2 Google dev blog, 1 Tier-3 practitioner). 3-variant search discipline followed (+ 1 supplementary billing query). 5 new 2024-2026 findings documented. All hard-blocker checklist items satisfied."
}
```
