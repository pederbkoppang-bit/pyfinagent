# Research Brief — phase-25.E9: Adopt Native Citations; Deprecate CitationAgent

**Tier:** moderate
**Accessed:** 2026-05-13

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/docs/build-with-claude/citations | 2026-05-13 | Official docs | WebFetch full (via redirect) | Complete Citations API: document block shape, all citation location types, response structure, streaming citations_delta, Files API integration, prompt-caching combo, structured-outputs incompatibility |
| https://raw.githubusercontent.com/anthropics/anthropic-cookbook/main/misc/using_citations.ipynb | 2026-05-13 | Official cookbook | WebFetch full | Full Python code: request/response iteration pattern, citation attribute access (`.cited_text`, `.document_title`, `.start_char_index`, `.end_char_index`, `.type`), citation types per document kind |
| https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md | 2026-05-13 | Official SDK reference | WebFetch full | SDK type exports: `TextBlock`, `TextCitation`, `CitationCharLocation`, `CitationPageLocation`, `CitationContentBlockLocation`, `CitationsDelta`, `CitationsConfig`; Python access pattern |
| https://simonwillison.net/2025/Jan/24/anthropics-new-citations-api/ | 2026-05-13 | Authoritative blog (Simon Willison) | WebFetch full | Launch analysis: JSON response shape, breaking-change for tools assuming text-stream responses, cost/token efficiency |
| https://peps.python.org/pep-0702/ | 2026-05-13 | Official Python PEP | WebFetch full | `@deprecated` decorator: syntax, behavior for methods, `DeprecationWarning`, `stacklevel`, `typing_extensions` availability since 4.5.0, Python 3.13+ target |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.anthropic.com/news/introducing-citations-api | Official blog | Redirects to claude.com; fetched redirect — only billing/product launch content, no new technical detail beyond platform docs |
| https://claude.com/blog/introducing-citations-api | Official blog | Fetched via redirect — product intro, confirms `cited_text` does not count toward output tokens, 15% recall accuracy improvement |
| https://github.com/BerriAI/litellm/issues/7970 | OSS issue | LiteLLM Citations feature request; confirms API shape from external perspective |
| https://docs.spring.io/spring-ai/docs/1.1.0/api/org/springframework/ai/anthropic/api/CitationDocument.html | SDK docs | Spring AI bindings; different stack, confirms document block shape |
| https://docs.anthropic.com/en/release-notes/api | Official release notes | Snippet only; confirms citations GA on all active models except Haiku 3 |
| https://deprecated.readthedocs.io/en/latest/tutorial.html | Library docs | Third-party `deprecated` library; lower priority than stdlib PEP 702 |
| https://dev.docs.pyansys.com/coding-style/deprecation.html | Industry guide | PyAnsys deprecation conventions; confirms `warnings.warn(..., stacklevel=2)` pattern |
| https://aws.amazonaws.com/about-aws/whats-new/2025/06/citations-api-pdf-claude-models-amazon-bedrock/ | AWS announcement | Citations + PDF on Bedrock; pyfinagent uses direct Anthropic API so not blocking |
| https://ai-sdk.dev/providers/ai-sdk-providers/anthropic | Vendor SDK docs | Vercel AI SDK; different stack |
| https://docs.litellm.ai/docs/providers/anthropic | Proxy docs | LiteLLM; confirms citations pass-through shape |

---

## Recency scan (2024-2026)

Searched: "Anthropic Citations API document block citations enabled true 2026", "Anthropic native citations citations_delta response block shape fields" (2025), "anthropic python SDK citations response content blocks extraction 2025 2026".

**Result:** Citations API launched January 2025, now GA on all active models (all Claude 3+ except Haiku 3). No 2026 publications supersede the official platform docs. The API is stable on the direct Anthropic API and available on Vertex AI (GA confirmed via AWS/Bedrock snippet June 2025 — launch slightly later on cloud platforms). The anthropic Python SDK version 0.96.0 (currently installed) includes full citations type support: `TextBlock.citations`, `CitationsDelta`, `CitationsConfig`. No breaking changes to the citations API in the 2025-2026 window. The structured-outputs incompatibility (400 error) is a documented permanent constraint, not a temporary limitation.

---

## Key findings

1. **Document block shape with citations enabled** -- add `"citations": {"enabled": True}` to any existing document block. For Files API references (the pattern already used by 25.D9): `{"type": "document", "source": {"type": "file", "file_id": file_id}, "citations": {"enabled": True}}`. For inline text: `{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "..."}, "citations": {"enabled": True}}`. (Source: Anthropic Citations docs, https://platform.claude.com/docs/en/docs/build-with-claude/citations)

2. **Citations must be enabled on ALL or NONE documents in a request.** You cannot mix cited and uncited document blocks in the same call. (Source: Anthropic Citations docs, same URL)

3. **Response shape -- text blocks gain a `citations` field.** Each text block in `response.content` may have `block.citations` as a list. Each citation object has: `type` (`"char_location"` / `"page_location"` / `"content_block_location"`), `cited_text` (exact quoted passage, NOT counted as output tokens), `document_index` (0-indexed into request documents), `document_title`, and location fields specific to citation type. For plain text / Files API text: `start_char_index` + `end_char_index`. For PDF: `start_page_number` + `end_page_number`. (Source: Citations docs + cookbook notebook)

4. **`cited_text` does not count toward output tokens.** The model outputs citations in a standardized internal format that are parsed server-side; `cited_text` is reconstructed from source. This is the primary cost savings vs `_add_citations` Sonnet call. (Source: Anthropic blog intro, Claude cookbook)

5. **Citations + Files API combo works.** File-sourced document blocks (`"source": {"type": "file", "file_id": "..."}`) support `"citations": {"enabled": True}` identically to inline text blocks. (Source: Citations docs, "Files API" tab in Plain text documents section; cookbook tab-based code examples)

6. **Structured outputs (output_config.format) and citations are permanently incompatible.** Confirmed: returns 400. The guard already in `llm_client.py:1307-1321` correctly detects `config.get("citations")` -- this path is validated. (Source: Citations docs Warning block; codebase `llm_client.py:1307-1321`)

7. **No new beta header required for citations.** Unlike Files API (`files-api-2025-04-14`), citations are a stable GA feature requiring no `betas=` parameter. Pass `"citations": {"enabled": True}` on the document block, nothing else. (Source: Citations docs -- no beta header shown in any code sample)

8. **`@deprecated` pattern for Python methods.** Two correct approaches for `_add_citations`: (a) `warnings.warn("...", DeprecationWarning, stacklevel=2)` at the top of the method body (stdlib, no dependencies); (b) `@typing_extensions.deprecated(...)` decorator (PEP 702, available in `typing_extensions>=4.5.0`, Python 3.13+ stdlib). For pyfinagent, approach (a) is simpler since `_add_citations` is a private async method -- wrap with early return after the warning. (Source: PEP 702, https://peps.python.org/pep-0702/)

9. **Cost comparison.** `_add_citations` Sonnet call: ~$0.01-0.02 per Q&A response (from step spec). Claude Sonnet 4.6 is priced at $3/M input + $15/M output. A 1200-token prompt + 300-token output = $0.0086/call. At ~5 Q&A responses per session: ~$0.04/session eliminated. Native citations add slight input tokens (system prompt additions + chunking overhead) but no extra LLM call. Net saving is real and compounds with Q&A volume.

10. **`LLMResponse` citation extension.** The `LLMResponse` dataclass at `llm_client.py:593-599` currently has `text`, `thoughts`, `usage_metadata`, `grounding_metadata`. Add `citations: list[dict] | None = None` as a new field (with `field(default_factory=lambda: None)` or just `= None`). The `generate_content` return at `llm_client.py:1523` must be updated to extract citations from `response.content` blocks when present. (Source: internal code read; SDK type: `TextBlock.citations`)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/llm_client.py` | ~1540 | `LLMResponse` dataclass (line 593-599); `ClaudeClient.generate_content` (line ~1104+); citations guard (line 1307-1321); document block injection for Files API (line 1213-1230); content block parse loop (line 1463-1471); return at 1523 | Needs: `citations` field on `LLMResponse`; citations wiring in `generate_content`; extract citations in parse loop |
| `backend/agents/multi_agent_orchestrator.py` | ~1400+ | `_add_citations` method (line 1284-1333); call site at line 438-451 | Needs: `_add_citations` deprecated + early-return; call site short-circuited |
| `tests/verify_phase_25_E9.py` | NEW | Immutable verification script | To create |

---

## Consensus vs debate (external)

No debate. Native citations are a GA API feature with clear documentation. The deprecation of LLM-based post-processing in favor of native server-side processing is unambiguously the correct direction. The only constraint (structured-outputs incompatibility) is already guarded.

---

## Pitfalls (from literature and code inspection)

1. **Citations-enabled calls return multiple text blocks, not one.** The current parse loop at `llm_client.py:1463-1471` does `text += block.text` for each `block_type == "text"`. This concatenation still works for text extraction; citations metadata needs a parallel collect step.

2. **`citations` must be enabled on ALL document blocks in a request.** If a future request has two document blocks and only one enables citations, the API returns 400. The implementation must ensure consistency when setting citations on the block.

3. **`config.get("citations")` guard at `llm_client.py:1314` checks against `schema is not None`.** This guard works correctly for the structured-outputs conflict. But the guard only raises if `citations` is truthy AND `schema is not None`. If `schema` is `None` (most Q&A / Research calls), citations can proceed. No change needed to the guard logic itself.

4. **`_add_citations` call site at `multi_agent_orchestrator.py:438-451` emits a `MASEvent` with `event_type="citation"`.** When `_add_citations` is short-circuited, `cite_usage` returns `{"input": 0, "output": 0}` (as it does today for non-QA/Research types). The `cited_response` will be falsy (or equal to original response), and the `if cited_response:` guard at line 445 prevents the bus event from firing. The step should preserve this existing early-return convention.

5. **`LLMResponse.citations` is `None` by default.** Callers that do not pass `citations` config get `None`. Only callers that explicitly enable citations (when `config.get("citations")` is set) will receive populated citation lists. Down-stream code in `multi_agent_orchestrator` can safely check `if llm_response.citations:`.

---

## Application to pyfinagent

### Files to modify

| File | Change |
|------|--------|
| `backend/agents/llm_client.py` | 1. Add `citations: list[dict] \| None = None` field to `LLMResponse` dataclass (line 599). 2. In `generate_content`, when `config.get("citations")` is true AND a document block is present, ensure `"citations": {"enabled": True}` is set on the block. 3. In content-block parse loop (line 1463-1471), collect citation dicts from `block.citations` when present. 4. Pass `citations` kwarg in `LLMResponse(...)` at line 1523. |
| `backend/agents/multi_agent_orchestrator.py` | 1. Add `DeprecationWarning` + docstring update to `_add_citations` (line 1284). 2. Add early-return short-circuit at the top of `_add_citations` body (after warning) so the method returns `(response, {"input": 0, "output": 0})` immediately. 3. Optional: preserve call site at 438-451 unchanged (early return handles no-op). |
| `tests/verify_phase_25_E9.py` | New file: verification script satisfying three success criteria. |

---

## Verbatim code shapes

### Document block with citations enabled (plain text / Files API)

```python
# Plain text inline
{
    "type": "document",
    "source": {
        "type": "text",
        "media_type": "text/plain",
        "data": "The grass is green. The sky is blue.",
    },
    "title": "My Document",          # optional
    "context": "Trustworthy source.",  # optional, not citeable
    "citations": {"enabled": True},
}

# Files API reference (25.D9 pattern extended)
{
    "type": "document",
    "source": {"type": "file", "file_id": skill_file_id},
    "citations": {"enabled": True},
}
```

No beta header needed for citations. If using Files API simultaneously, `betas=["files-api-2025-04-14"]` is still required (already handled at `llm_client.py:1227-1230`).

### Citations response block shape (from official docs)

```python
# Each text block in response.content may have:
# block.type == "text"
# block.text == "According to the document, the grass is green"
# block.citations == [
#     {
#         "type": "char_location",
#         "cited_text": "The grass is green.",   # NOT counted as output tokens
#         "document_index": 0,                   # 0-indexed from all document blocks
#         "document_title": "My Document",
#         "start_char_index": 0,                 # 0-indexed, inclusive
#         "end_char_index": 20,                  # 0-indexed, exclusive
#     }
# ]
#
# For PDF documents:
# {
#     "type": "page_location",
#     "cited_text": "...",
#     "document_index": 0,
#     "document_title": "...",
#     "start_page_number": 1,   # 1-indexed, inclusive
#     "end_page_number": 2,     # exclusive
# }
#
# For custom content documents:
# {
#     "type": "content_block_location",
#     "cited_text": "...",
#     "document_index": 0,
#     "document_title": "...",
#     "start_block_index": 0,   # 0-indexed, inclusive
#     "end_block_index": 1,     # exclusive
# }
```

### Extraction in `generate_content` parse loop

```python
# In ClaudeClient.generate_content, inside the content-block loop
# (extending llm_client.py:1463-1471):

text = ""
thoughts = ""
citations_collected: list[dict] = []

for block in response.content:
    block_type = getattr(block, "type", "")
    if block_type == "thinking":
        thoughts = str(getattr(block, "thinking", ""))[:2000]
    elif block_type == "text":
        text += getattr(block, "text", "")
        # Collect citation objects from this text block if present
        block_citations = getattr(block, "citations", None) or []
        for c in block_citations:
            citations_collected.append({
                "type": getattr(c, "type", ""),
                "cited_text": getattr(c, "cited_text", ""),
                "document_index": getattr(c, "document_index", 0),
                "document_title": getattr(c, "document_title", ""),
                # char_location fields
                "start_char_index": getattr(c, "start_char_index", None),
                "end_char_index": getattr(c, "end_char_index", None),
                # page_location fields
                "start_page_number": getattr(c, "start_page_number", None),
                "end_page_number": getattr(c, "end_page_number", None),
                # content_block_location fields
                "start_block_index": getattr(c, "start_block_index", None),
                "end_block_index": getattr(c, "end_block_index", None),
            })

# At return (line 1523):
return LLMResponse(
    text=text,
    thoughts=thoughts,
    usage_metadata=umeta,
    citations=citations_collected if citations_collected else None,
)
```

### `_add_citations` deprecation pattern

```python
# multi_agent_orchestrator.py:1284 -- replace existing method docstring + body

async def _add_citations(self, response, classification):
    """
    DEPRECATED (phase-25.E9): CitationAgent post-processing replaced by
    native Anthropic Citations API (citations.enabled=True on document blocks).
    Native citations are server-side, zero extra LLM call, zero extra cost.

    This method is retained as a no-op stub so existing call sites at
    line 440 do not require removal until a cleanup pass.
    """
    import warnings
    warnings.warn(
        "_add_citations is deprecated (phase-25.E9): "
        "use native Anthropic Citations API instead of CitationAgent post-processing.",
        DeprecationWarning,
        stacklevel=2,
    )
    return response, {"input": 0, "output": 0}
```

### `LLMResponse` extension

```python
# llm_client.py:593-599 -- extend dataclass

@dataclass
class LLMResponse:
    """Provider-agnostic response container."""
    text: str
    thoughts: str = ""
    usage_metadata: UsageMeta = field(default_factory=UsageMeta)
    grounding_metadata: list[dict] = field(default_factory=list)
    citations: list[dict] | None = None   # phase-25.E9: native citations metadata
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched: Anthropic Citations platform docs, Anthropic SDK api.md, Claude cookbook citations notebook raw, Simon Willison blog, PEP 702)
- [x] 10+ unique URLs total (incl. snippet-only) -- 15 URLs collected
- [x] Recency scan (last 2 years) performed and reported above
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (`llm_client.py:593-599`, `llm_client.py:1307-1321`, `llm_client.py:1213-1230`, `llm_client.py:1463-1471`, `llm_client.py:1523`, `multi_agent_orchestrator.py:1284-1333`, `multi_agent_orchestrator.py:438-451`)

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client.py full relevant range, multi_agent_orchestrator.py full relevant range)
- [x] No contradictions; Anthropic docs and codebase are in consensus
- [x] All claims cited per-claim with URL or file:line anchor

---

## Search queries run (three-variant discipline)

1. Current-year frontier: "Anthropic Citations API document block citations enabled true 2026"
2. Last-2-year window: "Anthropic native citations citations_delta response block shape fields" (2025)
3. Year-less canonical: "anthropic python SDK citations response content blocks extraction"
4. Deprecation (year-less): "Python deprecated decorator warnings.warn DeprecationWarning class method"
5. Files API + Citations combo (2025): "Anthropic Citations API Files API file_id document block citations enabled 2025"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
