# Sprint Contract -- phase-25.E9 -- Adopt native Citations; deprecate CitationAgent

**Cycle:** phase-25 cycle 26 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.E9
**Priority:** P1
**Audit basis:** bucket 24.9 F-6 -- `multi_agent_orchestrator.py:1284-1333` runs separate Sonnet call for footnotes; native Citations does this server-side at zero extra cost

## Research-gate

Researcher spawned this cycle (agent a9179000eebba5ea2). Brief at
`handoff/current/research_brief.md`. Gate envelope: 5 sources read in full,
15 URLs, recency scan performed, 2 internal files inspected, gate_passed=true.

Key research conclusions:
- **Document block shape:** add `"citations": {"enabled": True}` to any document block. Extends the 25.D9 Files API block at `llm_client.py:1213-1230`.
- **GA, no beta header.** Citations are GA on all active Claude 4.x models; no extra beta header required (just `betas=["files-api-2025-04-14"]` if the document came from Files API).
- **`cited_text` is free** (server-reconstructs from source; not billed). This is the primary cost saving vs `_add_citations` (~$0.01-0.02 per Q&A response eliminated).
- **All-or-none rule:** all document blocks in a request must have `"citations": {"enabled": True}` or none. Mixed = 400.
- **Structured-outputs incompatibility already guarded** at `llm_client.py:1307-1321` -- no change there.
- **Response shape:** N text blocks each with optional `.citations` list. Each citation: `{type, cited_text, document_index, document_title, start_char_index, end_char_index, start_page_number, end_page_number, start_block_index, end_block_index}`.
- **LLMResponse extension:** add `citations: list[dict] | None = None` field.
- **Deprecation pattern:** `warnings.warn("...", DeprecationWarning, stacklevel=2)` + early-return `(response, {"input": 0, "output": 0})`. Existing call site at `multi_agent_orchestrator.py:438-451` has `if cited_response:` guard that handles transparent short-circuit.

## Hypothesis

Adding (a) `citations: list[dict] | None = None` field to `LLMResponse`,
(b) `"citations": {"enabled": True}` on the document block when
`config.get("citations")` is true (extending the 25.D9 Files API
injection at `llm_client.py:1213-1230`), (c) a parse loop that collects
`block.citations` alongside `block.text`, (d) `DeprecationWarning` +
early-return in `_add_citations` -- eliminates the post-processing Sonnet
call and surfaces native server-side citation metadata at zero extra
cost.

## Success criteria (verbatim from masterplan)

1. `citations_enabled_true_on_document_content_blocks`
2. `citationagent_class_marked_deprecated`
3. `q_and_a_response_includes_citation_metadata`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_E9.py`

Live check (per masterplan):
`Q&A response shows inline citations without separate LLM call`

## Plan

1. **`backend/agents/llm_client.py`** -- 4 edits:
   - Extend `LLMResponse` dataclass at line ~593 with `citations: list[dict] | None = None`.
   - At the document-block injection site (line ~1213, added by 25.D9), when `config.get("citations")` is True (or by default when a skill_file_id is present for Q&A flows), add `"citations": {"enabled": True}` to the block.
   - In the parse loop that iterates `response.content` (~line 1463), collect `block.citations` per text block into a `citations_collected: list[dict]` using `getattr(block, "citations", None) or []`. Map each citation object's attributes to a serializable dict.
   - In the `LLMResponse(...)` return (~line 1523), pass `citations=citations_collected if citations_collected else None`.
2. **`backend/agents/multi_agent_orchestrator.py::_add_citations`** -- mark deprecated:
   - Add `warnings.warn(..., DeprecationWarning, stacklevel=2)` at the top of the method body.
   - Replace the body with `return response, {"input": 0, "output": 0}` (transparent short-circuit).
   - Update the docstring to reference phase-25.E9 + native Citations.
   - Existing call site at line 438-451 doesn't need changes -- the `if cited_response:` guard handles the unchanged-response case naturally.
3. **Verifier** -- `tests/verify_phase_25_E9.py` -- 10+ claims:
   - Claim 1: `LLMResponse` declares `citations: list[dict] | None = None`.
   - Claim 2: `llm_client.py` contains `"citations": {"enabled": True}` literal (in the document block path).
   - Claim 3: `llm_client.py::ClaudeClient.generate_content` reads `config.get("citations")` and feeds it into the document block.
   - Claim 4: `multi_agent_orchestrator._add_citations` contains `DeprecationWarning` and a `warnings.warn(...)` call.
   - Claim 5: `_add_citations` early-returns the input response unchanged (`return response, {"input": 0, "output": 0}`).
   - Claim 6: `_add_citations` docstring mentions "deprecated" + "phase-25.E9".
   - Claim 7: **Behavioral citation extraction** -- mock anthropic SDK response with a text block carrying a citation; assert `LLMResponse.citations` contains the citation dict with `type, cited_text, document_index, document_title` fields.
   - Claim 8: **Behavioral no-citations path** -- mock response with text-only blocks; assert `LLMResponse.citations is None` (not an empty list -- to keep the field semantically meaningful for downstream consumers).
   - Claim 9: **Behavioral document-block injection** -- call `generate_content(prompt, config={"skill_file_id": "file_x", "citations": True})`; assert the document block in `messages.create` kwargs has `"citations": {"enabled": True}`.
   - Claim 10: **Behavioral _add_citations early-return** -- call the deprecated method; assert it (a) raises a `DeprecationWarning`, (b) returns the input response unchanged, and (c) the usage dict is `{"input": 0, "output": 0}`.
   - Claim 11: cross-link guard preserved (`citations` + `schema` still raises ValueError at line ~1314).

## Non-goals

- No removal of `_add_citations` method body (just deprecation + short-circuit). Removal is a follow-up cleanup.
- No frontend changes to render citation metadata (Q&A surface render is a separate phase).
- No Gemini-side change (Anthropic-only feature).
- No new BQ schema (citations are response-time metadata, not persisted).
- No structured-outputs path change -- the existing guard at `llm_client.py:1307-1321` correctly rejects citations + schema combos.

## References

- `handoff/current/research_brief.md` -- full brief
- `backend/agents/llm_client.py:593-599` (`LLMResponse`), :1213-1230 (25.D9 document block), :1307-1321 (citations + schema guard), :1463-1471 (parse loop), :1523 (return shape)
- `backend/agents/multi_agent_orchestrator.py:438-451` (call site), :1284-1333 (`_add_citations` to deprecate)
- Anthropic Citations API docs (cited in brief)
- PEP 702 (deprecation marker pattern)
