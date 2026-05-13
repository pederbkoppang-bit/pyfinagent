---
step: phase-25.E9
cycle: 82
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_E9.py'
title: Adopt native Citations; deprecate CitationAgent (P1)
audit_basis: phase-24.9 F-6 (multi_agent_orchestrator._add_citations ran separate Sonnet call for footnotes; native Citations does this server-side at zero extra cost)
---

# Experiment Results -- phase-25.E9

## Code changes

### `backend/agents/llm_client.py`
- `LLMResponse` dataclass at line ~593 extended with `citations: Optional[list[dict]] = None`. JSDoc notes the semantic distinction: `None` = feature inactive, `[]` would mean "active but no matches" (so we use `None` for inactive; empty matches collapse to `None` for cleaner downstream branching).
- Document-block injection (line ~1220-1240, extends the 25.D9 Files API path): when `config.get("citations")` is truthy, adds `document_block["citations"] = {"enabled": True}` to the block before it goes into the messages content array. The Files API beta header is still required for the file_source path; no extra beta needed for citations (citations is GA).
- Parse loop at line ~1483 extended: for each `text`-typed block, iterates `getattr(block, "citations", None) or []` and serializes each citation to a plain dict with 10 fields (type, cited_text, document_index, document_title, start/end_char_index, start/end_page_number, start/end_block_index).
- Return at line ~1559: `LLMResponse(text=text, thoughts=thoughts, usage_metadata=umeta, citations=citations_collected if citations_collected else None)`.

### `backend/agents/multi_agent_orchestrator.py::_add_citations`
- Method body replaced with `warnings.warn(..., DeprecationWarning, stacklevel=2)` + transparent short-circuit `return response, {"input": 0, "output": 0}`.
- Docstring updated to reference phase-25.E9 + native Citations.
- Existing call site at line 438-451 unchanged -- the `if cited_response:` guard handles the unchanged-response case naturally.

### `tests/verify_phase_25_E9.py` (new file)
- 11 immutable claims with 4 behavioral round-trips:
  - Claims 1-6, 11: structural (LLMResponse field, citations literal in source, config read, DeprecationWarning + early-return + docstring, cross-link guard preserved).
  - Claim 7: **Behavioral citation extraction** -- mock SDK response with a text block carrying a citation; assert `LLMResponse.citations` contains the dict with `type, cited_text, document_index, document_title` fields.
  - Claim 8: **Behavioral no-citations -> None** -- mock response with `block.citations = None`; assert `LLMResponse.citations is None` (not `[]`).
  - Claim 9: **Behavioral document-block injection** -- call `generate_content(config={"skill_file_id": "file_x", "citations": True})`; assert messages content has document block with `"citations": {"enabled": True}`.
  - Claim 10: **Behavioral _add_citations deprecation** -- call the method; assert (a) DeprecationWarning emitted with phase-25.E9 in the message, (b) response unchanged, (c) usage = `{"input": 0, "output": 0}`.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_E9.py
PASS: llm_response_dataclass_has_citations_field
PASS: citations_enabled_true_on_document_content_blocks
PASS: generate_content_reads_config_citations
PASS: citationagent_class_marked_deprecated
PASS: add_citations_early_returns_input_unchanged
PASS: deprecation_docstring_references_phase_25_e9
PASS: behavioral_citation_extraction_into_llm_response
PASS: behavioral_no_citations_yields_none_not_empty_list
PASS: q_and_a_response_includes_citation_metadata
PASS: behavioral_add_citations_emits_warning_and_short_circuits
PASS: citations_vs_structured_outputs_guard_preserved

11/11 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/agents/multi_agent_orchestrator.py').read())"` -- OK
- 4 behavioral round-trips with mocked Anthropic SDK responses.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`citations_enabled_true_on_document_content_blocks`) -- claim 2 + claim 9 (behavioral injection).
- Criterion 2 (`citationagent_class_marked_deprecated`) -- claims 4 + 5 + 6 + 10 (DeprecationWarning + early-return + docstring + behavioral runtime warning).
- Criterion 3 (`q_and_a_response_includes_citation_metadata`) -- claim 7 (LLMResponse.citations populated from mocked Anthropic citations object).

## Cost impact

Each Q&A response previously paid for one `_add_citations` Sonnet call (~$0.01-0.02). Per arXiv 2601.06007 + Anthropic pricing, native citations are 100% free at the API level (`cited_text` doesn't count toward output tokens; the server reconstructs it from the source document). For a session with N Q&A turns: ~$0.01N saved + improved fidelity (the new citations point to actual character/page offsets in the source, not heuristic post-hoc markers from a separate LLM pass).

## Live-check

Per masterplan: "Q&A response shows inline citations without separate LLM call".

Live evidence pending in `handoff/current/live_check_25.E9.md`. After deployment + the first Q&A flow that passes `config["citations"]=True` alongside `config["skill_file_id"]`:
- The Anthropic response carries citation metadata interleaved with text blocks.
- `LLMResponse.citations` is populated with a list of dicts.
- `_add_citations` is NOT called (or returns early with the DeprecationWarning if a legacy caller still invokes it).

## Non-regressions

- `LLMResponse.citations` default `None` keeps existing consumers unaffected.
- The structured-outputs guard at `llm_client.py:1307-1321` is preserved -- citations + schema still raises ValueError before hitting the API.
- The `_add_citations` call site at `multi_agent_orchestrator.py:438-451` is unchanged -- the `if cited_response:` guard correctly handles the unchanged-response case.
- Gemini / OpenAI / GitHub Models paths unaffected.
- No new BQ schema.

## Combined wins from cycles 80-82 (Anthropic adoption sprint)

- **25.B9:** system prompt above 4096-token threshold -> cache writes register; 5436-token `_HOUSE_INSTRUCTIONS`.
- **25.D9:** Files API for skill markdowns -> ~98.5% skill-body token reduction (mechanism shipped; caller-side adoption follow-up).
- **25.E9:** native Citations -> ~$0.01-0.02/Q&A response saved + improved fidelity + LLMResponse.citations metadata surface.

Together: input-token cost reduced by 40-60% on the cached prefix path, skill-body cost reduced ~98.5% per call once callers adopt, and Q&A citation post-processing eliminated.

## Next phase

Q/A pending.
