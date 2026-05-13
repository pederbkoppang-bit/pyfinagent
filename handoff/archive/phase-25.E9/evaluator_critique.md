---
step: phase-25.E9
cycle: 82
cycle_date: 2026-05-13
agent: qa
verdict: PASS
ok: true
violated_criteria: []
certified_fallback: false
---

# Q/A Critique -- phase-25.E9 (Adopt native Citations; deprecate CitationAgent)

## 5-item harness-compliance audit

1. **Researcher spawn (25.E9)** -- PASS. `handoff/current/research_brief.md`
   header is "phase-25.E9: Adopt Native Citations; Deprecate CitationAgent",
   tier=moderate, gate envelope: `external_sources_read_in_full=5,
   urls_collected=15, recency_scan_performed=true,
   internal_files_inspected=2, gate_passed=true`. Three-variant search
   queries visible (current-year frontier 2026, last-2-year 2025,
   year-less canonical). Five sources are top-tier (Anthropic platform
   docs, Anthropic SDK api.md, Anthropic cookbook notebook, Simon
   Willison blog, PEP 702). Floor satisfied.
2. **Contract pre-commit** -- PASS. `handoff/current/contract.md` is for
   step phase-25.E9 and reproduces the three immutable success criteria
   verbatim from `.claude/masterplan.json`
   (`citations_enabled_true_on_document_content_blocks`,
   `citationagent_class_marked_deprecated`,
   `q_and_a_response_includes_citation_metadata`). Research-gate
   summary present.
3. **Results captured** -- PASS. `handoff/current/experiment_results.md`
   contains the verbatim verifier output ("11/11 claims PASS, 0 FAIL")
   plus per-file change summary and live-check pointer.
4. **Log-last** -- PASS. `grep "25\.E9" handoff/harness_log.md` returns
   only forward-references from prior cycles 79/80/81 as "next cycle
   candidate". No `## Cycle 82 ... phase=25.E9` block yet, so the log
   append is correctly deferred to AFTER this Q/A PASS and BEFORE the
   masterplan status flip.
5. **No verdict-shopping** -- PASS. First Q/A spawn for this step
   (prior `evaluator_critique.md` was for 25.D9; this file is the
   canonical first verdict for 25.E9).

All 5 protocol gates clear.

## Deterministic checks

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
EXIT=0
```

AST parse of `backend/agents/llm_client.py` and
`backend/agents/multi_agent_orchestrator.py` -- both OK.

Spot-grep confirms:
- `llm_client.py:606` declares `citations: Optional[list[dict]] = None`.
- `llm_client.py:1228, 1233-1234` reads `bool(config.get("citations"))`
  and sets `document_block["citations"] = {"enabled": True}` on the
  block.
- `llm_client.py:1331-1337` preserves the citations + schema guard.
- `llm_client.py:1488, 1495-1496, 1566` collects per-block citations
  into `citations_collected` and threads `citations=...` into the
  `LLMResponse` return (None when empty).
- `multi_agent_orchestrator.py:1284-1310` -- `_add_citations`
  docstring mentions "DEPRECATED (phase-25.E9)" and the body emits
  `warnings.warn(..., DeprecationWarning, stacklevel=2)` before
  returning `(response, {"input": 0, "output": 0})`.
- `multi_agent_orchestrator.py:438-440` call site preserved; the
  early-return makes it a transparent no-op so the existing
  `if cited_response:` guard handles the unchanged-response case.

## Per-criterion judgment

**Criterion 1 -- `citations_enabled_true_on_document_content_blocks`:**
PASS. Verifier claim 2 (literal `"citations": {"enabled": True}` in
source) + claim 9 (behavioral injection: calling
`generate_content(config={"skill_file_id": "file_x", "citations": True})`
results in the document block carrying `"citations": {"enabled": True}`
in `messages.create` kwargs).

**Criterion 2 -- `citationagent_class_marked_deprecated`:** PASS.
Verifier claim 4 (`DeprecationWarning` + `warnings.warn` present) +
claim 5 (early-return `(response, {"input": 0, "output": 0})`) +
claim 6 (docstring references "deprecated" + "phase-25.E9") +
claim 10 (behavioral runtime: calling `_add_citations` emits a
`DeprecationWarning` with `phase-25.E9` in the message AND returns
the input response unchanged AND usage = `{"input": 0, "output": 0}`).

**Criterion 3 -- `q_and_a_response_includes_citation_metadata`:** PASS.
Verifier claim 7 (behavioral: mocked Anthropic SDK response with a
text block carrying a citation object -> `LLMResponse.citations`
contains a serialized dict with `type, cited_text, document_index,
document_title` fields) + claim 8 (no-citations path yields `None`
not `[]`, preserving the "feature inactive" vs "active but no
matches" semantic distinction).

## Anti-rubber-stamp mutation matrix (all six caught)

| Mutation | Caught by | Result |
|----------|-----------|--------|
| Drop `"citations": {"enabled": True}` literal | claim 2 grep + claim 9 behavioral | FAIL both |
| Stop reading `config.get("citations")` | claim 3 grep + claim 9 behavioral | FAIL both |
| Remove `DeprecationWarning` | claim 4 grep + claim 10 behavioral | FAIL both |
| Return shape `None` vs `[]` flipped | claim 8 behavioral | FAIL |
| Re-enable Sonnet call inside `_add_citations` | claim 5 + claim 10 (response would differ from input) | FAIL both |
| Drop structured-outputs guard at :1331 | claim 11 grep | FAIL |

All six spirit-breaking mutations are covered by independent claims;
no single grep is load-bearing alone -- each criterion has at least
one structural + one behavioral check.

## Scope honesty

The contract correctly flags that `_add_citations` body is retained as
a deprecated no-op stub rather than deleted -- removal is a follow-up
cleanup. Frontend rendering of citations is explicitly out of scope.
Gemini / OpenAI paths untouched. The structured-outputs incompatibility
guard at `llm_client.py:1331-1337` is preserved unchanged. Research
brief confirmed citations is GA on direct Anthropic API (no beta
header required for citations alone; `betas=["files-api-2025-04-14"]`
still required for Files API document blocks, unchanged from 25.D9).
Live-check evidence file `live_check_25.E9.md` already present in
`handoff/current/`.

## Research-gate compliance

Contract `References` cites `handoff/current/research_brief.md`
explicitly + file:line anchors. Brief is moderate-tier with the full
Anthropic Citations platform docs, SDK api.md, cookbook notebook,
Simon Willison blog, and PEP 702 read in full. Implementation matches
the verbatim code shapes in the brief.

## Verdict

PASS.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met. Verifier 11/11 claims PASS exit=0. 4 behavioral round-trips with mocked Anthropic SDK responses cover the 6-mutation matrix end-to-end. Deprecation is via DeprecationWarning + early-return short-circuit (no body deletion, transparent to call site). Structured-outputs guard preserved unchanged at llm_client.py:1331-1337. Research-gate satisfied (5 sources read in full, recency scan performed, three-variant queries visible). 5-item harness-compliance audit clean.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item_audit",
    "verification_command",
    "ast_parse_both_files",
    "structural_grep_citations_dataclass",
    "structural_grep_document_block_injection",
    "structural_grep_deprecation_warning",
    "behavioral_mutation_matrix",
    "scope_honesty",
    "research_gate_compliance"
  ]
}
```
