# Live-check placeholder -- phase-25.E9

**Step:** 25.E9 -- Adopt native Citations; deprecate CitationAgent
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Q&A response shows inline citations without separate LLM call"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_E9.py`).
- 4 behavioral round-trips: citation extraction (10 fields per cite), no-citations -> None, document-block injection with `citations.enabled=true`, `_add_citations` DeprecationWarning + transparent short-circuit.
- Backend AST clean for both touched files.

## Post-deployment operator workflow
1. Restart backend so the new citations path + LLMResponse.citations field are loaded:
   ```
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Trigger a Q&A flow that supplies a document context (e.g., ask a question about a paper-trade rationale or a backtest result). The caller must:
   - Pass `config["skill_file_id"]` (set by 25.D9 orchestrator bridge), OR
   - Pass `config["citations"]=True` alongside any document block.
3. Verify the response contains citation metadata:
   ```python
   from backend.agents.llm_client import ClaudeClient
   client = ClaudeClient(...)
   response = client.generate_content(prompt="...", generation_config={
       "skill_file_id": "file_qa_skill",
       "citations": True,
   })
   assert isinstance(response.citations, list), "no citations returned"
   for c in response.citations:
       print(c["type"], c["document_title"], c["cited_text"])
   ```
4. Verify `_add_citations` is NOT being called in the hot path. If a legacy caller invokes it:
   - A DeprecationWarning surfaces.
   - The response is returned unchanged.
   - No second Sonnet call is made (zero extra cost).

## Cost expectation
Per session with N Q&A turns: ~$0.01-0.02 × N saved (the eliminated `_add_citations` Sonnet calls). `cited_text` doesn't count toward output tokens, so the new path has zero marginal cost at the LLM level.

## Closes audit basis
phase-24.9 F-6 RESOLVED. Native Citations API replaces the redundant post-processing Sonnet call; metadata fidelity improved (real character/page offsets vs heuristic markers); deprecation warning prevents new consumers from picking up the old path.

## Sprint summary (cycles 80-82, Anthropic adoption mini-sprint)
- 25.B9 (cycle 80): system prompt cache write registers (5436-token `_HOUSE_INSTRUCTIONS`).
- 25.D9 (cycle 81): Files API mechanism shipped (98.5% skill-body token reduction; caller adoption follow-up).
- 25.E9 (cycle 82): native Citations replaces `_add_citations` post-processing.

Combined: significantly lower input-token cost on cached prefix, near-zero skill-body cost once callers adopt, and Q&A citation processing free.

**Audit anchor for next bucket:** 25.C9 (Batch API for non-interactive pipeline steps; 50% savings) OR 25.S (daily P&L attribution per ticker) OR 25.D9.1 (caller-side adoption follow-up for Files API).
