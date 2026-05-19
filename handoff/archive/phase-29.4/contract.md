# Contract — phase-29.4 (3 OWASP LLM Top-10 v2.0 heuristics in qa.md)

**Step ID:** phase-29.4
**Date:** 2026-05-19
**Author:** Main (overnight execution)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Sources read in full | 9 |
| Snippet-only | 11 |
| URLs collected | 20 |
| Recency scan + frontier-sync | DONE |
| `gate_passed` | true |

**Brief:** `handoff/current/research_brief.md`.

**Headline findings:**
1. LLM07 (System Prompt Leakage), LLM08 (Vector/Embedding Weaknesses), LLM10 (Unbounded Consumption) — OWASP LLM Top-10 v2.0 (2025) entries; v2.0 is still current as of May 2026 (no v3.0).
2. **pyfinagent uses BM25 (lexical), NOT vector embeddings** — Vec2Text inversion (50-92% recovery confirmed by Georgia Tech) does NOT directly apply. The real LLM08 risk is **memory poisoning** via `add_memory()` / `load_from_bq_rows()` paths in `backend/agents/memory.py`. Hence the new heuristic name `rag-memory-poisoning` (not `rag-input-sanitization` from phase-29.0 draft) — more accurate.
3. **All existing harness loops are properly bounded** (run_harness.py:1111 `cycles`; multi_agent_orchestrator.py:523 `MAX_RESEARCH_ITERATIONS=3`; :1048 `MAX_TOOL_TURNS=5`; :57-58 `MAX_CONSECUTIVE_FAIL=3` / `MAX_RESEARCH_ITER=3`). LLM10 heuristic flags REMOVAL or bypass of these constants, not their existence.
4. The existing qa.md:288 `system-prompt-leakage` row has a narrow cue ("New endpoint/log serializing full `messages` list"). The phase-29.4 edit ENHANCES that cue with explicit grep patterns.

---

## Audit-basis (from phase-29.0)

phase-29.0 §3a: qa.md:271-296 covers OWASP v1.1 (2023). v2.0 (2025) added 3 entries — LLM07, LLM08, LLM10 — missing from qa.md. This cycle adds them as detection-cue + severity rows in the Dimension-1 table + appends negation bullets.

---

## Verbatim immutable success criteria

1. `qa_md_dimension1_table_has_llm07_enhanced_cue` — `system-prompt-leakage` row now mentions `agent_config.system_prompt` or has the explicit grep pattern.
2. `qa_md_dimension1_table_has_rag_memory_poisoning_row` — new row name `rag-memory-poisoning` present with severity WARN.
3. `qa_md_dimension1_table_has_unbounded_llm_loop_row` — new row name `unbounded-llm-loop` present with severity WARN.
4. `qa_md_negation_list_has_3_new_bullets` — bullets covering system-prompt-leakage (FP exclusion), rag-memory-poisoning (FP exclusion), unbounded-llm-loop (FP exclusion).
5. `qa_md_owasp_v2_2025_cited` — text in section references OWASP v2.0 (2025) explicitly.
6. `qa_md_line_count_grew_within_bounds` — file grew by 5-15 lines (3 table rows + 3 negation bullets ≈ 6-12 lines).
7. `qa_md_syntax_unchanged` — `python3 -c "import re; assert re.search(r'^---$', open('.claude/agents/qa.md').read(), re.M)"` (frontmatter intact).

**Verification command:**
```bash
grep -q 'rag-memory-poisoning' .claude/agents/qa.md && \
grep -q 'unbounded-llm-loop' .claude/agents/qa.md && \
grep -q 'agent_config.system_prompt' .claude/agents/qa.md && \
grep -q 'OWASP LLM Top-10 v2.0\|OWASP v2.0\|OWASP LLM08:2025' .claude/agents/qa.md && \
grep -q 'BM25 corpus is not subject to Vec2Text' .claude/agents/qa.md && \
grep -q 'MAX_TOOL_TURNS\|MAX_RESEARCH_ITERATIONS\|MAX_CONSECUTIVE_FAIL' .claude/agents/qa.md
```

**`verification.live_check`:** `"live_check_29.4.md captures (a) verbatim diff of the 3 new/enhanced rows + 3 negation bullets, (b) the cited file:line anchors in memory.py / multi_agent_orchestrator.py / run_harness.py confirming the heuristics target real risk surfaces."`

---

## Plan

1. DONE — Researcher.
2. DONE — Contract.
3. NEXT — GENERATE:
   - EDIT 1: Replace existing qa.md:288 `system-prompt-leakage` row with enhanced version.
   - EDIT 2: Insert new `rag-memory-poisoning` row after the enhanced LLM07 row.
   - EDIT 3: Insert new `unbounded-llm-loop` row after row 2.
   - EDIT 4: Append 3 negation-list bullets to the "What NOT to flag" block.
   - EDIT 5: Add a one-line citation to OWASP v2.0 (2025) near the Dimension-1 source line.
   - EDIT 6: Update masterplan.json 29.4 entry.
   - EDIT 7: Write experiment_results.md + live_check_29.4.md.
4. Spawn `qa`. Circuit breaker: 2 fresh-qa.
5. Log → flip → commit.

---

## Out of scope

- Implementing the LLM10 timeout suggestion on `_call_agent()` (researcher noted; not in scope).
- Adding access-control wrapper around `add_memory()` (separate phase).
- Vector-store migration discussion (BM25 stays per existing design).
