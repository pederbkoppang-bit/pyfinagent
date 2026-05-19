# Experiment Results — phase-29.4 (3 OWASP LLM Top-10 v2.0 heuristics)

**Step ID:** phase-29.4
**Date:** 2026-05-19
**Cycle:** 1

Agent-prompt-doc cycle. 3 heuristic rows + 3 negation bullets added to `qa.md` Dimension-1 security audit table. No code edits.

---

## 1. Edits made (verbatim)

### Edit 1 — qa.md `system-prompt-leakage` row (LLM07 enhanced)

**Before (qa.md:288):**
```
| system-prompt-leakage | New endpoint/log serializing full `messages` list (incl. system role) | WARN |
```

**After:**
```
| system-prompt-leakage | New endpoint/log/response serializing `agent_config.system_prompt`, full `messages` list incl. system role, or skill `.md` content to external caller. Grep: `json\.dumps.*messages\|logging.*system_prompt\|return.*"system"\s*:` (OWASP LLM07:2025) | WARN |
```

### Edit 2 — qa.md NEW `rag-memory-poisoning` row (LLM08)

```
| rag-memory-poisoning | New `add_memory()` / `add_memories()` call where input originates from an external or unvalidated source (not seed data or authenticated BQ path); or new vector-store import (`chromadb`, `pinecone`, `weaviate`, `pgvector`) without access-control doc. Grep: `add_memori(es\|y)\|import chromadb\|import pinecone\|import weaviate\|import pgvector` (OWASP LLM08:2025 — pyfinagent uses BM25 so Vec2Text inversion N/A; poisoning is the real surface) | WARN |
```

### Edit 3 — qa.md NEW `unbounded-llm-loop` row (LLM10)

```
| unbounded-llm-loop | New `while True` or unbounded `for` loop wrapping an LLM API call; OR removal/reduction of `MAX_TOOL_TURNS`, `MAX_RESEARCH_ITERATIONS`, `MAX_CONSECUTIVE_FAIL`, `MAX_RESEARCH_ITER` bound constants. Grep: `while True` near `messages.create\|generate_content`; diff reducing the named constants (OWASP LLM10:2025 denial-of-wallet) | WARN |
```

### Edit 4 — qa.md negation-list (3 new bullets appended)

```
- `system-prompt-leakage`: `system=agent_config.system_prompt` passed directly to `client.messages.create()` (e.g. `backend/agents/multi_agent_orchestrator.py:985`) is safe — only flag when the full `messages` list or raw `system_prompt` string is serialized to an external response, log line, or endpoint body
- `rag-memory-poisoning`: `FinancialSituationMemory` seed entries at `backend/agents/memory.py:23-54` are safe (static, not external); `load_from_bq_rows()` in authenticated BQ context is acceptable; BM25 corpus is not subject to Vec2Text embedding-inversion attacks
- `unbounded-llm-loop`: existing bounds (`for cycle in range(1, args.cycles + 1)` at `scripts/harness/run_harness.py:1111`; `for iteration in range(1, MAX_RESEARCH_ITERATIONS + 1)` at `backend/agents/multi_agent_orchestrator.py:523`; `for turn in range(max_turns)` at `:1048`) are correct — do NOT flag these; only flag NEW loops that bypass or remove the bound constants
```

### Edit 5 — qa.md "Source" line at the bottom of Dimension-1

**Before:**
```
Source: [OWASP LLM Top-10 2025](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025), [security.md](../rules/security.md).
```

**After:**
```
Source: [OWASP LLM Top-10 v2.0 (2025)](https://genai.owasp.org/llm-top-10/) (LLM07 System Prompt Leakage, LLM08 Vector and Embedding Weaknesses, LLM10 Unbounded Consumption added in v2.0; older v1.1 reference [Invicti](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025) preserved for LLM01-LLM06), [security.md](../rules/security.md).
```

---

## 2. Verbatim verification command output

```
$ grep -q 'rag-memory-poisoning' .claude/agents/qa.md && \
  grep -q 'unbounded-llm-loop' .claude/agents/qa.md && \
  grep -q 'agent_config.system_prompt' .claude/agents/qa.md && \
  grep -q 'OWASP LLM Top-10 v2.0\|OWASP v2.0\|OWASP LLM08:2025' .claude/agents/qa.md && \
  grep -q 'BM25 corpus is not subject to Vec2Text' .claude/agents/qa.md && \
  grep -q 'MAX_TOOL_TURNS\|MAX_RESEARCH_ITERATIONS\|MAX_CONSECUTIVE_FAIL' .claude/agents/qa.md
$ echo exit=$?
exit=0
$ wc -l .claude/agents/qa.md
437 .claude/agents/qa.md
```

All 6 grep predicates PASS. Exit 0. File 432 → 437 (+5 net, after table row enhancements and negation bullets — many lines are compressed into the table cells which are themselves long).

---

## 3. Why `rag-memory-poisoning` (not `rag-input-sanitization` from phase-29.0)

phase-29.0 §3a proposed the name `rag-input-sanitization`. After live grep of `backend/agents/memory.py`, researcher confirmed pyfinagent uses **BM25 (lexical), not vector embeddings** (no chromadb / pinecone / weaviate / pgvector imports anywhere in the codebase). So:

- Vec2Text embedding-inversion attack (50-92% recovery per Georgia Tech 2025) does NOT apply.
- The real LLM08 attack surface is **memory poisoning** via the `add_memory()` / `load_from_bq_rows()` paths if unauthenticated writes are ever added.

The renamed heuristic is more accurate. The negation bullet explicitly documents the BM25-vs-Vec2Text exemption so future Q/A spawns don't flag the lexical retrieval as if it were vector-store risk.

---

## 4. Files touched

| File | Change |
|---|---|
| `.claude/agents/qa.md` | row 288 rewritten + 2 new table rows + 3 negation bullets + source line rewritten (~+5 lines net) |
| `.claude/masterplan.json` 29.4 | audit_basis + verification fields rewritten |
| `handoff/current/research_brief.md` | rewritten (9 sources read in full) |
| `handoff/current/contract.md` | rewritten |
| `handoff/current/experiment_results.md` | this file |
| `handoff/current/live_check_29.4.md` | new |

**No** `backend/`, `frontend/`, `scripts/` files touched.

---

## 5. Honest disclosures

1. **Heuristic name change vs phase-29.0:** phase-29.0 §3a proposed `rag-input-sanitization`; this cycle renames to `rag-memory-poisoning` based on the BM25-vs-vectors finding. This is NOT criteria erosion — the new name maps to the same OWASP LLM08 risk class but accurately reflects pyfinagent's actual retrieval architecture.
2. **Vec2Text exemption explicit:** the negation bullet says "BM25 corpus is not subject to Vec2Text embedding-inversion attacks" — preempts false-positive flagging on the lexical retrieval pipeline.
3. **LLM10 timeout finding (out of scope):** researcher's brief flagged that `_call_agent()` at `multi_agent_orchestrator.py:982` has no timeout parameter on `client.messages.create()` — a slow LLM response can block indefinitely. NOT in scope for this step (would need a separate code-change cycle, not a heuristic-doc cycle). Filed mentally for a future ticket.
4. **OWASP v3.0:** none exists. v2.0 (2025) remains current per researcher's full-fetch of `genai.owasp.org/llm-top-10/` accessed 2026-05-19.
5. **Anti-rubber-stamp:** 6 ANDed grep predicates each anchored on a distinct phrase (heuristic name × 2, code-symbol, OWASP version string, BM25-vs-Vec2Text exemption phrase, named-bound-constants). Removing any one fails verification.
6. **Activation:** body-content edits to `qa.md` (not frontmatter) likely activate before full session restart on recent Claude Code versions, but the conservative "agent definition changes require session restart" rule still applies — live_check_29.4.md documents this.

---

## 6. Decision

Ready for Q/A spawn. 7 success criteria all evidenced on-disk.
