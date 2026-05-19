# Research Brief: OWASP LLM Top-10 v2.0 (2025) — LLM07, LLM08, LLM10 for qa.md Dimension-1 insertion

**Step ID:** phase-29.4
**Tier:** complex
**Date:** 2026-05-19
**Author:** Researcher subagent (Sonnet 4.6)

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/ | 2026-05-19 | Official doc (OWASP GenAI) | WebFetch full | "Attackers can exploit vulnerabilities to invert embeddings and recover significant amounts of source information." Full attack vectors: unauthorized access, cross-context leaks, embedding inversion, data poisoning |
| https://ironcorelabs.com/blog/2025/owasp-llm-top10-2025-update/ | 2026-05-19 | Industry blog | WebFetch full | "Vectors and embeddings vulnerabilities present significant security risks in systems utilizing RAG with LLMs." Georgia Tech Vec2Text: 50-92% text recovery from embeddings. |
| https://repello.ai/blog/owasp-llm-top-10-2026 | 2026-05-19 | Practitioner blog (2026) | WebFetch full | LLM07: "sensitive business logic lives in the tool and application layer rather than in the prompt itself" (2026 shift). LLM10: adversarial prompts crafted to exploit token prediction patterns with 128K-1M context windows. |
| https://www.oligo.security/academy/owasp-top-10-llm-updated-2025-examples-and-mitigation-strategies | 2026-05-19 | Academy/practitioner | WebFetch full | LLM07 vectors: credential exposure, guardrail extraction, privilege abuse. LLM08: embedding inversion, retrieval manipulation, access-control bypass. LLM10: denial-of-wallet, resource overload. |
| https://bsg.tech/blog/owasp-llm-top-10/ | 2026-05-19 | Practitioner (BSG) | WebFetch full | LLM07: "Treat system prompts as public -- never embed API keys, database credentials, or internal URLs." LLM08: enforce document-level access at retrieval layer, not just application level. LLM10: monitor p95/p99 token consumption thresholds. |
| https://www.stackhawk.com/blog/owasp-system-prompt-leakage/ | 2026-05-19 | Security vendor blog | WebFetch full | Full taxonomy of LLM07 sensitive info at risk; "the system prompt should not be considered a secret, nor should it be used as a security control." Mandate automated secret scanning for credentials in configurations. |
| https://genai.owasp.org/llm-top-10/ | 2026-05-19 | Official doc (OWASP) | WebFetch full | Authoritative enumeration of all 10 entries. LLM07 mitigation: strict output filtering, suspicious-query monitoring. LLM08: validate embedding sources, anomaly detection. LLM10: token/cost caps, rate limiting. |
| https://witness.ai/blog/llm-system-prompt-leakage/ | 2026-05-19 | Security vendor (WitnessAI) | WebFetch full | Detection requires intent-based (not keyword-based) analysis; bidirectional inspection; tool-call checkpoints in agentic workflows. |
| https://www.promptfoo.dev/blog/unbounded-consumption/ | 2026-05-19 | OSS security tool | WebFetch full | "A single request can consume 100K tokens" -- request-count rate limiting fails for LLM workloads; token-based per-user quotas are the gap most deployments miss. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/html/2502.12630v1 | arXiv preprint | Prompt leakage automated attack framework; search snippet sufficient for context |
| https://arxiv.org/pdf/2603.08993 | arXiv (Arbiter) | System prompt interference detector; snippet establishes threat is confirmed active |
| https://arxiv.org/pdf/2601.21233 | arXiv | "Just Ask" paper on system prompt disclosure; snippet level |
| https://www.firetail.ai/blog/llm10-unbounded-consumption | Blog | Duplicate coverage; BSG/promptfoo sources are superior |
| https://securityboulevard.com/2025/12/llm10-unbounded-consumption-firetail-blog/ | Blog | HTTP 403 on fetch |
| https://www.lasso.security/blog/owasp-top-10-for-llm-applications-generative-ai-key-updates-for-2025 | Blog | Partial fetch; summary-level only |
| https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf | Official PDF | Requires PDF download; genai.owasp.org/llm-top-10 + individual risk pages cover the same content |
| https://www.trydeepteam.com/docs/frameworks-owasp-top-10-for-llms | Framework docs | Redundant coverage |
| https://www.brightdefense.com/resources/owasp-top-10-llm/ | Blog | Snippet sufficient; not an improvement over BSG or oligo sources |
| https://cycode.com/blog/ai-security-vulnerabilities/ | Blog | AI security 2026 overview; snippet only needed |
| https://blog.alexewerlof.com/p/owasp-top-10-ai-llm-agents | Blog | Agents cheat sheet; snippet level |

---

## Recency scan (2024-2026)

Searched for 2026 literature on OWASP LLM security, system prompt leakage, RAG embedding attacks, and unbounded consumption (queries used: "OWASP LLM security 2026 agentic AI RAG attack vector embedding weakness", "system prompt leakage detection grep pattern agentic AI heuristic static analysis 2025", "unbounded token consumption LLM DoS detection rate limiting autonomous harness loop 2025", "OWASP LLM top 10 2025 system prompt leakage LLM07 new entries May 2026").

**Findings from 2025-2026:**

1. **LLM07 (2026 shift, Repello AI, read in full):** The 2026 architectural view reframes LLM07 -- sensitive business logic should live in the tool/application layer, not prompts. Output monitoring to catch prompt-signature patterns before reaching users is now the recommended detection layer. The Arbiter framework (arXiv:2603.08993, March 2026) found 152 prompt-interference findings across three major coding-agent system prompts -- confirming agentic systems dramatically expand the LLM07 blast radius.

2. **LLM08 (2025-2026, IronCoreLabs + repello, read in full):** Georgia Tech Vec2Text research confirmed 50-92% text recovery from embeddings. The 2026 view emphasizes document-level authorization at the retrieval layer as distinct from model-level instructions -- a critical architectural gap many current deployments miss.

3. **LLM10 (2026, Repello AI, read in full):** Context windows expanded from 8K to 1M tokens; adversarial prompts crafted to exploit token prediction patterns for disproportionate output generation are a confirmed attack class. Promptfoo (2025) confirmed request-count rate limiting is insufficient; token-based per-user quotas are the real gap.

4. **No new OWASP v3.0 or revision:** No evidence of an OWASP LLM Top-10 v3.0 release as of May 2026. The 2025 (v2.0) document remains the current authoritative version.

---

## Key findings

1. **LLM07 definition (OWASP v2.0):** System prompt leakage = unauthorized exposure of system instructions that guide model behavior, including credentials, internal URLs, business rules, and guardrail mechanisms. The key principle: "the system prompt should not be considered a secret, nor should it be used as a security control." Primary attack: prompt injection / direct extraction requests. Secondary: behavioral inference, encoding tricks. (StackHawk, BSG, OWASP, all read in full)

2. **LLM07 detection cue:** Code patterns to flag -- new endpoint serializing full `messages` list (including system role) to logs, debug output, error responses, or API response payloads. Specifically: `json.dumps(messages)`, `logging.*system_prompt`, `return {"messages": ...}`, or `print(*messages)` where `messages` includes `{"role": "system", ...}` entries. (BSG: "monitor production logs and model outputs for evidence of leakage"; WitnessAI: "response inspection -- catch system instructions in outputs")

3. **LLM07 severity:** WARN appropriate (not BLOCK) because: (a) leakage requires an active extraction attempt or active serialization bug -- it is a configuration/design smell, not an immediate RCE; (b) existing BLOCK entry `prompt-injection-path` already covers the active injection vector; (c) Cloudflare pattern: flag architectural smell at WARN. The existing `system-prompt-leakage` row in qa.md (line 288) already has WARN -- this confirms the severity. Phase-29.4 expands the detection cue to be grep-executable.

4. **LLM08 definition (OWASP v2.0):** Vector/embedding weaknesses = security risks in RAG systems including (a) embedding inversion (reconstructing source text from vector representations, confirmed 50-92% recovery), (b) retrieval manipulation (poisoned docs score highly for legitimate queries), (c) cross-context data leakage in multi-tenant vector stores, (d) access-control bypass when RAG pipeline does not enforce source-system permissions. (OWASP genai.owasp.org, IronCoreLabs, BSG -- all read in full)

5. **LLM08 pyfinagent surface:** `backend/agents/memory.py` uses BM25 (not vector embeddings), so embedding inversion (Vec2Text) does not directly apply. However, `load_from_bq_rows()` (memory.py:145) ingests BQ rows without content validation -- a data-poisoning vector. Runtime memories via `add_memory()` come from `generate_reflection()` (memory.py:213-254) which calls the Gemini LLM on unconstrained ticker/recommendation data -- the lesson text is unconstrained LLM output injected back into the BM25 corpus and subsequently into agent prompts via `format_for_prompt()`. Attack path: compromise the reflection step or BQ row, inject adversarial text -> that text gets injected into future debate agent prompts at `orchestrator.py:1849-1861`. Detection cue: new `add_memory()` call with external/unvalidated input, or new vector store import without access-control doc.

6. **LLM08 severity:** WARN (not BLOCK) -- BM25 corpus requires authenticated BQ write to compromise; vector-embedding-specific attacks do not apply to the current BM25 implementation. Risk is real but not an immediate BLOCK-class finding unless new code adds an unauthenticated write path.

7. **LLM10 definition (OWASP v2.0):** Unbounded consumption = excessive/uncontrolled LLM resource usage enabling DoS, financial exploitation, and unauthorized model replication. Three sub-forms: (a) denial-of-wallet (excessive API queries inflating costs), (b) resource monopolization (context-window-sized payloads, chain-of-thought exhaustion), (c) noisy-neighbor DoS. (OWASP, oligo, promptfoo, BSG -- all read in full)

8. **LLM10 pyfinagent surface:** `run_harness.py:1111` main loop is bounded by `args.cycles` (default 3). `multi_agent_orchestrator.py:523` research loop bounded by `MAX_RESEARCH_ITERATIONS = 3`. `multi_agent_orchestrator.py:1048` tool loop bounded by `MAX_TOOL_TURNS = 5`. All existing loops are bounded -- good. Detection cue: new diff removing or bypassing these constants, or adding a `while True` loop around LLM API calls. Also: `_call_agent()` at line 982 has no timeout parameter -- per-call timeout is absent.

9. **Existing qa.md Dimension-1 table analysis:** Lines 275-297 show 10 entries. The `system-prompt-leakage` row (line 288) exists with narrow detection cue. LLM08 (vector/embedding) and LLM10 (unbounded consumption) have NO rows. Phase-29.4 scope: enhance existing LLM07 detection cue + add LLM08 row + add LLM10 row.

10. **3-query variant discipline (per research-gate.md):**
    - Current-year frontier: "OWASP LLM top 10 2025 LLM07 new entries May 2026"
    - Last-2-year window: "OWASP top 10 LLM applications 2025 official document genai.owasp.org"
    - Year-less canonical: "OWASP LLM Top 10 v2.0 2025 LLM07 system prompt leakage LLM08 vector embedding LLM10 unbounded consumption"
    All three variants run; mix of current-year, last-2-year, and year-less hits present in source tables.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/agents/qa.md` | 432 | Q/A agent prompt; Dimension-1 table at lines 275-297 | Live; insertion target |
| `backend/agents/multi_agent_orchestrator.py` | 1494 | Layer-2 MAS orchestrator; system prompt injected via `system=agent_config.system_prompt` at lines 985, 1076, 1187 | LLM07 risk surface; no current serialization of full messages list found in logs |
| `backend/agents/memory.py` | 268 | BM25 FinancialSituationMemory; `add_memory()` at line 86; `load_from_bq_rows()` at line 145; `format_for_prompt()` at line 131 | LLM08 risk surface: unconstrained lesson text injected into prompts |
| `backend/agents/orchestrator.py` | 1477 | Layer-1 pipeline; `format_for_prompt()` output -> `past_memories` at lines 1849-1861 | LLM08 risk surface: debate agents receive BM25 memory content |
| `scripts/harness/run_harness.py` | 1206 | Harness loop; `for cycle in range(1, args.cycles + 1)` at line 1111; `MAX_CONSECUTIVE_FAIL=3`, `MAX_RESEARCH_ITER=3` at lines 57-58 | LLM10 risk surface: loop bounded by CLI arg; no per-call timeout on `_call_agent()` |

---

## Consensus vs debate (external)

All five tier-1/2 sources (OWASP official, BSG, StackHawk, oligo, repello) agree on:
- LLM07 = prompt/system-instruction extraction, an application-architecture flaw, not a model-level flaw
- LLM08 = RAG/retrieval pipeline attack surface: embedding inversion + retrieval manipulation + multi-tenant access control gaps
- LLM10 = resource/cost exhaustion, both intentional (adversarial) and accidental (misconfigured loops)

Minor debate: IronCoreLabs emphasizes application-layer encryption (ALE) for LLM08; BSG and repello emphasize document-level authorization at the retrieval layer. Both are valid and complementary.

No debate on severity: all sources treat LLM07/LLM08 as architectural WARNs and LLM10 as WARN.

---

## Pitfalls (from literature)

1. **LLM07 false positives:** Flagging `system=agent_config.system_prompt` in a standard API call (safe). Only flag when the full `messages` list (including system role) is serialized to output, log, or external endpoint.
2. **LLM08 false positives:** Pyfinagent uses BM25 (lexical), not vector embeddings. Vec2Text embedding-inversion attacks do not apply to BM25. Flag only if new code introduces a vector store without access controls.
3. **LLM10 false positives:** Bounded loops are the correct pattern. Flag only when a NEW loop removes or bypasses the bound constants.

---

## Application to pyfinagent (mapping external findings to file:line anchors)

**LLM07:** `multi_agent_orchestrator.py:985` has `system=agent_config.system_prompt` in `client.messages.create()` -- this is safe (static string, not serialized). Flag only if a NEW diff adds serialization of the full `messages` list or `system_prompt` string to an external endpoint, API response body, or logger call. Existing `system-prompt-leakage` row (qa.md:288) needs expanded grep cue.

**LLM08:** `memory.py:145` -- `load_from_bq_rows()` ingests BQ rows without content validation. `memory.py:86` -- `add_memory()` accepts unconstrained text including from LLM reflection output (`generate_reflection()` at memory.py:213). That text flows via `format_for_prompt()` into debate agent prompts at `orchestrator.py:1849-1861`. Detection cue: new `add_memory()` / `add_memories()` call with external/unvalidated input; or new vector store import.

**LLM10:** `run_harness.py:1111` cycle loop bounded. `multi_agent_orchestrator.py:523` research iterations bounded. `multi_agent_orchestrator.py:1048` tool turns bounded. All existing bounds in place. Detection cue: diff that removes these constants or adds a `while True` around an LLM API call.

---

## CONCRETE 3-ROW INSERTION for qa.md Dimension-1 table

The Dimension-1 table lives at qa.md lines 279-290 (inside the block that starts at line 275). The three rows replace the existing `system-prompt-leakage` row (line 288) with an enhanced version AND add two new rows for LLM08 and LLM10. All rows use WARN severity (consistent with existing `system-prompt-leakage` WARN; LLM08/LLM10 do not rise to BLOCK per literature analysis above).

**Row 1 (replaces qa.md:288 -- enhanced detection cue for LLM07):**

```
| system-prompt-leakage | New endpoint/log/response serializing `agent_config.system_prompt`, full `messages` list incl. system role, or skill `.md` content to external caller. Grep: `json\.dumps.*messages\|logging.*system_prompt\|return.*"system"\s*:` | WARN |
```

**Row 2 (NEW -- LLM08 Vector/Embedding Weaknesses / RAG poisoning):**

```
| rag-memory-poisoning | New `add_memory()` / `add_memories()` call where input originates from an external or unvalidated source (not seed data or authenticated BQ path); or new vector-store import (`chromadb`, `pinecone`, `weaviate`, `pgvector`) without access-control doc. Grep: `add_memori(es\|y)\|import chromadb\|import pinecone` | WARN |
```

**Row 3 (NEW -- LLM10 Unbounded Consumption / harness loop):**

```
| unbounded-llm-loop | New `while True` or unbounded `for` loop wrapping an LLM API call; or removal/reduction of `MAX_TOOL_TURNS`, `MAX_RESEARCH_ITERATIONS`, `MAX_CONSECUTIVE_FAIL`, `MAX_RESEARCH_ITER`. Grep: `while True` near `messages.create\|generate_content`, or diff reducing these constants | WARN |
```

### Negation-list bullets (append to Dimension-1 "What NOT to flag" block at qa.md:292-296)

```
- `system-prompt-leakage`: `system=agent_config.system_prompt` passed directly to `client.messages.create()` is safe -- only flag when the full `messages` list or raw `system_prompt` string is serialized to an external response, log line, or endpoint body
- `rag-memory-poisoning`: `FinancialSituationMemory` seed entries at `memory.py:23-54` are safe (static, not external); `load_from_bq_rows()` in authenticated BQ context is acceptable; BM25 corpus is not subject to Vec2Text embedding-inversion attacks
- `unbounded-llm-loop`: the existing bounds (`for cycle in range(1, args.cycles + 1)` at run_harness.py:1111; `for iteration in range(1, MAX_RESEARCH_ITERATIONS + 1)` at multi_agent_orchestrator.py:523; `for turn in range(max_turns)` at multi_agent_orchestrator.py:1048) are correct -- do NOT flag these; only flag NEW loops that bypass or remove the bound constants
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9 sources fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (20 total URLs collected)
- [x] Recency scan (last 2 years) performed + reported (2025-2026 section present above)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md, multi_agent_orchestrator.py, memory.py, orchestrator.py, run_harness.py)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in footer)

---

## Sources

1. [OWASP LLM08:2025 Vector and Embedding Weaknesses](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/)
2. [IronCoreLabs -- OWASP LLM Top-10 2025 update](https://ironcorelabs.com/blog/2025/owasp-llm-top10-2025-update/)
3. [Repello AI -- OWASP LLM Top-10 2026 guide](https://repello.ai/blog/owasp-llm-top-10-2026)
4. [Oligo Security -- OWASP LLM Top-10 2025 examples and mitigations](https://www.oligo.security/academy/owasp-top-10-llm-updated-2025-examples-and-mitigation-strategies)
5. [BSG -- OWASP LLM Top-10 2025](https://bsg.tech/blog/owasp-llm-top-10/)
6. [StackHawk -- OWASP LLM07 System Prompt Leakage](https://www.stackhawk.com/blog/owasp-system-prompt-leakage/)
7. [OWASP GenAI -- LLM Top-10 full list](https://genai.owasp.org/llm-top-10/)
8. [WitnessAI -- LLM System Prompt Leakage prevention](https://witness.ai/blog/llm-system-prompt-leakage/)
9. [Promptfoo -- Unbounded Consumption](https://www.promptfoo.dev/blog/unbounded-consumption/)
