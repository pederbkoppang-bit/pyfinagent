# Sprint Contract -- phase-11.0 Audit + Migration Plan Doc

**Written:** 2026-04-19 PRE-commit.
**Step id:** `11.0` in phase-11 (Vertex AI generative-models SDK migration; phase-11 itself has 5 steps — this cycle closes only 11.0, which is the planning deliverable all subsequent steps depend on).
**Immutable verification:** `test -f docs/VERTEX_AI_GENAI_MIGRATION.md && python -c "import pathlib; p=pathlib.Path('docs/VERTEX_AI_GENAI_MIGRATION.md'); assert p.stat().st_size > 2000, 'plan doc too thin'; print('ok')"`.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 7, snippet_only_sources: 6, urls_collected: 13, recency_scan_performed: true, internal_files_inspected: 10, gate_passed: true}`. **First spawn under the new three-query-variant discipline** (`.claude/rules/research-gate.md` "Search-query composition") and compliance confirmed: year-less canonical + 2026 + 2025 variants all run, with the year-less search correctly surfacing the official Google migration guide that year-locked queries would have missed.

Staked recommendations adopted into contract:
- **Per-file migration categorization**: trivial (evaluator_agent, skill_optimizer, test_evaluator_agent), moderate (debate, risk_debate, nlp_sentiment), complex (orchestrator, llm_client). The two complex files own all grounding/RAG, google_search protobuf hack, ThinkingConfig injection, and `_flatten_schema` Pydantic flattener.
- **Shim pattern** `backend/agents/_genai_client.py` factory — hard cutover preferred over dual-SDK branching.
- **Pin**: `google-genai==1.73.1` (2026-04-14 stable).
- **Keep `google-cloud-aiplatform`** — still needed for BigQuery, Vertex AI Search non-generative APIs, `google.oauth2.service_account.Credentials`.

**Critical non-obvious finding**: ThinkingConfig is a silent breaking change. Old `generation_config={"thinking": {...}}` dict does nothing in the new SDK — no error, but extended thinking silently disabled on Moderator / Critic / Risk Judge / Synthesis agents.

## Hypothesis

Writing a thorough migration plan doc (`docs/VERTEX_AI_GENAI_MIGRATION.md`) that enumerates every call site with its bucket (trivial/moderate/complex), the per-method API diff, the ThinkingConfig silent-breakage mitigation, and the Rainbow Deploys integration (phase-12) makes the remaining steps 11.1-11.4 mechanical to execute and auditable in isolation.

## Success criteria

**Functional:**
1. `docs/VERTEX_AI_GENAI_MIGRATION.md` exists with sections:
   - Background & deadline (2026-06-24).
   - Call-site inventory: file:line-anchored table categorizing every `vertexai.generative_models` / `from vertexai import generative_models` / `GenerativeModel` / `vertexai.init` usage in `backend/` and `scripts/`.
   - Per-bucket migration recipe (trivial / moderate / complex).
   - API diff table: `GenerativeModel(model_name)` → `genai.Client` + `.models.generate_content(...)`; streaming; async; tool-use; function-calling; structured output / response_schema; system instructions; ThinkingConfig.
   - **ThinkingConfig silent-breakage mitigation** — explicit section with assertion lines to drop into code + a Q/A grep pattern.
   - Authentication parity notes (GOOGLE_APPLICATION_CREDENTIALS + project/location flow).
   - Dependency plan: add `google-genai==1.73.1` to `backend/requirements.txt` in step 11.1; DO NOT remove `google-cloud-aiplatform` (still needed).
   - Step-by-step breakdown for 11.1 through 11.4 — what each step delivers, how it is verified, what its rollback is.
   - Rainbow Deploys integration (phase-12 cross-ref): step 11.4 final cutover uses Rainbow pattern per phase-12 step 12.4.
   - Runbook: how to roll back mid-migration.
   - References section listing the 7 sources read in full + 6 snippet-only.
2. Doc size > 2000 bytes (enforced by the immutable verification command).
3. Internal grep-produced call-site inventory must match actual source: I will run `grep -rn "vertexai.generative_models\|from vertexai import generative_models\|\.GenerativeModel(" backend/ scripts/` myself during PLAN and lock the output into the doc, so Q/A can re-run the exact grep and byte-compare.
4. Doc cross-links phase-12 (Rainbow Deploys) for the final cutover.

**Correctness verification commands:**
- Immutable: `test -f docs/VERTEX_AI_GENAI_MIGRATION.md && python -c "import pathlib; p=pathlib.Path('docs/VERTEX_AI_GENAI_MIGRATION.md'); assert p.stat().st_size > 2000; print('ok')"` — exit 0.
- Inventory parity: `grep -rn "vertexai\.generative_models\|from vertexai import generative_models" backend/ scripts/ | wc -l` — the number must appear verbatim in the doc's inventory section.
- No code changes in this cycle: `git diff --name-only | grep -v '^docs/\|^handoff/\|^.claude/masterplan.json'` — empty.

**Non-goals (phase-11.0 scope boundary):**
- NOT pinning `google-genai` (phase-11.1).
- NOT editing any Python file (phase-11.1-11.4).
- NOT writing the shim module (phase-11.1).
- NOT migrating any call site (phase-11.2-11.3).
- NOT removing `vertexai` (phase-11.4).
- NOT writing tests for phase-11 code (each sub-step will add its own tests).
- NOT implementing Rainbow Deploys (phase-12).

## Plan steps

1. `grep -rn "vertexai\.generative_models\|from vertexai import generative_models\|\.GenerativeModel(\|vertexai\.init" backend/ scripts/` — capture verbatim output for the doc's inventory table.
2. Read each flagged file at the flagged line to categorize bucket (trivial / moderate / complex) — validate the researcher's categorization against actual source (pre-Q/A self-check).
3. Write `docs/VERTEX_AI_GENAI_MIGRATION.md` per the section list above.
4. Run the immutable verification command + inventory parity check.

## References

- `handoff/current/phase-11-research-brief.md`
- `backend/agents/evaluator_agent.py` (trivial per research)
- `backend/agents/orchestrator.py` + `backend/agents/llm_client.py` (complex per research)
- External read-in-full: Google's official migration guide, googleapis/python-genai API ref, Medium migration post, leoy.blog post, google-genai PyPI, Vertex AI Search grounding doc, Thinking doc.

## Researcher agent id

`a3b44ee0e852aad73`
