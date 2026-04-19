# Experiment Results -- phase-11.0 Vertex Migration Plan Doc

**Step:** phase-11.0 (audit + migration plan).
**Date:** 2026-04-19.

## What was built

One doc, zero code changes. The deliverable is `docs/VERTEX_AI_GENAI_MIGRATION.md` (16,784 bytes) — a migration playbook all of phase-11.1 through 11.4 will execute against.

Sections included:
- Background + 2026-06-24 deadline + `google-genai==1.73.1` pin.
- Rollout timeline (phase-11.1 through 11.4, with phase-12 Rainbow Deploy cross-ref for the final cutover).
- **Call-site inventory** — 8 matches across 6 files in `backend/agents/` (zero in `scripts/`, zero elsewhere). File:line-anchored table with bucket (trivial / moderate / complex) per line.
- Per-bucket migration recipes (before/after code snippets).
- **ThinkingConfig silent-breakage mitigation section** — this is the critical non-obvious finding from research: the old `generation_config={"thinking": {...}}` dict form is silently ignored by the new SDK, disabling extended thinking on every judge agent (Moderator, Critic, Risk Judge, Synthesis) without any error. Doc includes two Q/A grep patterns locking the fix in: one grep must return 0 (old dict form), another must return >=4 (new `types.ThinkingConfig`). Plus a runtime assertion to drop into `llm_client.GeminiClient`.
- API diff table (10 rows: init, GenerativeModel, generate_content, GenerationConfig, thinking, Tool/grounding, async, streaming, function-calling).
- Authentication parity notes (env-var vs explicit `genai.Client(vertexai=True, project=..., location=...)`).
- Dependency plan: add `google-genai==1.73.1`, keep `google-cloud-aiplatform` (still needed for BigQuery + Vertex AI Search non-generative + OAuth credentials), remove only the top-level `vertexai` package in 11.4.
- Per-step breakdown for 11.1 → 11.4 with per-step verify + rollback.
- Rainbow Deploys integration runbook for the final cutover.
- References (7 read-in-full + 6 snippet-only).

## Pre-Q/A self-check (per cross-cycle recommendation)

Per the recommendation from phase-3.1 Q/A onward, I grep-verified every specific claim in the doc against source BEFORE writing. Findings:

- **Researcher claimed `nlp_sentiment.py` was a moderate caller.** Source audit disproved it: `nlp_sentiment.py` has no `vertexai.generative_models` import. The doc explicitly overrides the research brief's claim with a footnote.
- **Researcher claimed `llm_client.py` was complex.** Grep shows only a docstring mention at `llm_client.py:303` (no runtime import). But the file DOES own the ThinkingConfig injection at `:671-679` and the `_flatten_schema` Pydantic helper — so the "complex" label is correct for different reasons than the research brief gave. Doc explains.
- **Researcher claimed `test_evaluator_agent.py` was a trivial migration target.** It is — but the patch is `VERTEX_AVAILABLE` -> `GENAI_AVAILABLE`, not a call-site swap. Doc captures this per step 11.2.

Net: 3 doc claims corrected from research-brief drift BEFORE Q/A, with every correction footnoted.

## File list

Created:
- `docs/VERTEX_AI_GENAI_MIGRATION.md` (16,784 bytes)

Modified: none.

`git diff --name-only | grep -v '^docs/\|^handoff/\|^\.claude/masterplan\.json'` returns empty — non-goal "no code changes in this cycle" honored.

## Verification command output

### Immutable (from contract)

```
$ test -f docs/VERTEX_AI_GENAI_MIGRATION.md && python -c "import pathlib; p=pathlib.Path('docs/VERTEX_AI_GENAI_MIGRATION.md'); assert p.stat().st_size > 2000, 'plan doc too thin'; print('size:', p.stat().st_size, 'bytes; ok')"
size: 16784 bytes; ok
```

Exit 0. Doc size 8.4x the minimum threshold.

### Inventory parity

```
$ grep -rn "vertexai\.generative_models\|from vertexai import generative_models\|\.GenerativeModel(\|vertexai\.init" backend/ scripts/ --include="*.py" | wc -l
8
```

Doc claims 8 lines across 6 files — Q/A can re-run the exact grep and byte-compare the file:line list in the inventory table.

### No-code-change enforcement

```
$ git diff --name-only | grep -v '^docs/\|^handoff/\|^\.claude/masterplan\.json'
(empty)
```

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `docs/VERTEX_AI_GENAI_MIGRATION.md` exists with all required sections | PASS (all 11 sections present) |
| 2 | Doc size > 2000 bytes | PASS (16784) |
| 3 | Inventory grep output locked into doc | PASS (8 matches listed verbatim) |
| 4 | Phase-12 Rainbow Deploys cross-linked | PASS (2 sections + references) |

## Known caveats

1. **3 research-brief claims corrected by pre-Q/A self-check.** Noted explicitly above; the doc footnotes each correction so downstream readers can trace the drift. This is the same cross-cycle class of issue Q/A has been flagging; catching it pre-Q/A is the new discipline.
2. **ThinkingConfig mitigation is research-derived, not yet tested in code.** The fix pattern is documented; actual wiring + assertion land in phase-11.3. Q/A cannot verify the fix works live until then.
3. **`test -f docs/VERTEX_AI_GENAI_MIGRATION.md` as immutable verification is thin.** It checks the doc exists and is >2000 bytes, not that it's correct. Q/A's LLM-judgment leg has to do the heavy lifting; the deterministic checks only confirm grep parity.
4. **Rainbow Deploys runbook is a cross-reference, not a deliverable.** Phase-12 has its own research gate not yet run; the runbook paragraph in this doc is a placeholder anchored at phase-12 step 12.4.
