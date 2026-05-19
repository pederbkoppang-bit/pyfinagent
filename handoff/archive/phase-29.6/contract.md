# Contract — phase-29.6 (Extract qa.md code-review heuristics to skill)

**Step ID:** phase-29.6
**Date:** 2026-05-19
**Author:** Main (overnight execution)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Sources read in full | 6 |
| Snippet-only | 10 |
| URLs collected | 16 |
| Recency scan + frontier-sync | DONE (v2.1.140-144 covered) |
| `gate_passed` | true |

**Brief:** `handoff/current/research_brief.md` (this cycle).

**Headline findings:**
1. **`skills:` frontmatter syntax**: YAML block-list (`skills:\n  - code-review-trading-domain`); inline JSON-array form `["..."]` is NOT documented.
2. **Auto-preload combo**: `user-invocable: false` + `disable-model-invocation` ABSENT (default false). Setting `disable-model-invocation: true` BLOCKS preload — verbatim from official docs: "You cannot preload skills that set `disable-model-invocation: true`."
3. **Body injection at spawn**: full SKILL.md body injects at subagent SPAWN, equivalent to current inline content. Real benefit is qa.md legibility (437 → ~212 lines), not per-spawn token savings.
4. **Naming convention**: lowercase + hyphens, ≤64 chars. `code-review-trading-domain` (28 chars) ✓ — more accurate than phase-29.0 §5 draft `code-review-heuristics` because the content IS trading-domain-specific.
5. **Body size**: 230 lines, well under best-practices 500-line ceiling.
6. **Self-contained extraction**: lines 207-437 have no back-references to other qa.md sections; only external pointer is the phase-16.59 archive citation (preserved in SKILL.md Sources block).
7. **Known bug #29441 does NOT apply**: the preload-fails-silently issue affects team-spawned process-level subagents only. Q/A spawned via Agent tool from Main's session uses the in-process path which works.

---

## Audit-basis (from phase-29.0)

phase-29.0 §5 Skills extraction proposed extracting `qa.md:207-429` to `.claude/skills/code-review-heuristics/SKILL.md`. Three refinements this cycle:
- Renamed to `code-review-trading-domain` (more accurate; reflects the heuristic content).
- Target range adjusted to `qa.md:207-437` (phase-29.4 added 3 OWASP rows + 3 negation bullets after phase-29.0 was written).
- `skills:` syntax confirmed as YAML block-list (researcher verified via official sub-agents doc).

---

## Verbatim immutable success criteria

1. `skill_md_created_at_canonical_path` — `.claude/skills/code-review-trading-domain/SKILL.md` exists.
2. `skill_md_frontmatter_correct` — has `name: code-review-trading-domain` + `description:` + `user-invocable: false`. Does NOT set `disable-model-invocation: true` (would block preload).
3. `qa_md_frontmatter_lists_skill` — `skills:\n  - code-review-trading-domain` block present after `permissionMode: plan`.
4. `qa_md_body_replaced_with_cross_reference` — old `## Code review heuristics (phase-16.59)` section (~230 lines) replaced with 3-5 line cross-reference pointing to the skill.
5. `qa_md_shrunk_substantially` — file line count dropped from 437 to ≤220.
6. `skill_md_contains_all_5_dimensions` — Dimensions 1-5 + Top-15 + Reporting block + Sources block all preserved verbatim.
7. `phase_29_4_OWASP_heuristics_preserved_in_skill` — `rag-memory-poisoning`, `unbounded-llm-loop`, `BM25 corpus is not subject to Vec2Text` all present in the new SKILL.md (the 29.4 work isn't accidentally lost in the move).

**Verification command:**
```bash
test -f .claude/skills/code-review-trading-domain/SKILL.md && \
grep -q '^name: code-review-trading-domain' .claude/skills/code-review-trading-domain/SKILL.md && \
grep -q '^user-invocable: false' .claude/skills/code-review-trading-domain/SKILL.md && \
! grep -q '^disable-model-invocation: true' .claude/skills/code-review-trading-domain/SKILL.md && \
grep -A1 '^skills:' .claude/agents/qa.md | grep -q '  - code-review-trading-domain' && \
grep -q 'code-review-trading-domain' .claude/agents/qa.md && \
[ $(wc -l < .claude/agents/qa.md) -le 220 ] && \
grep -q 'Dimension 5' .claude/skills/code-review-trading-domain/SKILL.md && \
grep -q 'rag-memory-poisoning' .claude/skills/code-review-trading-domain/SKILL.md && \
grep -q 'unbounded-llm-loop' .claude/skills/code-review-trading-domain/SKILL.md && \
grep -q 'BM25 corpus is not subject to Vec2Text' .claude/skills/code-review-trading-domain/SKILL.md
```

**`verification.live_check`:** `"live_check_29.6.md captures (a) pre/post qa.md line count, (b) byte-level diff hash of dimensions 1-5 content (preserved verbatim across move), (c) post-restart operator recipe: spawn Q/A and ask 'list your code-review heuristics' to confirm the preloaded skill provides the full Top-15 + 5 dimensions."`

---

## Plan

1. DONE — Researcher (complex, 6 sources read in full).
2. DONE — Contract.
3. NEXT — GENERATE:
   - EDIT 1: `mkdir -p .claude/skills/code-review-trading-domain` and `Write` the SKILL.md (frontmatter + extracted body verbatim — researcher provided the full content).
   - EDIT 2: `.claude/agents/qa.md` frontmatter — insert `skills:` block-list after `permissionMode: plan`.
   - EDIT 3: `.claude/agents/qa.md` body — replace `qa.md:207-437` with a 5-line cross-reference.
   - EDIT 4: Update masterplan 29.6 entry.
   - EDIT 5: Write experiment_results.md + live_check_29.6.md.
4. Spawn Q/A. Circuit breaker: 2 fresh-qa.
5. Log → flip → commit.

---

## Out of scope

- Layer-2 skill extraction (`backend/agents/skills/*.md` is a separate system per CLAUDE.md).
- Researcher.md extraction (phase-29.0 §5 says it's NOT a strong candidate).
- Migrating any other qa.md content to skills.
