---
step: phase-10.7.2
title: Recursive Prompt Optimization (Research Directive rewriter)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-10.7
---

# Sprint Contract -- phase-10.7.2

## Research-gate summary

`handoff/current/phase-10.7.2-research-brief.md`. tier=moderate, 6 in-full, 16 URLs, recency scan, gate_passed=true.

## Key research findings

1. **Promptbreeder (arXiv 2309.16797)** — self-referential loop where mutation-prompts evolve alongside task-prompts. The Research Directive IS the mutation-prompt for our researcher subagent.
2. **SIPDO (arXiv 2505.19514, 2025)** — local+global confirmation prevents regression; accuracy score non-decreasing by construction. Adopt this pattern.
3. **Anthropic harness design** — evaluator prompt updated by HUMAN when judgments diverge; HITL is the documented Anthropic pattern. The rewriter PROPOSES; Peder APPROVES; Main writes `.claude/agents/researcher.md`; session restart required (per CLAUDE.md).
4. **Anti-drift guards (multiple sources):** `MIN_BRIEFS=5` floor; global-confirmation step; simplicity criterion (favor smaller deltas); LLM-judge score ≥ 0.6.
5. **LLM fallback:** when Anthropic 401s on `sk-ant-oat-*`, route via Gemini (per phase-16.31's MAS pattern). Fail-open returns `None` (no proposal) rather than crashing.

## Hypothesis

A new `backend/meta_evolution/directive_rewriter.py` (~280 LOC) with:
- `DirectiveVersion` dataclass (proposed text, score, components)
- `rewrite_directive(current, recent_briefs, outcome_signals) -> DirectiveVersion | None`
- LLM client lazy-init (Anthropic primary, Gemini fallback)
- `persist_version(bq_client, version)` for the new `directive_versions` table
- Anti-drift guards (MIN_BRIEFS=5, score-floor, simplicity)

Plus migration script + test file with 7 cases. Total ~430-470 LOC across 4 files.

## Success Criteria (verbatim, immutable)

```
python -m pytest tests/meta_evolution/test_directive_rewriter.py -v
```

## Plan steps

1. Create `backend/meta_evolution/directive_rewriter.py` (~280 LOC)
2. Create `scripts/migrations/create_directive_versions_table.py` (~100 LOC, mirror alpha_velocity migration)
3. Create `tests/meta_evolution/test_directive_rewriter.py` (~150 LOC, 7 test cases using FakeBQ + FakeLLM stubs)
4. Update `backend/meta_evolution/__init__.py` if needed (add to scope notes)
5. Run pytest verification
6. Spawn Q/A

## What Q/A must audit

1. All 4 files created with proper docstrings + type hints
2. Pytest 7/7 PASS
3. Anti-drift guards present (MIN_BRIEFS, simplicity, score floor)
4. HITL gate: rewriter PROPOSES, doesn't auto-modify `.claude/agents/researcher.md` (CLAUDE.md says agent-md edits require session restart + separation-of-duties)
5. Gemini fallback wired (consistent with 16.31 MAS pattern)
6. No regression on broader pytest suite (182 baseline)
7. BQ migration is `--apply` / `--verify` / `--dry-run` triple, not auto-applied
