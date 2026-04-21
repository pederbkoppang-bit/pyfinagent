# Experiment Results -- phase-4.16.3

## What was built
Five documentation edits wiring the new research-gate + handoff
discipline into the project's canonical reference + how-to surfaces.

1. **`ARCHITECTURE.md`** -- appended a MADR-structured section
   "Research Gate Discipline (phase-4.16)" with Context / Decision /
   Consequences / Handoff folder convention / Confirmation /
   Cross-references subheadings. Cross-links to researcher.md,
   rules/research-gate.md, CLAUDE.md, docs/runbooks/per-step-protocol.md.

2. **`.claude/rules/research-gate.md` (NEW)** -- the how-to guide.
   Documents the ≥5 sources floor, last-2-year recency scan,
   source-quality hierarchy, URL collection rule, JSON envelope
   shape, and the full `handoff/` tree convention
   (`current/` / `archive/` / `audit/` / `logs/`). Points readers
   back to the researcher agent prompt + ARCHITECTURE + CLAUDE.md.

3. **`.claude/context/research-gate.md`** -- patched the stale
   "read 3-5 in full" to ">=5 in full (phase-4.16.1 floor)" +
   added a checklist item for the mandatory last-2-yr recency scan.

4. **`.claude/context/mas-architecture.md`** -- updated the stale
   4-agent Claude Code MAS topology (Lead + Researcher + QA
   Evaluator + Harness Verifier) to the current 3-agent topology
   (Main + Researcher + Q/A), with a historical note that
   phase-4.15.0 merged the splits and re-splitting is forbidden.

5. **Memory** -- `feedback_research_gate_min_three_sources.md`
   kept as historical audit pointer; the new file
   `feedback_research_gate_min_three_sources.md` already notes
   phase-4.16.1 supersession in the researcher.md edits. No retire.

## Files changed
- `ARCHITECTURE.md` (+49 lines: MADR-style Research Gate section)
- NEW `.claude/rules/research-gate.md` (101 lines)
- `.claude/context/research-gate.md` (2 edits; recency + floor)
- `.claude/context/mas-architecture.md` (1 edit; 3-agent topology)

## Verbatim verification
```
$ grep -q "Research Gate" ARCHITECTURE.md && grep -q "5 sources" .claude/rules/*.md && echo PASS
IMMUTABLE VERIFICATION PASS
```

Cross-link audit:
- ARCHITECTURE.md -> 4 cross-links to researcher.md / research-gate.md.
- rules/research-gate.md -> 3 cross-links back to ARCHITECTURE + CLAUDE + runbook.
- Cross-link graph is bidirectional, consistent.

## Success criteria coverage
| Criterion | Status |
|-----------|--------|
| architecture_md_documents_research_gate_floor | MET -- MADR section with >=5 / last-2yr clauses |
| claude_rules_references_handoff_layout | MET -- research-gate.md "Handoff folder convention" table |
| cross_links_between_rules_consistent | MET -- bidirectional cross-links confirmed |

## Scope honesty
- CLAUDE.md intentionally NOT edited (kept authoritative for cycle
  protocol; cross-links to the new files instead of duplicating rules).
- Other `.claude/rules/*.md` files intentionally NOT edited -- adding
  the 5-sources clause to ALL of them would be rule duplication;
  one dedicated file is the right shape per Diátaxis.

## References
- Contract (pre-commit): `handoff/current/phase-4.16.3-contract.md`
- Research: `handoff/current/phase-4.16.3-research-brief.md`
  (7 sources read in full; Anthropic harness-design + built-multi-agent-research-system + building-effective-agents; C4 model + MADR + Diátaxis)
