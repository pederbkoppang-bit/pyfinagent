# Sprint Contract -- phase-4.16.3
Step: Update ARCHITECTURE.md + .claude/rules with new discipline

## Research Gate
researcher_4163 (tier=moderate) gate_passed=true. 7 read in full, 14 URLs collected. Brief: `handoff/current/phase-4.16.3-research-brief.md`.

Key findings:
- Anthropic harness-design + built-multi-agent-research-system + building-effective-agents (all read in full) converge on: file-based handoffs, source-quality enforcement in agent prompts (not just operator docs), stress-test every harness assumption.
- C4 / MADR / Diátaxis: separate reference docs (ARCHITECTURE.md) from how-to guides (rules/*.md). Single canonical source per audience; cross-link instead of duplicating.
- Internal stale files identified: `.claude/context/research-gate.md:10` says "3-5 sources" (obsolete), `.claude/context/mas-architecture.md:13-14` lists old 4-agent topology.
- No existing `.claude/rules/research-gate.md`; creating it as how-to guide is the right shape.

## Hypothesis
Applying the 5 edits from the brief (new ARCHITECTURE section as MADR-style ADR; new `.claude/rules/research-gate.md`; 2 stale-context patches; memory-file check) satisfies 4.16.3 without duplicating rule text across files.

## Success Criteria (immutable)
```
grep -q "Research Gate" ARCHITECTURE.md && grep -q "5 sources" .claude/rules/*.md
```
Plus 3 sub-criteria:
- architecture_md_documents_research_gate_floor
- claude_rules_references_handoff_layout
- cross_links_between_rules_consistent

## Plan (PRE-commit; will NOT diverge)
1. Append a new section "Research Gate Discipline (phase-4.16)" to
   ARCHITECTURE.md in MADR shape (Context / Decision / Consequences /
   Confirmation) with the >=5 source floor, last-2yr scan, file-layout
   rule. Cross-link to researcher.md + new rules file.
2. Write `.claude/rules/research-gate.md` as the how-to guide that
   includes "5 sources" + handoff-layout rules. This is what the
   immutable grep anchor targets.
3. Patch `.claude/context/research-gate.md` line 10: "3-5 sources" -> ">=5 sources (phase-4.16.1)".
4. Patch `.claude/context/mas-architecture.md` lines 13-14: 4-agent
   topology -> 3-agent (Main + Researcher + Q/A) per phase-4.15.0 merge.
5. Verify immutable grep-pair + 3 sub-criteria.

## Scope honesty
- CLAUDE.md stays authoritative but does NOT duplicate the new rule
  text -- it cross-links.
- `memory/feedback_research_gate_min_three_sources.md` is superseded
  by phase-4.16.1 but kept as a historical audit record (annotated
  pointer to phase-4.16.1 in the body, not deletion).

## References
- Research brief: `handoff/current/phase-4.16.3-research-brief.md`
- Anthropic: harness-design-long-running-apps, built-multi-agent-research-system, building-effective-agents
- C4 model, MADR, Diátaxis
