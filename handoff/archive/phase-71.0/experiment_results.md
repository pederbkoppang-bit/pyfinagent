# Experiment results — step 71.0 (Harness + MAS upgrade design pack)

**Phase/step:** phase-71 → 71.0 (phase opener) | **Date:** 2026-07-17 | **Type:** design + research only (offline,
$0, NO production code).

## What was produced

1. **`handoff/current/research_brief_71.0.md`** — research-gate output. Envelope: `gate_passed=true`,
   `external_sources_read_in_full=7` (floor 5), 13 snippet-only, 20 URLs, recency scan performed, 10 internal
   files re-anchored. It re-validated every grounding WITH URLs, confirmed every register fact still holds on HEAD
   7d54d30d, and refined the design (dropped the oversold worst-of-N #8a, reframed 71.5 as config hygiene, flagged
   3 rider-traps). Launched via Workflow structured-output (Opus 4.8, $0).
2. **`handoff/current/contract.md`** — step 71.0 contract; verbatim immutable criteria; research summary; plan;
   boundaries. Written BEFORE the design (mtime-proven).
3. **`handoff/current/design_harness_mas_71.md`** — the design pack (GENERATE deliverable): per-step design for
   71.1–71.6, EACH grounded in a specific Claude Code feature / Anthropic doc WITH a URL; the binding constraints
   (exactly-3-agents at L3, no self-eval, $0 L3 rail, Layer-2 cost-sensitivity, local-only, file-based handoffs,
   separation-of-duties + roster-snapshot for the agent-file-editing steps); and all 15 REJECTED proposals
   enumerated with disqualifiers (+ the rider-trap note so a rejected idea can't ride in on a kept one).

## Verification command output (verbatim)

```
$ bash -c 'test -f handoff/current/harness_proposals.json && test -f handoff/current/design_harness_mas_71.md && grep -Eqi "structured.?output|workflow" handoff/current/design_harness_mas_71.md && grep -Eqi "verbatim|transcrib" handoff/current/design_harness_mas_71.md && grep -Eqi "clobber|883|structured" handoff/current/design_harness_mas_71.md'
VERIFICATION: PASS (exit 0)
```
mtime ordering (research → contract → design): `1784288555 < 1784288681 < 1784288753` (contract BEFORE generate).

## Criterion evidence
- **C1 (grounded with URLs):** design_harness_mas_71.md cites, per step: 71.1 → code.claude.com/docs/en/workflows
  (resumability) + multi-agent-research + harness-design; 71.2 → structured-outputs GA + building-effective-agents;
  71.3 → multi-agent-research judge rubric; 71.4 → multi-agent-research + building-effective-agents stopping
  conditions; 71.5 → effort doc + model-config doc; 71.6 → multi-agent-research + harness-design. All with URLs.
- **C2 (constraints + separation-of-duties):** the "Binding constraints" section states each verbatim, and calls
  out separation-of-duties + roster-snapshot handling for **71.1/71.3/71.4/71.5/71.6** — every downstream step
  EXCEPT the pure Layer-2 backend step 71.2 (all edit `.claude/agents/*.md`). *(Cycle-2 fix: the first Q/A
  correctly flagged that 71.6's envelope-return change edits `qa.md`/`researcher.md`, and 71.4's `coverage`-field
  addition edits `researcher.md`'s envelope block — both were missing from the enumeration; now added at the
  Binding-constraints list, the 71.4/71.6 per-step sections, and the Sequencing note.)*
- **C3 (rejected acknowledged):** all 15 rejected proposals (R1–R15) enumerated with disqualifiers + the rider-trap
  note (R1/R4/R11 ride on 71.1, R13/R14/R15 on the effort theme) — the design adopts the kept ideas WITHOUT them.

## Do-no-harm / scope honesty
71.0 is design + research only. NO production code changed (git: only handoff/ + masterplan/task files); NO
live-loop behavior change; historical_macro untouched; $0 metered (Workflow on the Opus Max rail). It delivers the
DESIGN for 71.1–71.6; it does NOT implement any change. The design deliberately DROPS one kept proposal (#8a
worst-of-N) whose grounding the researcher found oversold, and DESCOPES #12 to report-only — honest refinements,
not scope creep.
