# Contract -- 67.3 Researcher write-first / incremental-brief discipline

Step: masterplan phase-67 / 67.3 (P1, depends_on 67.1). Research gate: PASSED (simple
tier, floors met; research_brief_67_3.md -- 5 sources read in full, 27 URLs, recency
scan done, 6 internal files). Note the brief itself was written incrementally from the
first tool call -- a live demonstration of the discipline this step codifies.

## Research-gate summary

- The 2026-05-16 incident (132K tokens, 53 tool calls, ZERO brief) is the named failure
  mode; Anthropic's Fable prompting guide describes it exactly ("end a turn with a
  text-only statement of intent without issuing the corresponding tool call") and the
  remedy (check the last paragraph; do the write now).
- Ownership map (three-file non-duplication invariant): researcher.md owns behavioral
  directives -> write-first lives THERE; research-gate.md gets a cross-ref subsection
  only; ARCHITECTURE.md unchanged (owns the floor record).
- STALE #1: per-step-protocol.md:33 "Researcher (sonnet)" -- only "(sonnet)" in file.
- STALE #2: researcher.md Domain context -- "May 2026 go-live" is past; "DSR 0.9984" is
  factually WRONG (optimizer_best.json:28 says dsr=0.9525811; Sharpe 1.1705 matches).
  Hardcoded metrics drift every optimizer run -> point to the file instead.
- Floors to preserve verbatim (grep-verifiable): >=5 read-in-full, >=10 URLs, recency
  scan, source hierarchy, JSON envelope (external_sources_read_in_full), deep-tier
  >=20/adversarial/multi-pass, effort table.
- Fable de-prescription: the new section must be SHORT (goal + invariant, <=8 lines);
  phrase as "write the BRIEF" (deliverable), never "narrate your thinking"
  (reasoning_extraction risk).

## Hypothesis (falsifiable)

Codifying write-first in the agent definition makes the zero-output failure mode
structurally impossible (any gate-failing session still leaves a partial brief +
honest envelope on disk), verifiable by the presence of the directive in the
definition and by subsequent researcher spawns' file mtimes preceding their final
messages -- already evidenced by all three 67.x briefs this session.

## Success criteria (verbatim from .claude/masterplan.json 67.3 -- IMMUTABLE)

1. ".claude/agents/researcher.md codifies WRITE-FIRST: the brief artifact is created
   early and written incrementally as sources are read, and a session that cannot meet
   the gate still leaves a partial brief plus an honest gate_passed:false envelope"
2. "No research-gate floor is weakened: the >=5 read-in-full floor, >=10 URL
   collection, mandatory recency scan, source-quality hierarchy, and the JSON envelope
   all remain (grep-verifiable)"
3. "Stale scaffolding pruned: the runbook 3-agent diagram no longer labels Researcher
   '(sonnet)'; hardcoded point-in-time metrics in researcher.md are dated or removed;
   changes stay cross-linked and non-duplicative with .claude/rules/research-gate.md"
4. "Fresh Q/A PASS on the diff"

## Design (files; texts per the brief's R1-R4)

1. `.claude/agents/researcher.md`: insert "## Write-first (non-negotiable)" section
   (R1 text, 8 lines) after "## When invoked"; replace the two stale Domain-context
   lines (R2): go-live line -> "live paper-trading (US + EU + KR paper markets)";
   metrics line -> single-source-of-truth pointer to
   backend/backtest/experiments/optimizer_best.json with a no-hardcode note.
2. `docs/runbooks/per-step-protocol.md:33`: diagram cell "Researcher (sonnet)" ->
   "Researcher" and "Q/A (opus)" -> "Q/A" (R3; models live in frontmatter, not the
   diagram; preserve box widths -- left interior 24 chars, right 19).
3. `.claude/rules/research-gate.md`: add the short "## Write-first discipline"
   cross-ref subsection (R4; points at researcher.md as owner, duplicates no wording).

## Anti-patterns guarded

- Floor erosion: append/replace edits only; Q/A greps every floor string post-diff.
- Over-prescription (Fable guidance): 8-line directive, goal + invariant, no procedure.
- reasoning_extraction: directive phrased on the BRIEF artifact, not on thinking.
- Rule duplication across the three research-gate files: cross-ref only.

## Out of scope

Effort-tier table values; deep-tier requirements; any change to gate LOGIC; qa.md;
CLAUDE.md; agent_definitions.py stale text (registered separately).

## Risk

- ASCII diagram misalignment if widths drift -> counted replacement strings in R3.
- A future optimizer run makes even the pointer text stale -> pointer names the FILE,
  not values; nothing left to drift.
