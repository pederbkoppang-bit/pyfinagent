# Misattribution notice -- `phase-38`

Written by `scripts/housekeeping/quarantine_misattributed_archives.py`
(phase-75.11.4, criterion 12). **Additive only: nothing in this directory was
moved, renamed, or deleted.**

## What was measured

This directory is named for step **38**, but the `contract.md` it
contains declares step **62.2**.

Cause (phase-86.29, fixed at source): `.claude/hooks/archive-handoff.sh` used
to copy the four BARE rolling filenames (`contract.md`, `experiment_results.md`,
`evaluator_critique.md`, `research_brief.md`) into the closing step's directory
regardless of which step last wrote them, while that step's own suffix-named
artifacts (`contract_38.md`, ...) stayed behind in `handoff/current/`.
The hook now derives names from the step id and guards the rolling fallback
with a content declaration check, so directories created after that fix are not
affected.

## What this notice does NOT claim

- It does not claim the OTHER files here belong to another step; only
  `contract.md` was classified.
- It does not claim step 38 did not run. Its real artifacts may still be
  in `handoff/current/` under `*_38.md` names.
- The classifier's measured precision is **0.9936**, not 1.0.

## How to re-derive this

    python scripts/qa/derive_archive_misattribution_86_29.py --list-wrong

That script validates recall against two known positives and REFUSES to print a
census if it misses either, so a run that prints is itself evidence the method
has recall.
