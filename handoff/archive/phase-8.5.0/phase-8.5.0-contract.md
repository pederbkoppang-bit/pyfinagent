# Sprint Contract — phase-8.5 / 8.5.0 (Retire phase-2 step 2.10 stub)

**Step id:** 8.5.0 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Hypothesis

Write `handoff/phase-2.10-supersede.md` documenting that phase-2's 2.10 placeholder ("Karpathy Autoresearch Integration") is superseded by the 11-step phase-8.5 "Autonomous Strategy Research (Karpathy loop)". Confirm 2.10 already has `status: "superseded"` in masterplan (it does, from an earlier cycle).

## Immutable criterion

- `test -f handoff/phase-2.10-supersede.md`

## Plan

1. Write the supersede doc at `handoff/phase-2.10-supersede.md` (note: lives at `handoff/` root, not `handoff/current/`, per the criterion path).
2. Verify the file exists + that `phase-2 step 2.10` is marked `status: superseded` in `.claude/masterplan.json`.
3. Q/A, log, flip.

## Out of scope

- No code changes. No masterplan edit (2.10 already superseded).
- ASCII-only.
