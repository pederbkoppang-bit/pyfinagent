# Experiment Results -- phase-2.10 + phase-4.14.20 Audit Cycle

**Cycle type:** paired audit/formalization. Both steps are non-actionable as implementation items; this cycle closes missing audit artifacts and aligns masterplan status so forward-blockers don't trip.

## What was built

Two audit records + one masterplan status change. Zero code changes. Zero agent-file edits.

1. `handoff/phase-2.10-supersede.md` -- decision record naming `backend/agents/skill_optimizer.py` (lines 4, 129, 270, 453) as the Karpathy-autoresearch absorber. Unblocks phase-8.5.0's immutable `test -f handoff/phase-2.10-supersede.md`.
2. `handoff/phase-4.14.20-supersede.md` -- decision record explaining why the original immutable grep cannot be satisfied (two of three target files were deleted by phase-4.15.0 MAS restructure) and showing that semantic intent (trigger phrasing on the Agent-tool auto-delegation surface) is already satisfied by `.claude/agents/qa.md:3` + `.claude/agents/researcher.md:3`.
3. `.claude/masterplan.json` phase-4.14.20 status: `blocked` -> `superseded` with `superseded_by: phase-4.15.0` + `superseded_at` timestamp. Immutable `verification` block preserved verbatim.

## File list

Created:
- `handoff/phase-2.10-supersede.md`
- `handoff/phase-4.14.20-supersede.md`

Modified:
- `.claude/masterplan.json` (phase-4.14.20 status + superseded_by only; verification block unchanged)

NOT touched: any Python source, any `.claude/agents/*.md`, any test file, any schema / config / settings.

## Verification

```
$ test -f handoff/phase-2.10-supersede.md && echo "phase-2.10 file: OK"
phase-2.10 file: OK

$ test -f handoff/phase-4.14.20-supersede.md && echo "phase-4.14.20 file: OK"
phase-4.14.20 file: OK

$ grep -c 'use proactively\|MUST BE USED\|use immediately after' .claude/agents/qa.md .claude/agents/researcher.md
.claude/agents/qa.md:1
.claude/agents/researcher.md:1
```

(Per-file count is 1 because `grep -c` counts LINES matching, not total matches. Both files have the phrasing on their `description:` line; opening the file shows multiple phrases on that one line. Q/A may `head -3` each file to see the phrases.)

```
$ python3 -c "... (masterplan status inspection)"
2.10 -> superseded phase-8.5
4.14.20 -> superseded phase-4.15.0
```

Both steps now correctly marked with `superseded_by` pointing at their absorber. Phase-2.10 (already superseded before this cycle) is preserved as-is; phase-4.14.20 (was `blocked`) is now `superseded`.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `handoff/phase-2.10-supersede.md` exists with required sections | PASS |
| 2 | Masterplan phase-2.10 status stays `superseded` | PASS (unchanged) |
| 3 | Grep evidence for phase-4.14.20 captured | PASS (above) |
| 4 | Masterplan phase-4.14.20 status -> `superseded` with `superseded_by: phase-4.15.0`; verification preserved | PASS |
| 5 | `handoff/phase-4.14.20-supersede.md` exists with required sections | PASS |
| Read-only | Trigger phrases present in qa.md + researcher.md | PASS |
| Read-only | Both supersede files exist | PASS |
| Read-only | `superseded` count in masterplan increased by 1 | PASS |

## Known caveats

1. **`masterplan.json` diff includes `updated_at` timestamp change.** This is the only metadata field modified outside `phase-4.14.20`. Expected.
2. **Research brief's claim about `qa.md` containing "use immediately after" was slightly optimistic** -- the actual phrase is "immediately before marking a masterplan step done". The semantic intent is preserved (both are ordering triggers for auto-delegation). Audit record `phase-4.14.20-supersede.md` quotes the actual description string to avoid propagating the inaccuracy. Not a blocker; just a precision note.
3. **No Q/A-detectable regression is possible** because no behavior surface changed. The audit is pure metadata.
