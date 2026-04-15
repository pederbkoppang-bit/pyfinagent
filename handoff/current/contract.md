# Contract — Cycle 12: Phase 4.4.3.5 Incident Log P0 Verification

## Target
`docs/GO_LIVE_CHECKLIST.md` item 4.4.3.5: "Incident log shows no unresolved P0 incidents"

## Verification criteria (from checklist)
Read `.claude/context/known-blockers.md` and confirm no entry is tagged `P0` without a `resolved:` line. Any open P0 blocks launch until resolved or downgraded with Peder's explicit note.

## Approach
1. Write a stdlib-only drill at `scripts/go_live_drills/incident_log_p0_test.py` that:
   - Reads `known-blockers.md`
   - Scans for any line/section containing "P0" (case-insensitive)
   - For each P0 found, checks if it's in the RESOLVED section or has a "resolved:" line
   - Exits 0 if no unresolved P0s, exits 1 otherwise
2. Run the drill, verify exit 0
3. Flip checklist item `[ ]` -> `[x]` with evidence line
4. Commit and push

## Research gate
WAIVED — pure-doc verification item per rule 4. The file has already been read and contains zero P0 entries. The drill is a re-runnable verification of that fact.

## Success criteria
- SC1: Drill reads known-blockers.md without error
- SC2: Drill correctly identifies zero unresolved P0 entries
- SC3: Drill exits 0
- SC4: Checklist item flipped with evidence line matching existing format
- SC5: Drill is stdlib-only (pathlib, re, sys only)
- SC6: Drill is re-runnable and idempotent

## Scope
- `scripts/go_live_drills/incident_log_p0_test.py` — NEW file
- `docs/GO_LIVE_CHECKLIST.md` — 2 lines touched (checkbox flip + evidence)
- ZERO backend code touched
- ZERO edits to existing drill files

## DO NOT
- Edit any backend code
- Touch masterplan.json
- Manually update CHANGELOG.md
