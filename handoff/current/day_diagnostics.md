# Day diagnostics -- 2026-08-17

| step | verdict | IMMUTABLE criterion missed (quoted) | quality-gap-only findings | evaluator's NAMED fix | attempts | tokens |
|---|---|---|---|---|---|---|
| **86.94** | **FAIL** (park at R1) | criterion 5: *"any figure found to be unreproducible is CORRECTED IN EVERY FILE THAT CARRIES IT, not merely annotated in one -- a correction must replace, not accompany"* | `WARN_provenance_control_is_circular_overclaim` -- K1/K2 survive by provenancing a fixture to a file the author controls; no criterion names it | (1) regenerate §G's `ALL GREEN: 45 passed` (measured 77); (2) regenerate §H1's census `851` (measured 852, the +1 being this cycle's own commit); (3) correct J5 Class B's disposition, false for its own carrier; (4) withdraw "a control cannot be invented" | 3 today (6 overall) | ~2.2M FRESH at park |

Re-run of the failing evidence, by me, verbatim:

```
$ python scripts/qa/verify_no_sliding_windows_86_94.py | tail -1
ALL GREEN: 77 passed, 0 failed          <- §G claimed "45 passed"

$ python scripts/qa/verify_changelog_flip_86_91.py | tail -1
ALL GREEN: 42 passed, 0 failed          <- §G's sibling line, reproduces

$ node scripts/qa/verify_workflow_args_boundary.mjs | tail -1
ALL GREEN: 96 passed, 0 failed          <- §G's sibling line, reproduces

$ git ls-files scripts .claude/hooks backend | grep -E '\.(py|sh)$' | wc -l
852                                     <- §H1 claimed "tracked py/sh: 851"
```

Both stale figures confirmed. The two sibling lines in §G reproducing exactly is
what makes it a defect rather than a dated snapshot.

| **86.97** | **CONDITIONAL** (park: R3 ceiling, not the attempt cap) | none — **all 7 immutable criteria recorded MET and independently re-executed at cycles 4 and 5** | all three caps were quality gaps: `bump` parsed but asserted nowhere; a grep figure that did not reproduce under the command it named; superseded blocks left un-annotated. Then at cycle 5: W1 a false claim still standing, W2 a live survivor at the phase-emptied branch, W3 a truncated end-to-end drive | (1) pin `bump` in every scenario; (2) quote the command that produces each figure; (3) mark superseded blocks in place; (4) replace the false coverage sentence; (5) add a phase-emptied scenario; (6) fix the CHANGELOG seed separator | 2 today (5 overall) | ceiling hit at 4,585,189 / 4,500,000 |

Re-run of the failing evidence, by me, verbatim:

```
$ grep -cE "reach(es|ed)? the detector|pre-detector|bash exit|recursion guard|86\.97" handoff/current/live_check_86.91.md
4                     <- I had reported 5; grep -c counts LINES
$ grep -oE "...same pattern..." handoff/current/live_check_86.91.md | wc -l
5                     <- 5 matches across 4 lines; and my original 5 came from a
                         pattern with an extra alternative I never quoted

# W3, the truncated drive -- old seed restored, against a green 57/0 control:
FAILED: 52 passed, 1 failed
  FAIL [3] the drive REACHES the CHANGELOG write (a truncated heredoc looks
           identical to a successful end-to-end run)
```

**Both of today's steps parked with every immutable criterion met on the product
and every cap on my own evidence prose.** That is the single finding worth acting
on.
