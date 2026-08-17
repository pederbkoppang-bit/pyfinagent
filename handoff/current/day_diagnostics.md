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
