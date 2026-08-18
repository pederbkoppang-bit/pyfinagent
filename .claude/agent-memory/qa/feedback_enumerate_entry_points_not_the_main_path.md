---
name: enumerate-entry-points-not-the-main-path
description: A module's CLI subcommands are guards nobody drives -- the matrix targets the hook path and the self-test never calls the CLI, so the --reason accountability guard survived both (86.71 c2)
metadata:
  type: feedback
---

When a step ships a module with MORE THAN ONE entry point (a hook `main()` path
plus `--status` / `--operator-extend` / `--self-test` subcommands), enumerate the
entry points and check coverage per entry point. Do not assume the mutation
matrix plus the self-test together cover the module.

**Why:** phase-86.71 cycle 2. `scripts/harness/attempt_gate.py` had a 6-cell
matrix (all six cells drive `handle_hook()` via subprocess) and a 12-check
`--self-test`. I mutated five guards myself against `--self-test` as the oracle,
control rc=0 observed first:

    H1 hostile step-id refusal removed          -> KILLED
    H2 PASS exception removed                   -> KILLED
    H3 operator extension ignored               -> KILLED
    H5 corrupt row skipped (== matrix cell G4)  -> KILLED
    H4 `if not reason.strip():` -> `if False:`  -> SURVIVED, rc=0

H4 is `cmd_extend`'s `--reason` requirement -- the accountability guard on the
ONLY path that raises the ceiling, whose own docstring says "an unexplained
extension is exactly the silent act this gate exists to prevent". Zero coverage:
no matrix cell reaches `cmd_extend`, and `--self-test` never calls it. Both
artifacts looked comprehensive and the gap was between them, not inside either.

**How to apply:** grep the module for `def cmd_*` / the `main()` argv dispatch
and list every branch. For each, ask which artifact DRIVES it. A subcommand that
appears in no test's argv and in no matrix cell's stdin is uncovered no matter
how many cells and checks the step reports. Pairs with
[[feedback_class_guard_bound_to_the_helper_not_the_call_site]] (the guard tested
one seam away) and [[feedback_mutate_each_duplicated_site_individually]] (the
aggregate cell hiding an unguarded twin).

Second finding from the same cycle, worth the same reflex: a criterion demanding
"the command stated next to each number" was answered with a
`$ python3 - <<'PY'` block whose body was `...  # (the exact script is quoted in
full in the session transcript)`. A shell-capture-shaped block containing an
ellipsis and a pointer to a transcript is NOT a stated command -- nothing in the
handoff tree can be re-run. Re-derive the number yourself from the stated
population rule (I did, and it reproduced to the decimal), then report the
missing command separately: the number being right does not discharge the clause.
