# live_check — step 86.79

Everything below is **verbatim tool output**, re-runnable by:

```
source .venv/bin/activate
python scripts/qa/verify_counter_86_79.py      # 55 checks, exit 0
python scripts/qa/mutation_matrix_86_79.py     # 11 cells, exit 0
```

Every write in both scripts goes to a temp directory. The live repo is only READ.
`.claude/agents/qa.md` is **not modified by this step** — proof in §8.

> **THESE ARE THE CURRENT NUMBERS (cycle 3).** The captures below are kept verbatim
> as the record of what each Q/A actually graded — **not** because they are current:
>
> | sections | cycle | totals AS CAPTURED |
> |---|---|---|
> | §1–§7 | 1 | 42 checks / floor 30 / 7 cells |
> | §12–§14 | 2 | 50 checks / floor 48 / 9 cells — adds C3b, C3c, M8, M9; F401 removed |
> | **§15–§17** | **3** | **55 checks / floor 53 / 11 cells** — adds the C3c boundary regime, C3d crash-safety, M10, M11 |
>
> **Only §15–§17 reproduce against a current run.** Line numbers inside the earlier
> captures are likewise as-of-then and have since moved; the checker greps for its
> anchors at runtime and never relies on them.

---

## §0. Control — the immutable verification command, BEFORE any change

Run against the pre-change tree, so criterion 7's "control observed GREEN first"
has a real anchor:

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/qa_wip.py\").read())" && echo qa_wip-parses'
qa_wip-parses
exit=0
```

Same command **after** the change:

```
qa_wip-parses
exit=0
```

Note this command only proves the file parses. It is criterion 1 of the
verification gate, not the evidence — the evidence is §1–§7.

---

## §1–§6. `verify_counter_86_79.py` — full verbatim output

```
==========================================================================
C1 -- records_retained counts the CURRENT spawn (prior + 1)
==========================================================================
  [PASS] the producing statement is found by grep (line NOT hardcoded) -- qa_wip.py:507: "records_retained": len(records),
  [PASS] records_retained == priors + 1 -- priors=2 records_retained=3
  [PASS] len(prior_records) == priors
  [PASS] the NEW attempt_number is the same number, but unit-stated -- attempt_number=3 prior_attempts=2
  [PASS] records_retained_unit states it is a GAUGE, not a counter

==========================================================================
C2 -- the old number was right ONLY because write-first ran first
==========================================================================
  [PASS] records_retained DIFFERS across the write-first boundary (the coupling) -- before=2 after=3
  [PASS] the pre-write number is LOWER -- the coupling fails OPEN
  [PASS] attempt_number REFUSES before the write-first record exists -- status=no_record_for_this_spawn
  [PASS] ...and it refuses with the right reason
  [PASS] ...while still reporting the priors it genuinely knows -- prior_attempts=2
  [PASS] ...and it tells the reader what to do instead of falling back
  [PASS] attempt_number is correct once write-first has run -- attempt_number=3

==========================================================================
C3 -- pruning saturates records_retained; attempt_number survives it
==========================================================================
  [PASS] 6 records -> records_retained == 6 and attempt_number == 6
  [PASS] prune(keep=3) removed 3
  [PASS] records_retained SATURATES at keep (the defect, still visible) -- records_retained=3 (true attempts: 6)
  [PASS] attempt_number SURVIVES the prune and still reports 6 -- attempt_number=6 lost=3
  [PASS] the loss ledger accounts for exactly what was destroyed
  [PASS] the loss ledger is dot-prefixed (invisible to the verdict_wip_* globs)
  [PASS] ...and list_wip_records does NOT pick it up
  [PASS] the loss account is MONOTONIC across a no-op prune -- lost=3
  [PASS] ...and RISES when more is destroyed -- lost=5

  ENUMERATION -- automatic callers of prune_wip_records in the live tree.
  command: grep -rn 'prune_wip_records' --include='*.py' --include='*.js' --include='*.sh' . | grep -v -e '/\.venv/' -e '/node_modules/' -e '/\.git/' -e '^\./handoff/' -e '^\./docs/'
    ./scripts/qa/mutate_counter_source_86_21.py:56:    # convenience. `prune_wip_records(keep=DEFAULT_KEEP)` DELETES old records as
    ./scripts/qa/mutate_counter_source_86_21.py:93:print("           (prune_wip_records deletes by design) -- see the M2 comment.")
    ./scripts/qa/verify_counter_86_79.py:165:removed = qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
    ./scripts/qa/verify_counter_86_79.py:184:qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
    ./scripts/qa/verify_counter_86_79.py:189:qa_wip.prune_wip_records(SID, repo=tmp, keep=1)
    ./scripts/qa/verify_counter_86_79.py:195:print("\n  ENUMERATION -- automatic callers of prune_wip_records in the live tree.")
    ./scripts/qa/verify_counter_86_79.py:196:CMD = (r"grep -rn 'prune_wip_records' --include='*.py' --include='*.js' "
    ./scripts/qa/verify_counter_86_79.py:216:check("prune_wip_records has NO production caller (defect is LATENT, not live)",
    ./scripts/qa/verify_counter_86_79.py:240:qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
    ./scripts/qa/verify_counter_86_79.py:263:qa_wip.prune_wip_records(SID, repo=tmp, keep=qa_wip.DEFAULT_KEEP)
    ./scripts/qa/verify_wip_retention_86_36.py:129:        removed = qa_wip.prune_wip_records(step, repo=tmp, keep=3)
    ./scripts/qa/verify_wip_retention_86_36.py:135:            qa_wip.prune_wip_records(step, repo=tmp, keep=0)
    ./scripts/qa/qa_wip.py:127:#: attempts", which promises FOUR retained while `prune_wip_records` does
    ./scripts/qa/qa_wip.py:276:def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,
    ./scripts/qa/mutation_matrix_86_36.py:45:        anchor="def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,\n"
    ./scripts/qa/mutation_matrix_86_36.py:47:        repl="def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,\n"
  [PASS] prune_wip_records has NO production caller (defect is LATENT, not live) -- non-allowlisted hits: []
  [PASS] the enumeration is not vacuous -- it found the definition itself -- 16 hits total

==========================================================================
C4 -- the DOC and the CODE agree (in-module), and the residual is LOUD
==========================================================================
  [PASS] DEFAULT_KEEP and its comment block are found -- DEFAULT_KEEP = 3
  [PASS] the comment states the UNIT (TOTAL / INCLUSIVE), not just a name
  [PASS] the old off-by-one wording is GONE from the normative sentence
  [PASS] measured retention == keep TOTAL, exactly as the comment now claims -- retained=3 keep=3
  [PASS] the un-applied qa.md correction is written out for the operator -- handoff/current/qa_md_patch_86.79.md
  [PASS] ...and it names the exact line it would change

==========================================================================
C5 -- both escalation bounds still fire against the CORRECTED number
==========================================================================
  [PASS] OLD number does NOT reach F1b's ceiling after a prune (the bug) -- attempts_used=3/5
  [PASS] NEW number DOES escalate after a prune (the fix) -- attempts_used=6/5
  [PASS] the escalation summary is written and refuses to read as a pass
  [PASS] 1 CONDITIONAL -> auto-FAIL NOT armed -- consecutive=1
  [PASS] 2 consecutive CONDITIONALs -> auto-FAIL ARMED (the boundary fires) -- consecutive=2
  [PASS] a PASS RESETS the run -- consecutive=0
  [PASS] a missing ledger returns None, NOT 0

==========================================================================
C6 -- verdict semantics untouched; uncomputable == None, never 0
==========================================================================
  [PASS] missing sink -> attempt_number is None, not 0
  [PASS] empty sink -> attempt_number is None, not 0
  [PASS] no spawned_at -> attempt_number is None, not 0
  [PASS] NO report() variant ever carries a `verdict` key
  [PASS] every report() variant states is_verdict: false
  [PASS] budget exhaustion cannot produce a PASS under ANY flag combination -- close_kind over all flags -> ['ESCALATE']

==========================================================================
RESULT
==========================================================================
  checks run : 42   (cardinality floor 30)
  failed     : 0

  ALL CHECKS PASS
```

`exit=0`, captured directly (**not** through a pipe — `$?` after a pipe reads the
last stage, and `PIPESTATUS` is bash-only; this shell is zsh).

### Notes on the numbers above

- **`qa_wip.py:507`** is where `"records_retained": len(records)` now lives. It was
  at **:315** before this change and at **:316** in the research brief's inventory.
  The checker **greps for it at runtime and never hardcodes it** — three different
  values inside one cycle is exactly why.
- **C3's enumeration** — the population rule is stated in the command itself.
  A hit is classed non-production iff its file is on an **explicit allowlist**
  (`verify_counter_86_79.py`'s `NON_PRODUCTION` set). A pattern like `verify_*`
  was rejected: it would let a future production caller hide behind a
  checker-shaped filename.
- **C5's "OLD number" row is the defect made concrete.** Six real attempts, pruned
  to three: the old field feeds `attempts_used=3/5` → `CONTINUE`. The corrected
  number feeds `6/5` → `ESCALATE`. That is F1b's ceiling becoming reachable again.

---

## §7. `mutation_matrix_86_79.py` — full verbatim output

```
subject : scripts/qa/qa_wip.py  sha256[:16]=146600b722a02481
checker : scripts/qa/verify_counter_86_79.py

[CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (42 checks)

==========================================================================
MUTATION MATRIX
==========================================================================
  KILLED            M1-DROP-IDENTITY-GUARD         by: attempt_number REFUSES before the write-first record exists
  KILLED            M2-PRUNE-STOPS-ACCOUNTING      by: attempt_number SURVIVES the prune and still reports 6
  KILLED            M3-LOSS-LEDGER-CAN-DECREASE    by: the loss account is MONOTONIC across a no-op prune
  KILLED            M4-FAIL-OPEN-WITH-ZERO         by: missing sink -> attempt_number is None, not 0
  KILLED            M5-RESTORE-OFF-BY-ONE-COMMENT  by: the comment states the UNIT (TOTAL / INCLUSIVE), not just a name
  KILLED            M6-LEAK-A-VERDICT-KEY          by: NO report() variant ever carries a `verdict` key
  KILLED            M7-DROP-THE-UNIT               by: records_retained_unit states it is a GAUGE, not a counter

  subject sha256[:16] before=146600b722a02481 after=146600b722a02481 -> tracked file UNCHANGED
  cells: 7   killed: 7   survived/unearned: 0

  ALL CELLS KILLED
```

**The control was observed GREEN first** (`exit 0`, 42 checks) — printed above the
matrix by the script itself, not asserted afterwards.

### A cell that did NOT earn its kill, and was refused

M7's first draft produced a **broken string literal**, so the mutant crashed and
the checker went red **with no `[FAIL]` lines at all**. The matrix scored it
`RED-WRONG-REASON`, not `KILLED`, and exited 1:

```
  RED-WRONG-REASON  M7-DROP-THE-UNIT   expected one of ('records_retained_unit states it is a GAUGE, not a counter',), got no [FAIL] lines
  cells: 7   killed: 6   survived/unearned: 1
  MATRIX INCOMPLETE
```

This is the discrimination rule doing its job: **red is not a kill unless the
named assertion is among the failures.** The cell was rewritten to be
syntactically valid and then killed on the assertion it was aimed at.

---

## §8. Scope — what this step did and did not touch

```
$ git diff --stat -- scripts/qa/qa_wip.py
 scripts/qa/qa_wip.py | 227 +++++++++++++++++++++++++++++++++++++++++++++++++--
 1 file changed, 220 insertions(+), 7 deletions(-)

$ git status --porcelain -- handoff/current/ scripts/qa/
?? handoff/current/contract_86.79.md
?? handoff/current/qa_md_patch_86.79.md
?? handoff/current/research_brief_86.79.md
?? scripts/qa/mutation_matrix_86_79.py
?? scripts/qa/verify_counter_86_79.py

$ git diff --stat -- .claude/agents/qa.md CLAUDE.md .claude/rules/research-gate.md
(no output)
```

**`.claude/agents/qa.md` has a ZERO-line diff.** No fifth Main-authored edit was
made. See §9.

---

## §9. Regression — every other checker that exercises `qa_wip.py`

Derived by `grep -rln 'import qa_wip\|qa_wip\.' --include='*.py' scripts/ backend/ tests/`,
which returns 10 files; the 3 belonging to this step are excluded, `qa_wip.py`
itself is the subject, and `reproduce_wip_destruction_86_36.py` /
`simulate_qa_drop_86_31.py` are demonstrations rather than gates. The remaining
**5 gates all ran**:

| checker | exit | result |
|---|---|---|
| `verify_wip_retention_86_36.py` | 0 | `ALL GREEN -- 23 passed, 0 failed` |
| `verify_qa_write_first_86_31.py` | 0 | `ALL GREEN -- 244 passed, 0 failed` |
| `mutation_matrix_86_31.py` | 0 | `MATRIX: 24/24 KILLED` |
| `mutation_matrix_86_36.py` | 0 | `OK -- all 5 cells KILLED on a named assertion` |
| `mutate_counter_source_86_21.py` | 0 | `mutants surviving (undetected): 0` |

`mutation_matrix_86_36.py` reports the same subject digest
`146600b722a02481 -> 146600b722a02481`, independently confirming this step's
matrix did not write to the tracked file.

---

## §10. Live behaviour on a real step

```
$ python scripts/qa/qa_wip.py 86.32
  records_retained                 5
  attempt_number                   None
  prior_attempts                   None
  attempt_number_status            'no_spawn_identity'
  attempt_number_is_lower_bound    True
  records_pruned_known             None
  source_present                   True
  is_verdict                       False
  has verdict key: False
```

Without `--spawned-at` the counter now **refuses** instead of handing back a
number whose correctness depends on a rule in another file. `records_retained`
is unchanged at 5 — no live number was silently shifted by this step.

---

## §11. What I could NOT verify — stated, not hidden

1. **Hand-deleted records remain undetectable.** The loss ledger accounts for the
   only automated deleter (`prune_wip_records`). A record removed with `rm` at any
   time, or before this change shipped, is invisible to every counter in the
   system. `attempt_number_is_lower_bound` exists for exactly this and is `True`
   for every live step today, because no ledger exists yet for any of them.
2. **`attempt_number_is_lower_bound` is a heuristic, not a proof.** It is `True`
   iff no loss account exists **and** the retained set has reached `DEFAULT_KEEP`
   — the regime in which a default prune *would* have removed something. Below the
   window it reports `False`, which is a claim about *automated* loss only.
3. **The fix is not yet consumed by anything.** `qa.md` still points the Q/A at
   `records_retained`, so until route A or B in `handoff/current/qa_md_patch_86.79.md`
   is taken, the new fields are **available but unread** by the live rail. The
   step improves the instrument; it does not yet change what the rail measures
   with. This is the honest limit of what criterion 4 could reach here.
4. **Wiring `prune_wip_records` was not attempted and is still out of scope.** The
   saturation defect remains **latent**; this step makes it *safe if wired*.
5. **No claim is made that the mutation matrix is exhaustive.** It licenses
   exactly one statement: these 7 mutations were killed by named assertions.

---

# CYCLE 2 — evidence for the four cycle-1 findings

## §12. verify_counter_86_79.py — the two NEW sections (verbatim)

```
C3b -- loss account x no_record_for_this_spawn (the branch N6 survived in)
==========================================================================
  [PASS] precondition: the loss account is NON-ZERO, so the add-back term is LIVE -- lost=3 -- if this were 0 the assertion below could not fail and would be vacuous
  [PASS] ...the branch under test is actually reached -- status=no_record_for_this_spawn
  [PASS] prior_attempts counts PRUNED-AWAY records too, not just retained ones -- prior_attempts=6 (retained=3, lost=3; without the add-back it would be 3)
  [PASS] ...and that difference is what makes F1b's ceiling reachable on this path -- attempt 7/5 -> ESCALATE (dropping the add-back gives 4/5 -> CONTINUE)

==========================================================================
C3c -- attempt_number_is_lower_bound actually discriminates
==========================================================================
  [PASS] below the retention window with no ledger -> NOT flagged a floor -- retained=2 keep=3
  [PASS] at/above the window with no loss account -> FLAGGED as a floor -- retained=5
  [PASS] once the loss IS accounted -> no longer a floor -- lost=2
  [PASS] the three states are not all the same value (the field discriminates)

```

## §13. Full checker + matrix totals after cycle 2 (verbatim)

```
  checks run : 50   (cardinality floor 48)
  failed     : 0
  ALL CHECKS PASS

subject : scripts/qa/qa_wip.py  sha256[:16]=146600b722a02481
checker : scripts/qa/verify_counter_86_79.py

[CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (50 checks)

==========================================================================
MUTATION MATRIX
==========================================================================
  KILLED            M1-DROP-IDENTITY-GUARD         by: attempt_number REFUSES before the write-first record exists
  KILLED            M2-PRUNE-STOPS-ACCOUNTING      by: attempt_number SURVIVES the prune and still reports 6
  KILLED            M3-LOSS-LEDGER-CAN-DECREASE    by: the loss account is MONOTONIC across a no-op prune
  KILLED            M4-FAIL-OPEN-WITH-ZERO         by: missing sink -> attempt_number is None, not 0
  KILLED            M5-RESTORE-OFF-BY-ONE-COMMENT  by: the comment states the UNIT (TOTAL / INCLUSIVE), not just a name
  KILLED            M8-DROP-LOSS-ADDBACK-ON-NO-RECORD-PATH by: prior_attempts counts PRUNED-AWAY records too, not just retained ones
  KILLED            M9-LOWER-BOUND-FLAG-NEVER-SET  by: at/above the window with no loss account -> FLAGGED as a floor
  KILLED            M6-LEAK-A-VERDICT-KEY          by: NO report() variant ever carries a `verdict` key
  KILLED            M7-DROP-THE-UNIT               by: records_retained_unit states it is a GAUGE, not a counter

  subject sha256[:16] before=146600b722a02481 after=146600b722a02481 -> tracked file UNCHANGED
  cells: 9   killed: 9   survived/unearned: 0

  ALL CELLS KILLED
```

## §14. Lint gate — the cycle-1 RED is now GREEN

Scope DERIVED, not assumed: `git diff --name-only HEAD -- '*.py'` UNION
`git ls-files --others --exclude-standard -- '*.py'`, asserted non-empty.

```
scripts/qa/mutation_matrix_86_79.py
scripts/qa/qa_wip.py
scripts/qa/verify_counter_86_79.py
COUNT=3
$ uvx ruff check --select F821,F401,F811 <the 3 files>
All checks passed!
ruff EXIT=0
```

---

# CYCLE 3 — evidence for the cycle-2 findings

The cycle-2 Q/A executed two mutants that survived (Q-A boundary, Q-E crash
safety) and found this artifact set internally stale. Sections 12-14 above are
the CYCLE-2 capture; the current run is below and supersedes their totals.

## §15. verify_counter_86_79.py -- the two NEW sections (verbatim)

```
C3c -- attempt_number_is_lower_bound actually discriminates
==========================================================================
  [PASS] precondition: the three unaccounted regimes really are below / AT / above -- retained = 2 / 3 / 5, keep=3
  [PASS] below the retention window with no ledger -> NOT flagged a floor
  [PASS] EXACTLY AT the window with no ledger -> FLAGGED (the boundary itself) -- retained == keep == 3; this is the state a legacy prune leaves behind, so `>` instead of `>=` must NOT survive
  [PASS] above the window with no loss account -> FLAGGED as a floor
  [PASS] once the loss IS accounted -> no longer a floor -- lost=2
  [PASS] the field discriminates (not all regimes agree)

==========================================================================
C3d -- crash mid-prune OVER-counts (the documented safe direction)
==========================================================================
  [PASS] precondition: the unlink really did fail (nothing was deleted) -- removed=0 still_on_disk=6
  [PASS] the loss was ALREADY recorded when the crash hit -> OVER-count, not None -- read_loss=3 -- recording AFTER the unlink would give None (or 0), which UNDER-counts and suppresses escalation
  [PASS] ...so the derived count errs STRICTLY HIGH after a crash, never low -- prior_attempts=9 -- must EXCEED the 6 real attempts (6 survivors + 3 already-accounted). Accounting after the unlink gives exactly 6, so `>= 6` would not discriminate and `> 6` does

```

## §16. Full run, current (verbatim)

```
  checks run : 55   (cardinality floor 53)
  failed     : 0
  ALL CHECKS PASS

subject : scripts/qa/qa_wip.py  sha256[:16]=146600b722a02481
checker : scripts/qa/verify_counter_86_79.py

[CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (55 checks)

==========================================================================
MUTATION MATRIX
==========================================================================
  KILLED            M1-DROP-IDENTITY-GUARD         by: attempt_number REFUSES before the write-first record exists
  KILLED            M2-PRUNE-STOPS-ACCOUNTING      by: attempt_number SURVIVES the prune and still reports 6
  KILLED            M3-LOSS-LEDGER-CAN-DECREASE    by: the loss account is MONOTONIC across a no-op prune
  KILLED            M4-FAIL-OPEN-WITH-ZERO         by: missing sink -> attempt_number is None, not 0
  KILLED            M5-RESTORE-OFF-BY-ONE-COMMENT  by: the comment states the UNIT (TOTAL / INCLUSIVE), not just a name
  KILLED            M8-DROP-LOSS-ADDBACK-ON-NO-RECORD-PATH by: prior_attempts counts PRUNED-AWAY records too, not just retained ones
  KILLED            M9-LOWER-BOUND-FLAG-NEVER-SET  by: above the window with no loss account -> FLAGGED as a floor
  KILLED            M10-LOWER-BOUND-MISSES-THE-BOUNDARY by: EXACTLY AT the window with no ledger -> FLAGGED (the boundary itself)
  KILLED            M11-RECORD-LOSS-AFTER-THE-UNLINK by: the loss was ALREADY recorded when the crash hit -> OVER-count, not None
  KILLED            M6-LEAK-A-VERDICT-KEY          by: NO report() variant ever carries a `verdict` key
  KILLED            M7-DROP-THE-UNIT               by: records_retained_unit states it is a GAUGE, not a counter

  subject sha256[:16] before=146600b722a02481 after=146600b722a02481 -> tracked file UNCHANGED
  cells: 11   killed: 11   survived/unearned: 0

  ALL CELLS KILLED
```

## §17. A cell REFUSED again, and again correctly

After C3c's assertions were renamed, M9 came back RED but not on the assertion
it was aimed at, so the matrix scored it `RED-WRONG-REASON` and exited 1:

```
  RED-WRONG-REASON  M9-LOWER-BOUND-FLAG-NEVER-SET  expected one of ('at/above the window with no loss account -> FLAGGED as a floor', ...), got ['EXACTLY AT the window with no ledger -> FLAGGED (the boundary itself)', 'above the window with no loss account -> FLAGGED as a floor', 'the field discriminates (not all regimes agree)']
  cells: 11   killed: 10   survived/unearned: 1
  MATRIX INCOMPLETE
```

M9 was then repointed at the ABOVE-the-window assertion specifically, which
M10's boundary mutation cannot break -- so the two cells stay distinguishable
instead of both being scored by whichever assertion happens to fire first.
