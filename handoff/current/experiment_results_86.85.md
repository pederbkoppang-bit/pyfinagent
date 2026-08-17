# Experiment results -- step 86.85

**Step:** the verdict ledger is never written for the step being evaluated, so the
3rd-CONDITIONAL auto-FAIL rule has no input and cannot ever fire.
**Date:** 2026-08-15. **Verification command:** GREEN, `parses`, exit 0.

---

## 0. Headline

**The writer existed nowhere; it exists now, and the rule it feeds is proven to fire
by execution.** With the ledger populated, `enforceEscalation` -- the shipped
consumer, run unmodified -- computes `consecutive_conditionals = 2`,
`would_auto_fail = true` from 86.74's priors **unaided**. That is the auto-FAIL Main
had to compute BY HAND earlier today because the sequence arrived as prose.

**The step was localised before anything was built, and it survived the re-scope
test the criteria demand.**

---

## 1. What was changed

| File | Change |
|---|---|
| `scripts/qa/verdict_ledger_write.py` | **new** -- the writer. Append-one-row, dedup-refusing, fail-loud, append-only. Carries `--emit-sequence` and `--self-test`. |
| `scripts/qa/mutation_matrix_86_85.py` | **new** -- mutation matrix, zero repo writes. **10 cells** as of the cycle-3 remediation. It shipped with 5; cycle 1 proved those 5 blind to ORDERING (+M6, M7), cycle 2 proved 7 still left NEW guards uncovered (+M8 I/O fail-loud, M9 step_id-in-key, M10 empty-step). See §5 and §6. |
| `backend/tests/test_phase_86_85_verdict_ledger_write.py` | **new** -- 25 pytest regressions, one per mutation cell. Promised in contract §6.3 and initially substituted by `--self-test` **without disclosure**; the cycle-2 Q/A flagged that as a scope-honesty violation and it is now discharged. |
| `handoff/verdict_ledger.jsonl` | 35 -> 43 rows at `d1c4a79d`; 86.74 backfilled 0 -> 8 labelled rows. Now 45 in the working tree, the +2 being this step's own cycle-1 and cycle-2 FAIL rows. |

**No production/trading code was touched.** No flag promoted, no `.env` written, no
manual cycle, no restart.

---

## 2. Criterion-by-criterion

### C1 -- failure LOCALISED before anything was built ✅

Commands and output are in `contract_86.85.md` §2 and reproduced in
`live_check_86.85.md` §1. Result: **CAUSE = NEVER-WRITTEN.**

- **not wrong-key** -- the positive control `--step 86.21` returns `status=ok` with
  5 verdicts through the *same reader* and the *same key*.
- **not pruned** -- the file is append-only and its max date was 2026-08-11,
  predating every 86.74 verdict.
- **not only-after-close** -- 86.21 is `status=pending` and has rows.
- all 35 pre-existing rows carry `recorded_by=main`; 12 share one microsecond
  `recorded_at`, i.e. bulk backfills, never a per-verdict write.

**The criterion also asks: if 86.74's verdicts ARE on disk somewhere, say so and
re-scope rather than build. They partly are, and the step still stands.** They exist
as **10 WIP artifacts** and as **3 narrative cycle sections** in
`evaluator_critique_86.74.md`. Neither substitutes:

- `qa_wip.py`'s own guidance: a WIP record is *"EVIDENCE FOR THE NEXT SPAWN ONLY --
  never count it as a verdict"*, and `records_retained` is *"a gauge, not a counter"*.
- the critique is prose; no machine reads a sequence from it.

**And a third candidate was eliminated by fresh measurement this session:**
`handoff/harness_log.md` cannot carry a per-step sequence either -- 86.74's cycle 5
was **never given a row at all**, and cycle numbers there are **not unique** (two
independent 193/194/195 runs exist). All three substitutes fail.

### C2 -- population rule stated beside every count, enumeration command quoted ✅

**Population = EVERY non-blank line in `handoff/verdict_ledger.jsonl`.**

**THESE COUNTS ARE ANCHORED, because this step's ledger MOVES WHILE THE STEP RUNS.**
86.85 records its own Q/A verdicts into the very file it counts, so any unanchored
total goes stale the moment the next cycle is graded -- which is exactly what the
cycle-2 Q/A caught (it measured 44 against a stated 43, the drift being this step's
own cycle-1 FAIL row). Each figure below therefore names the commit it was taken at.

```
$ git show d1c4a79d~1:handoff/verdict_ledger.jsonl | python3 -c "..."   [PRE-STEP]
  total rows            : 35
  step_ids present      : 10
  86.74 rows            : 0
  recorded_by           : {main: 35}
  run_id                : 35/35 key present, 35/35 non-empty, 35/35 wf_-prefixed
  max date              : 2026-08-11

$ git show d1c4a79d:handoff/verdict_ledger.jsonl | python3 -c "..."     [AS SHIPPED]
  total rows            : 43   (= 35 + 8 backfilled 86.74 rows)
  step_ids present      : 11
  86.74 rows            : 8
  verdict distribution  : {CONDITIONAL 23, FAIL 5, PASS 8, NO_VERDICT 7}
  rows with recorded_at : 29 / 43

$ python3 -c "rows=[json.loads(l) for l in open('handoff/verdict_ledger.jsonl') if l.strip()]"
                                                                        [WORKING TREE]
  measured 2026-08-17 after the cycle-5/6 writes (this figure MOVES while the
  step runs -- the ledger records this step's own verdicts -- so it carries its
  date and command, never a bare number; the cycle-5 Q/A caught 45-vs-46-vs-47
  disagreeing across three artifacts):
  total rows            : 52   (= 47 at the cycle-5 measurement + 4 backfilled
                                86.84 rows + this step's cycle-5 row)
  step_ids present      : 13
  verdict distribution  : {CONDITIONAL 29, FAIL 8, PASS 8, NO_VERDICT 7}
  rows with recorded_at : 38 / 52
```

`rows with recorded_at` is short of the total because **14 historical rows predate
the field**; every row this step writes carries it. Stated rather than rounded up.

### C3 -- cross-session persistence DEMONSTRATED across two process invocations ✅

Not asserted -- executed as two separate `python` invocations:

```
$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --step 77.1 \
      --verdict CONDITIONAL --run-id wf_p1 --cycle 1 --note proc1
{"step_id": "77.1", ..., "verdict": "CONDITIONAL", ...}          exit=0

$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --emit-sequence --step 77.1
["CONDITIONAL"]                                                   exit=0
```

*(A first attempt at this demonstration reported FAILED. That was a **probe defect**,
not a writer defect: the driver put the command in a shell variable and zsh does not
word-split unquoted parameters, so the write never ran and the read-back correctly
found nothing. Recorded rather than quietly re-run -- a red check that indicts the
probe is exactly as misleading as a green one that cannot fail.)*

### C4 -- the 3rd-CONDITIONAL rule proven to FIRE by DRIVING it ✅

The **shipped** `enforceEscalation` (lines 319-370 of `.claude/workflows/qa-verdict.js`)
was extracted and executed unmodified, fed sequences read back **out of the ledger**:

```
CONTROL  1 prior C + current CONDITIONAL       n=1  would_auto_fail=false
DRIVEN   2 prior C + current CONDITIONAL       n=2  would_auto_fail=true
CONTROL  2 prior C + current PASS              n=2  would_auto_fail=false
CONTROL  2 prior C + current FAIL              n=2  would_auto_fail=false
```

The first row is the anti-vacuity control: the rule does **not** fire early. And on
86.74's real backfilled priors:

```
priors from ledger: ["NO_VERDICT","NO_VERDICT","CONDITIONAL","CONDITIONAL","PASS","CONDITIONAL","CONDITIONAL"]
consecutive_conditionals = 2   would_auto_fail = true
```

**So the auto-FAIL that Main computed by hand today would have fired unaided.**

### C5 -- interaction with 86.79 and 86.45 resolved explicitly in writing ✅

- **86.45** owns whether a `NO_VERDICT` row grades. This step **records** a drop
  faithfully and **changes no counting semantics**. Measured rather than assumed:
  the shipped consumer already skips it (`if (v === 'NO_VERDICT') continue` -- it
  *neither extends nor resets*), so recording a drop **cannot** clear an escalation
  today. 86.85 preserves that; it does not re-decide it.
- **86.79** owns the `records_retained` off-by-one in the **parallel** `qa_wip.py`
  trail. This step writes a **different file** and never reads `records_retained`.
  No overlap, so no blocker.
- **86.71** (cumulative budget) would be the ledger's second consumer; out of scope.
- **86.21** owns the counter's in-flight blindness; out of scope.

### C6 -- a drop must not CLEAR an escalation; a missing row must not read as zero ✅

Both driven, not argued:

```
[C,C,NO_VERDICT] + CONDITIONAL -> n=2   would_auto_fail=true    (a drop does NOT reset)
absent sequence                -> n=null would_auto_fail=null
                                  sequence_status=not_supplied  (unknown, NOT zero)
```

**And the recorder was proven to have run before any zero was treated as evidence**
-- the positive control (`--step 86.21` -> `status=ok`, 5 verdicts) is what licenses
reading 86.74's zero as a measured zero rather than a broken reader.

**A design decision that this criterion forced, stated explicitly.** Cycle 7's row
records `CONDITIONAL` -- **what the rail returned** -- not the `FAIL` Main recorded
as the cycle outcome. Storing the derived FAIL would reset the consecutive run and
**clear the very escalation it represents**. The FAIL is computed by the consumer
from the sequence; it is deliberately not stored. This is why the ledger row and the
`harness_log` row differ, and they differ correctly: the ledger records the **verdict
event**, the log records the **cycle outcome**.

### C7 -- verdict semantics UNCHANGED ✅

- The writer **records**; it never transforms. The vocabulary guard **REJECTS**
  anything outside `{PASS, CONDITIONAL, FAIL, NO_VERDICT}` (exit 3) rather than
  coercing -- coercion is how a non-PASS would silently become something else.
- Demonstrated above: with 2 prior CONDITIONALs, a current **PASS** stays
  `would_auto_fail=false` and a current **FAIL** stays `false`. Nothing turns a
  non-PASS into a PASS.
- `qa.md` is untouched. `verdict_history_86_21.py` is untouched. `qa-verdict.js` is
  untouched. The only new surface is a writer.

### C8 -- mutation-tested, control GREEN first, byte-identical ❌ -> ✅ (see §5)

> **THIS SECTION WAS WRONG AS SHIPPED, and the cycle-1 Q/A proved it.** "5 cells,
> 5 KILLED" was true and *irrelevant to the property that mattered*: none of M1-M5
> touched ORDERING, and my self-test's ordering check used a **palindromic**
> fixture, so a mutant reversing `emit_sequence` **survived with every check
> green**. The matrix is now **7 cells, 7 KILLED**, with M6 = that exact surviving
> mutant. Read §5 before trusting the table below.

`scripts/qa/mutation_matrix_86_85.py` -- as originally shipped, **5 cells, 5 KILLED,
0 survived, 0 unscorable**, control observed GREEN *before* any cell ran:

| cell | guard reverted | result |
|---|---|---|
| M1 | remove the dedup refusal | KILLED (4 checks red) |
| M2 | remove the verdict vocabulary guard | KILLED |
| M3 | allow an unkeyed row | KILLED |
| M4 | swallow a corrupt ledger line | KILLED |
| M5 | collapse event time into write time | KILLED |

**Zero repo writes** -- every mutant is a temp copy executed in the OS tmpdir, so
there is no restore step to get wrong. `sha256` of the target printed before and
after and compared in-run: `146cf84e...1904` both times, `UNCHANGED: True`.

A cell that exits non-zero **without** `SELF-TEST FAILED` in its output is scored
**UNSCORABLE, not KILLED** -- otherwise a typo'd mutation would score as a kill it
did not earn (cf. `pytest exit 5 scores as a KILL`).

---

## 3. The measurement that changed the design

The brief flagged as **UNPROVEN** whether a `PostToolUse` hook can see the returned
verdict, and required it be measured before being designed around. Measured with a
temporary probe hook in the gitignored `.claude/settings.local.json`, restored
**byte-identical** (sha256 `8f03f194...66` before and after, compared in-run):

| question | answer |
|---|---|
| Does a PostToolUse hook receive `tool_response`? | **YES** |
| Does the matcher fire on `Workflow`? | **YES** (`tool_name: "Workflow"` observed) |
| Does `tool_response` carry the **verdict**? | **NO** |

`tool_response` for a Workflow is the **launch receipt** --
`{runId, scriptPath, status, summary, taskId, taskType, transcriptDir, workflowName}`.
Workflows *always* run in the background, so PostToolUse fires at launch and there is
no foreground mode. **A hook can therefore never author a verdict row.** This is
structural, not a quirk.

**Consequence:** the writer is an explicit call by Main at the seam -- the brief's
own stated fallback (§B). A hook remains viable *later* as a pure **alarm**, since it
does see `runId` at launch, which is exactly the key needed to notice a launched run
that never got a row. **That alarm is out of scope here** (86.85 = the WRITER only)
and is recorded as a follow-up.

---

## 4. What I could NOT verify

1. **The writer is not yet WIRED into the seam.** It exists, is tested, and was used
   to backfill -- but nothing *automatically* calls it when a verdict returns. Today
   Main must remember. The measurement above establishes that a hook cannot close
   that gap by authoring the row; the honest position is that **un-forgettability is
   not yet solved**, only made possible.
2. **The backfilled 86.74 rows are a RECONSTRUCTION**, not contemporaneous records.
   They are labelled as such in every `note` field, with their source named
   (`harness_log` row or `evaluator_critique` section). Two of them
   (`1-drop-a`/`1-drop-b`) split a single `harness_log` line that says "rail dropped
   x2" and have no `run_id` -- they are keyed by cycle label instead. A reconstruction
   is weaker evidence than a live write and is marked so rather than blended in.
3. **`recorded_at` is absent on 14 of 43 rows** because they predate the field.
   Those rows cannot distinguish event time from write time retrospectively.
4. **Only one consumer is proven.** `enforceEscalation` is driven end-to-end;
   `attempt_budget.py` (86.71) is still inert and unwired, so the ledger's second
   intended consumer remains hypothetical.
5. **No end-to-end run has yet used the ledger to supply `args.verdict_sequence`
   on a live spawn.** The mechanism is proven by driving the shipped function with
   ledger-sourced data; it has not yet ridden a real Q/A launch.

---

## 5. Cycle-2 remediation -- the cycle-1 Q/A returned FAIL, and it was right

**Verdict: FAIL** (`wf_5f5ce4b6-266`). 6 of 8 criteria met; **C8 and C2 not met**
and the mandatory ruff gate RED. Every blocker was reproduced by Main before being
fixed. Full verbatim return: `evaluator_critique_86.85.md`.

### C8 -- I wrote a guard named for a property it could not test

The self-test check named `"sequence is oldest->newest"` asserted against
`["CONDITIONAL","CONDITIONAL","CONDITIONAL"]` -- **a palindrome**. The Q/A's mutant
`emit_sequence -> return out[::-1]` therefore **SURVIVED with all 11 checks green,
including that one.** Reproduced by Main: reversing the function and re-running
`--self-test` prints `SELF-TEST PASSED`, exit 0.

Materiality, driven on the shipped `enforceEscalation`:

```
oldest->newest  [PASS,C,C] + CONDITIONAL -> n=2  would_auto_fail=TRUE
reversed        [C,C,PASS] + CONDITIONAL -> n=0  would_auto_fail=FALSE
```

Ordering is the load-bearing contract of **the one function feeding
`args.verdict_sequence`**, and a reversal silently DISARMS the escalation. None of
M1-M5 touched ordering.

**Fixed:** fixture is now `[PASS, CONDITIONAL, FAIL]` on a dedicated step id; a
**guard-on-the-guard** asserts the fixture is not equal to its own reverse, so it
cannot silently become palindromic again; and **M6** (the exact surviving mutant)
was added to the matrix and is now KILLED.

### C2 -- a number I carried instead of deriving

`run_id` "present on 33 of 35 rows" appeared in `verdict_ledger_write.py`,
`contract_86.85.md` **and** `research_brief_86.85.md`. It is **unreproducible**.
Population = every non-blank line of `handoff/verdict_ledger.jsonl` at
`d1c4a79d~1`; command
`git show d1c4a79d~1:handoff/verdict_ledger.jsonl | python3 -c "..."`:

```
total rows          : 35
run_id key present  : 35
run_id non-empty    : 35
run_id wf_-prefixed : 35
non-wf run_id values: []
```

**35 of 35 on every predicate.** Corrected at all three sites, with the population
rule and the command beside the number, and marked in place at the ORIGIN (the
brief) because the propagation path is the lesson: I took the figure from the brief
and never re-derived it, in the same session whose stated discipline is *"fixing
code does not fix prose"*.

### Lint -- `F401 shutil imported but unused`

`mutation_matrix_86_85.py:22`. Import removed;
`ruff --select F821,F401,F811` over both files is now **exit 0, "All checks
passed!"**.

### WARN -- `emit_sequence` laundered out-of-vocabulary tokens

It silently dropped any verdict outside the vocabulary. That **bypasses the
consumer's own fail-closed branch**: given the raw tokens `enforceEscalation`
returns `sequence_status="unparseable"`, `consecutive_conditionals=null`, whereas a
filtered list looks like a clean, confident, **shorter** sequence -- and shorter can
only ever UNDER-count a consecutive run, i.e. it fails OPEN. It was also internally
inconsistent with `read_rows`, which is deliberately loud for exactly the same
"would under-count" reason.

**Fixed:** it now raises `LedgerError` (exit 4). New self-test check, plus matrix
cell **M7**, killed.

### State after remediation

```
self-test                 : 13/13 ok, exit 0
mutation matrix           : 7 cells, 7 KILLED, 0 survived, 0 unscorable
                            control GREEN first; target sha256 identical
                            before/after (temp-copy mutants, zero repo writes)
ruff F821,F401,F811       : exit 0, "All checks passed!"
immutable command         : parses, exit 0
```

**A matrix caveat recorded rather than smoothed.** On the first re-run M2 scored
**UNSCORABLE (anchor matched 2x)**, because its anchor became a *substring* of the
more-indented copy M7 introduced -- `str.count` matches text, not lines. The matrix
refused to score it rather than reporting a kill it had not earned; the anchor was
made unique by including its preceding line. That behaviour is the matrix working,
and it is the same class as the C8 finding it was added to close.

---

## 6. Cycle-3 remediation -- the cycle-2 Q/A returned FAIL, and it was right again

**Verdict: FAIL** (`wf_879d28f2-9fc`). 6 of 8 criteria met; **C8 not met, C2 partial**,
plus a scope-honesty finding. Every blocker reproduced by Main before being fixed.

### C8 again -- and the lesson is that I fixed INSTANCES, not the CLASS

Cycle 1 found the ordering guard untested. I added M6 for it. Cycle 2 then found
**two more new guards with no coverage at all**:

- **QA-M6 / the fail-loud I/O guard** (`except OSError -> LedgerError(EXIT_IO)`).
  Reverted to `return row`, it **survives all 13 self-test checks and all 7 matrix
  cells**. Driven with a discriminating probe (append into a `0o500` directory):
  baseline `exit=4`, no file, loud stderr; mutant `exit=0`, **row printed to
  stdout, nothing on disk, empty stderr** -- a *silent writer*, which is the exact
  state this module's own docstring forbids and which criterion 6 calls
  unfalsifiable.
- **QA-M4 / `step_id` in the dedup key.** Dropped, the same `run_id` collides
  ACROSS steps and a legitimate second row is refused and **lost** -- an
  under-count, i.e. fails OPEN.

**The fix this time is class-level, not instance-level.** I enumerated every guard
from source -- all **9 `raise LedgerError` sites** plus every distinguishing branch
of `_dedup_key` -- and found the uncovered set was *larger than the two reported*:
`build_row`'s empty-`step_id` guard and **both CLI argument guards** were also
untested. All are now covered.

```
self-test        : 13 -> 20 checks (COUNTED)
mutation matrix  :  7 -> 12 cells, 12 KILLED, 0 survived, 0 unscorable
pytest           : 27 passed (new file, one test per cell) (COUNTED)
```

**And the matrix caught my own bad test, again.** M9 first scored **UNSCORABLE
(rc=1, no `SELF-TEST FAILED`)**: my new check called `append_row` unguarded, so the
mutant *crashed* the suite with a traceback instead of failing a check -- and a
crash is not evidence that a guard discriminated. Converting it to a caught
`try/except` turned it into a genuine KILL. That is the harness working exactly as
designed, and it is the same failure family as the C8 findings themselves.

### C2 -- counts that could not reproduce, and a correction that only annotated

Two distinct faults, both mine:

1. **Unanchored headline counts.** I stated 43 rows / 11 step_ids / FAIL 5 /
   29-of-43. Measured later: 44 / 12 / 6 / 30-of-44. **The drift is
   self-referential** -- 86.85 records its own verdicts into the very file it
   counts, so an unanchored total goes stale the moment the next cycle is graded.
   Every figure in §2 C2 now names the commit it was taken at (`d1c4a79d~1`
   pre-step, `d1c4a79d` as-shipped, working tree), which is the same discipline the
   C2 remediation figure already used and the headline block did not.
2. **`33/35` was ANNOTATED, not REPLACED.** I added a correction block at
   `research_brief_86.85.md:115` and left the wrong number standing at `:29`,
   `:126` and `:182`. That is precisely the *"a correction must REPLACE, not
   accompany"* failure I spent this morning fixing in 86.74 -- committed in the
   same session, in the opposite direction. All three sites now carry **35/35**
   with the population rule and command; the correction block remains only to name
   the error, which is history rather than a live claim.

### Scope honesty -- a promised artifact I silently substituted

`contract_86.85.md` §6.3 promised
`backend/tests/test_phase_86_85_verdict_ledger_write.py`. I shipped the checks as a
`--self-test` subcommand instead and **did not disclose the substitution**. The file
now exists (25 tests) and the two are deliberately not duplicates: `--self-test` is
the mutation-matrix target (one process, one exit code, dependency-free), the pytest
file is the regression suite that runs with the rest of the backend. Each test names
the matrix cell it mirrors so the two cannot silently diverge.

### Gates after remediation

```
pytest (new file)         : 27 passed, exit 0 (COUNTED)
self-test                 : 20/20 ok, exit 0 (COUNTED)
mutation matrix           : 12 cells, 12 KILLED, 0 survived, 0 unscorable
                            control GREEN first; target sha256 identical before/after
ruff F821,F401,F811       : exit 0, "All checks passed!"
immutable command         : parses, exit 0
```

---

# CYCLE 4 -- C8 ONLY (2026-08-15)

**Prior verdicts (as DATA):** `["FAIL", "FAIL", "FAIL"]`. All three were ONE
class -- a new guard shipped with no mutation cell. Each time the guard list was
written BY HAND and the Q/A found the one that was missed.

**This cycle does not write another hand-list.** It DERIVES guard-shaped
coverage instead. **The stronger claim originally written here -- "Completeness
is now DERIVED" -- is WITHDRAWN:** measured known-member recall against this
step's own three prior FAILs is **1 of 4** (see the corrected section below).
The gate catches guard-shaped omissions, not behavioural ones.

## Files changed

| file | change |
|---|---|
| `scripts/qa/verify_matrix_coverage_86_85.py` | **NEW** -- AST-enumerates the writer's guards and MEASURES per-cell coverage; no cell declares what it covers; plants a synthetic guard as a self-control and exits 2 if it cannot detect it |
| `scripts/qa/mutation_matrix_86_85.py` | +M13, +M14 (both derived, not spotted); calls the coverage checker and is **RED when a guard has no cell** |
| `scripts/qa/verdict_ledger_write.py` | +3 self-test checks pinning WHICH CLI refusal fires, +1 anti-vacuity check (20 -> 23 checks) |

## Result

| check | before | after |
|---|---|---|
| writer self-test | 20 checks | **23 checks**, PASSED |
| mutation matrix | 12 cells, 12 killed | **14 cells, 14 killed, 0 survived** |
| derived guard coverage | *did not exist* | **15 guards, 15 covered, 0 uncovered, 0 cell problems** |
| pytest (`-k '86_85 or ledger or verdict_ledger'`) | -- | **34 passed** |

## What the derivation found that a hand-list had not

**`main`'s CLI argument validation had NO cell aiming at it, while the matrix
reported 12/12 KILLED** -- a perfect score over an incomplete list. That is the
86.85 failure class in miniature, and it was surfaced by derivation rather than
by re-reading the file.

Adding the cells then exposed a second layer: **M14 SURVIVED**. The CLI guards
are defence-in-depth (`build_row` refuses anyway), so removing one changes only
the MESSAGE -- and the pre-existing self-test asserted only `exit == 3`, which
**both** paths return. The old checks were vacuous. Three checks now pin which
refusal fires; M14 is KILLED.

## Three defects the checker found in ITSELF, recorded not hidden

1. **False gaps** -- own-span-only matching missed guards nested inside a
   mutated branch (8 spurious gaps on the first run). Fixed with enclosing-`If`
   spans.
2. **An over-counted guard** -- `if args.emit_sequence:` counted because its
   body ends `return EXIT_OK`; a success return is not a refusal. Guards 16 ->
   15.
3. **Over-credited coverage, the dangerous direction.** Including `ast.Try` as
   an ancestor let `main`'s single wrapping `try:` credit any anchor with
   covering every guard inside it. **The tell was that dropping cell M14 left
   the gate GREEN** -- found only because the gate was tested for its ability to
   go RED, not merely observed passing. Ancestors are now `ast.If` only.

## Proof the gate is load-bearing

A drop-one-cell sweep (full table in `live_check_86.85.md` C8.5): removing any
of M1, M2, M3, M4, M7, M8, M10, M13, M14 turns the gate RED.

**CORRECTED after the cycle-4 Q/A, and re-measured by me rather than taken on
trust.** This paragraph previously said the other five cells (M5, M6, M9, M11,
M12) were "coverage-redundant -- another cell touches the same guard". **That is
false. No guard anywhere is covered by more than one cell, and those five cells
cover ZERO enumerated guards.** The gate stays green without them because their
targets are **invisible to the enumeration rule**, not because they are
duplicated. "Redundant" says the gate is complete; "invisible" says it is
blind. I wrote the reassuring one.

**Known-member recall = 1 of 4.** Against the three prior FAILs -- the checker's
own stated motivating class, not a set chosen after the fact -- dropping the
cell for each historical miss leaves the gate GREEN for ordering (M6),
step_id-in-key (M9) and cycle-fallback (M11+M12); only fail-loud I/O (M8) is
demanded. **The mechanism offered as the structural end of this failure class
would not have demanded 3 of the 4 guards whose omission caused it.** By this
project's own rule -- a scan that cannot find its own known members is a FAILED
gate -- **the claim "completeness is now DERIVED" is withdrawn.**

What the gate genuinely delivers is narrower and still real: it catches
**guard-shaped** omissions, which is how it found `main`'s uncovered CLI
validation while the matrix reported 12/12. It does not catch **behavioural**
omissions, which were 3 of the 4 historical misses. Extending the enumeration to
behavioural guards is queued as step **86.89**.

Also corrected: the behaviour list named "sequence filtering", for which no cell
exists, and omitted M5's real target. The five cells target ordering (M6),
dedup-key composition (M9), cycle fallback (M11, M12) and event/write-time
separation (M5). All five remain KILLED.

"Can this guard fail?" and "is any guard unmutated?" are different questions,
and conflating them is exactly how 12/12 coexisted with an uncovered guard --
that part stands.

## Scope and limits

- **C8 only.** C1-C7 unchanged from cycle 3, not re-litigated.
- A matrix licenses exactly "these 14 mutations were killed"; the checker adds
  only "no guard the enumeration can see is unmutated". Neither claims the guard
  SET is complete. The enumeration rule is written down in `live_check_86.85.md`
  C8.1 so its blind spots are auditable rather than implicit.
- **ZERO repo writes** during mutation: every mutated source is a temp copy or
  an in-memory string, and the writer's sha256 is printed before and after by
  both scripts. There is no restore step to get wrong.

---

## Cycle 5 GENERATE (2026-08-17, operator-attended session)

**Context.** The goal file parked 86.85 as "CONDITIONAL after 4 attempts; needs
your call". The operator delegated that call to this session. The verdict
sequence is FAIL, FAIL, FAIL, CONDITIONAL -- the FAILs reset the CONDITIONAL
counter, so no escalation rail constrains a cycle-5 verdict; the ordinary
changed-evidence respawn applies.

**Cycle-4's named remedy was verified ALREADY APPLIED** (both artifacts have
carried the corrections since 2026-08-15): the "coverage-redundant" claim is
replaced with the measured zero-coverage fact in `experiment_results` "Proof the
gate is load-bearing" and `live_check` C8.5; the 1-of-4 known-member recall is
stated beside the sweep with "completeness is now DERIVED" withdrawn; the
behaviour list is fixed (no "sequence filtering"; M5's event/write-time
separation named).

**Closed this cycle (the residue):**

1. **The sweep fence still carried the refuted word.** The five GREEN rows in
   live_check C8.5's drop-one table were labelled "(coverage-redundant)" INSIDE
   the fenced block, above their own refutation. Established by
   `grep -rn "coverage-redundant" scripts/qa/*.py` that no checker prints that
   phrase -- the labels were hand-authored, so relabelling them is correcting
   editorial text, not doctoring output. Relabelled to
   "(0 enumerated-guard coverage)" with a provenance note outside the fence.
2. **Stale heading annotated** ("12/12" is the cycle-3 count; matrix is 14/14
   since cycle 4) -- cycle-4 non-blocking note 1.
3. **Cycle-4's verdict is now IN the ledger.** The row was never written, so
   `--emit-sequence` under-reported the step's own history. Backfilled at the
   transcription seam with the true EVENT date (2026-08-15) and a write-time
   stamp of today -- the event/write-time separation is the writer's own design
   (M5's target). Verbatim:

   ```
   $ python3 scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.85
   ["FAIL", "FAIL", "FAIL", "CONDITIONAL"]
   $ python3 scripts/qa/verdict_ledger_write.py --self-test
   SELF-TEST PASSED   (exit 0)
   ```
4. **Cycle-4 non-blocking note 2: the CONCLUSION stands, and my cycle-5
   evidence for it was wrong** *(corrected on the cycle-5 Q/A's finding, which
   is transcribed in the critique)*. I wrote "Every `--emit-sequence` call in
   `_self_test` passes `--ledger`" -- refuted by my own quoted grep: `:492`
   `cli(["--emit-sequence"])` passes no `--ledger`. The real mechanism, settled
   by the evaluator BY EXECUTION (a monkeypatched read-trap on the real ledger
   observed ZERO reads): `main()` resolves the default path and then raises
   `--emit-sequence requires --step` BEFORE any read is reached, so "Touches no
   real file" holds. Cycle-4's note 2 ("it READS the real ledger via the module
   default") was therefore also factually wrong in its mechanism, which my
   cycle-5 text failed to say.

**No production code changed this cycle.** Changes: one data row appended to
`handoff/verdict_ledger.jsonl`, and the two handoff artifacts above.

---

## Cycle 6 GENERATE (2026-08-17): the QA-MUT-B class closed

Cycle-5 verdict transcribed verbatim in the critique. Its three findings, each
answered by the change it named:

**1. emit_sequence now orders by EVENT date** (`row["date"]`, ISO, stable by
file position within a date), so a backfilled older verdict lands where it
happened -- the shipped `--date` flag can no longer clear an escalation. An
undated row refuses LOUDLY (same doctrine as the out-of-vocabulary token: a row
that cannot be ordered must not be silently placed).

**2. Both ordering fixtures now carry DISTINCT event dates**, plus a dedicated
backfill fixture asserting `[C(11), C(12), PASS(10) backfilled]` emits
`[PASS, C, C]`, plus an undated-row loudness check in the self-test AND pytest.

**3. The matrix grew the QA-MUT-B class as permanent cells**: M15 collapses
event order back to file order (killed by the backfill fixture), M16 removes
the undated-row guard (killed by the loudness check), M6's anchor retargeted to
the new sort tail (reversal still killed). **A masked-fixture defect was found
and fixed in the process**: the out-of-vocabulary fixture's hand-written rows
carried no dates, so after the new guard landed, M7's mutant survived because a
DIFFERENT guard's refusal satisfied a check that only read the exit code -- the
rows now carry dates and the check pins the message text.

**4. The two claim defects corrected by replacement**: item 4 above now states
the true mechanism (and that cycle-4's note was wrong in mechanism too); the
'[WORKING TREE]' counts in both artifacts now carry their date, command, and
the reason the number moves.

Verbatim, post-change:

```
$ python3 scripts/qa/verdict_ledger_write.py --self-test
SELF-TEST PASSED                                                    (exit 0)
$ python -m pytest backend/tests -k '86_85 or ledger or verdict_ledger' -q
36 passed, 3514 deselected                                          (exit 0)
$ python3 scripts/qa/mutation_matrix_86_85.py        # 16 cells
16 x 'KILLED (rc=1)', 0 SURVIVED, 0 UNSCORABLE                      (exit 0)
$ python3 scripts/qa/verify_matrix_coverage_86_85.py
guards: 17   covered: 17   uncovered: 0   cell problems: 0          (exit 0)
$ python3 scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.85
["FAIL", "FAIL", "FAIL", "CONDITIONAL", "CONDITIONAL"]
```

Files changed: `scripts/qa/verdict_ledger_write.py` (emit_sequence + self-test
fixtures), `scripts/qa/mutation_matrix_86_85.py` (M6 anchor, M15, M16),
`backend/tests/test_phase_86_85_verdict_ledger_write.py` (dated fixture + two
new tests), the two artifacts, and this step's ledger rows. No gate weakened;
the sequence consumer (`enforceEscalation`) is untouched.

---

## Cycle 7 GENERATE (2026-08-17): the cycle-6 FAIL's findings closed in code

Cycle-6 verdict (FAIL) transcribed verbatim in the critique. Its three
findings, each closed by the change it named:

**1. QA-M-POS-const -- the same-date ordering hole.** The sort key now EXCLUDES
the verdict: `sorted(keyed, key=lambda t: (t[0], t[1]))`. Same-date rows are
the COMMON case (86.85's three FAILs share 2026-08-15), and the bare tuple sort
fell through to the verdict STRING there. With the key pinned: a
verdict-participates mutant (M17, `(t[0], t[2])`) and a within-date reversal
(M18, `(t[0], -t[1])`) both die against the new same-date fixtures, whose file
order (C, P, F) deliberately differs from alphabetical (C, F, P). Two mutants
became PROVABLY EQUIVALENT under the fix and are documented in the cell
comments rather than dropped: plain `sorted(keyed)` (pos still discriminates)
and pos-to-constant (Python's guaranteed-stable sort degrades it to file order
within a date -- the documented contract).

**2. QA-C6-1 -- the unvalidated ISO precondition.** `ISO_DATE_RE` now guards
BOTH seams: `build_row` refuses a non-ISO `--date` (`2026-8-10` -> LedgerError
EXIT_INVALID) and `emit_sequence` refuses to ORDER a stored non-ISO date
(range-shaped `2026-08-09/10` -> LedgerError, loud, never silently mis-placed).
Cells M19/M20 kill the removal of either seam. **Residual, stated:** 11 legacy
rows (36.17 x6, 86.20 x3, 86.17 x2) carry range-shaped dates; emitting THOSE
steps now errors loudly with repair instructions instead of mis-ordering --
the correction path for an append-only file (a new labelled row shares the
dedup key) is a small open design question recorded in the critique follow-up,
not silently decided here.

**3. QA-C6-3 -- the stale live_check, third recurrence.** The rule is now
written where the numbers live: every count in sections 6/C8 carries its
capture cycle, C8.6 is retitled as the cycle-4 capture, and C8.8 is the dated
CURRENT run. "A matrix count quoted without its cycle label or command is
wrong by default."

Verbatim, post-change: self-test PASSED (30 checks) · pytest 38 passed ·
matrix 20/20 KILLED, 0 unscorable · coverage 17/17 · ruff clean ·
emit-sequence [F, F, F, C, C, F].

Files changed: `scripts/qa/verdict_ledger_write.py`,
`scripts/qa/mutation_matrix_86_85.py`,
`backend/tests/test_phase_86_85_verdict_ledger_write.py`, the two artifacts.
The consumer (`enforceEscalation`) remains untouched.
