# live_check -- step 75.11.4 (2026-08-17; exits shown unpiped)

Every count below is stamped, because `handoff/current/` grew from 725 to 761
files **during this step** (three research gates were writing briefs). A count
here is a reading at a time, not a constant.

## 1. The immutable verification command (CYCLE 1 -- SUPERSEDED by section 16)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q
...................                                                      [100%]
19 passed in 0.61s
EXIT=0
```

**This 19-test green is exactly what the cycle-1 FAIL was about**: three
mutants passed it. Kept as the record of what was claimed. Cycle 2's "22
passed" was itself superseded for the same reason -- four more mutants
survived it. Section 16's "27 passed" was in turn superseded by cycle 5:
**the live reading is section 18e's: 31 passed, with the cycle-5 mutation
matrix (Q3/Q5/DRYMK, section 18c) also KILLED.**

## 2. Adjacent suites -- the two that PIN these scripts (regression check)

Both load the housekeeping scripts and would break on a careless refactor.
They constrained the design (see contract "two live constraints"), so they are
the regression set that matters:

```
$ .venv/bin/python -m pytest backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py \
      backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py -q
77 passed, 1 warning in 2.11s
```

## 3. Criterion 2 -- ALL FOUR branches in ONE run (verbatim, cycle 2)

The cycle-1 fixture exercised only two branches, which is why mutants M3 and
M4 survived. `_mixed_tree` now covers every keep/move path in a single run:

```
[warn] KEEP: research_brief_77.9.md -- names step 77.9, which the masterplan does not have
[99.2] moved: research_brief_99.2.md -> handoff/archive/phase-99.2/research_brief_99.2.md

Summary: done-moved=1 misc-moved=0 audit-moved=0 log-moved=0 root-kept=0 ambiguous=1
Kept in current/: protected=0 open-step=2 no-step-id=2 unknown-step=1
Unknown step id (left in current/ for manual review):
  - research_brief_77.9.md -- sid=77.9 (suffix) status=unknown
```

`misc-moved=0` is the load-bearing number: `misc/` IS the sweep. The
`no-step-id=2` and `unknown-step=1` counters are what M3 and M4 now break.
The separate referenced-path fixture (`[protected] KEEP: census_99.4.json`)
covers criteria 5 and 7.

## 4. Criterion 6 -- a bare invocation is now a DRY RUN, on the real tree
### (DATED CAPTURE, taken when `handoff/current/` held 761 files -- see section 18)

```
$ python scripts/housekeeping/backfill_handoff_archive.py
Summary: done-moved=436 misc-moved=0 audit-moved=1 log-moved=2 root-kept=1 ambiguous=8
Kept in current/: protected=19 open-step=148 no-step-id=58 unknown-step=8
DRY RUN -- nothing was moved. Re-run with --execute to apply.
$ ls handoff/current/ | wc -l
761          # unchanged
```

**Before this step the same bare command would have EXECUTED**, routing every
unrecognised name to `archive/misc/`. The `misc-moved=0` / `no-step-id=58`
pair is the fix: 58 files that the old code would have swept are now
explicitly kept, and 19 more are held back because a masterplan
`verification` criterion names them by literal path.

## 5. Criterion 9 + the regex fix -- BEFORE and AFTER on ONE identical tree

The "before" is a **reconstruction, not a historical run**: the pre-75.11.4
regex and branch logic re-applied to today's corpus, so both readings describe
the same input rather than two different days.

```
OLD code on TODAY's tree: no-step-id-prefix failures = 669
OLD code on TODAY's tree: done-step arm reached      = 0   <- structurally unreachable
OLD code on TODAY's tree: root-level failures        = 3
OLD total findings = 672
```

```
$ python scripts/housekeeping/verify_handoff_layout.py ; echo EXIT=$?
[info] 59 file(s) in current/ carry no step id (rolling files, day reports, incident notes) -- not an invariant violation; left in place
[warn] 9 file(s) name a step the masterplan does not have -- NOT archivable, left in place:
  - cc_rail_baseline_4000_1.md (sid=1, no such step)
  ...
handoff layout FAIL -- 455 invariant violation(s):
EXIT=1
```

Post-fix breakdown, which reconciles exactly:

| class | n |
|---|---|
| belongs to **done** step | 443 |
| belongs to **dropped** step | 6 |
| belongs to **superseded** step | 3 |
| root-level (2 logs + 1 audit stream) | 3 |
| **total failures** | **455** |
| unknown-step (reported, NOT a failure) | 9 |
| no step id (reported, NOT a failure) | 59 |

**The checker is still RED, and that is the correct answer.** 669 of the old
672 findings were one false class -- "has no step-id prefix" fired on every
artifact written in the convention the project actually uses, while the arm
that checks the real invariant could never run. Now the real invariant fires:
452 files in `current/` genuinely belong to closed steps and should be
archived. **Archiving them is step 86.105's job, not this one's** -- this step
ships the tool that can do it safely.

## 6. Criterion 12 -- remediation applied, and its idempotency

```
$ python scripts/housekeeping/quarantine_misattributed_archives.py --execute
Summary: dirs-scanned=845 mismatched-needing-marker=156 already-marked=0 contestable=43
$ find handoff/archive -name MISATTRIBUTION_NOTICE.md | wc -l
156
$ python scripts/housekeeping/quarantine_misattributed_archives.py --execute   # again
Summary: dirs-scanned=845 mismatched-needing-marker=0 already-marked=156 contestable=43
```

**A correction made during GENERATE, recorded rather than quietly fixed.** The
first version of this tool reported `contestable=1`, because it measured the
NARROW property (`confirm_mismatch`: does the dir's sid appear in a
*declaration*) while the notice text described the BROAD one (does it appear
at all, as in a batch contract "phase-10.5-batch (covers 10.5.0, ...)"). The
census module's own docstring warns about exactly this -- *"Keeping both
callable is what stops the report claiming the broad one while measuring the
narrow one"* -- and the tool fell into it anyway. Fixed to measure both; the
broad figure is **43**, matching the census.

## 7. Criteria 8/11/13 -- the hook driven HERMETICALLY, and why

`archive-handoff.sh:29` resolves everything from `CLAUDE_PROJECT_DIR`, so the
real hook runs against a scratch tree. **This deliberately replaces criterion
8's literal "flipping a scratch step"**, which would fire
`auto-commit-and-push.sh` -> `git add -A` and sweep a peer session's
uncommitted work into a commit named after a probe step. Recorded as a
deviation with its reason.

```
[archive-handoff] step 99.7 -> phase-99.7 (derived=2 copied=0 moved=0 skipped_rolling=0)
  handoff/archive/phase-99.7/contract.md      <- from contract_99.7.md
  handoff/archive/phase-99.7/live_check.md    <- from live_check_99.7.md   (criterion 11)
  handoff/archive/phase-99.7/PROVENANCE.md
```

`skipped_rolling=0` in that probe is the trap: `[ -f "$target/$f" ] && continue`
short-circuits the declaration guard whenever the derived branch already
supplied the name, so **that probe alone would credit the guard for work the
derived branch did.** The second probe omits the derived file so the guard is
actually reached, and includes one foreign AND one legitimate rolling file so
it must DISCRIMINATE rather than merely refuse:

```
[archive-handoff] step 99.8 -> phase-99.8 (derived=0 copied=1 moved=0 skipped_rolling=1)
PROVENANCE.md:
| -- | `handoff/current/contract.md` | SKIPPED: does not declare phase-99.8 |
| `experiment_results.md` | `handoff/current/experiment_results.md` | rolling file, declares this step |
```

## 8. Mutation matrix (control observed GREEN first; in-memory, no repo write)

Mutations are applied to SOURCE TEXT and exec'd into a throwaway module, so
the files on disk are never written. `test_the_scripts_on_disk_were_not_mutated_by_this_suite`
asserts no residue, and `_load_script` refuses a mutation that changed nothing
(a no-op mutant would otherwise score as a survivor for the wrong reason).

| cell | mutation | control | result |
|---|---|---|---|
| M1 | `if is_archivable(status)` -> `if True` | GREEN (pending file kept) | **KILLED** -- pending file moves |
| M2 | `if name in protected` -> `if False` | GREEN (referenced file kept) | **KILLED** -- referenced file moves |
| F1 | fixture: mark the pending step `done` | -- | **flips** as required (file must archive) |

**M2 initially SURVIVED, and the fixture was at fault, not the code.** The
referenced file was named `census_99.json` -> sid `99`, which was not a step in
the fixture, so the *unknown-step* branch held it in place even with the
reference guard removed -- two independent guards, one probe. Renamed to
`census_99.4.json` with `99.4` marked `done`, so removing the guard now
genuinely moves it. Recorded because a mutation cell that cannot discriminate
is worth less than no cell at all.

## 9. Disclosures

1. **The verifier's failure semantics changed for one class.** Files carrying
   no step id are now reported as `[info]` instead of counted as violations.
   This is not a loosened gate: `.claude/rules/research-gate.md` states the
   invariant as *"handoff/current/ contains NO files belonging to status=done
   steps"*, and a day report belongs to no step. The old message told the
   operator to move them to `archive/misc/` -- advice that, followed, is the
   sweep. Stated plainly so an evaluator can disagree with the judgement
   rather than have to discover it.
2. **The resolver has a measured precision limit**, pinned by
   `test_c1_resolver_precision_limit_is_pinned_not_papered_over`. Filenames
   using `_` where the convention uses `.` yield a WRONG id --
   `cc_rail_baseline_4000_1.md` -> `1`, `verdict_population_86_98_input.md` ->
   `86`. All land in the unknown-step branch and are kept, and the test asserts
   those ids are not real steps, so a mis-read cannot cause a move *today*. The
   test fails loudly if that ever stops being true.
3. **Root-file handling is UNCHANGED and still hazardous.** The dry run offers
   `prompt_leak_redteam_audit.jsonl -> handoff/audit/...-v4.jsonl`. 86.105
   measured that the destination already holds 2,035 bytes the source lacks and
   requires a MERGE, not a move. No criterion here covers root files, so the
   behaviour is untouched -- but with dry-run now the default, an accidental
   bare run can no longer perform it.
4. **`git mv` preference added** (`_git_mv`, mirroring `archive-handoff.sh:279`)
   with a `shutil.move` fallback for untracked files and non-repo test trees.
5. **No `.claude/hooks/**` file was modified**; no flag promoted; no `.env`
   written; no gate loosened; the backfill was never run with `--execute`
   against the live tree.
6. **Scope check on the commit** (`git status --short`, scoped):
   `M scripts/housekeeping/backfill_handoff_archive.py`,
   `M scripts/housekeeping/verify_handoff_layout.py`,
   `?? scripts/housekeeping/handoff_naming.py`,
   `?? scripts/housekeeping/quarantine_misattributed_archives.py`,
   `?? backend/tests/test_phase_75_11_4_backfill_status_aware.py`,
   plus the 156 additive `MISATTRIBUTION_NOTICE.md` files under
   `handoff/archive/`. Peer sessions hold uncommitted `backend/api/`
   and `frontend/src/` edits dated 2026-08-14 which are NOT this step's and
   must not ride its commit.

---

## 10. CYCLE 2 -- the repaired mutation matrix, run against the REAL suite

Cycle 1 returned FAIL: three mutants passed the entire suite. The cycle-1
matrix in section 8 above is **superseded** -- it recorded only the two cells
the author ran, and the evaluator's independent battery found three survivors
the author never tried. The table below REPLACES it.

The check that matters is not "is the mutant non-equivalent" (it was) but
"does the SUITE go red", so each mutant is applied to the real file and the
real suite is run:

```
sha256 baseline: 6c8e0e5ac49cb114ec6c73984294ac851aa15299449bf4477a561bf69fd9512a
CONTROL: exit=0  22 passed in 1.21s
M-INV (bare run executes):                          exit=1  3 failed, 19 passed  -> KILLED
M3 (no-step-id branch sweeps):                      exit=1  2 failed, 20 passed  -> KILLED
M4 (unknown-step branch sweeps, WARN still prints): exit=1  2 failed, 20 passed  -> KILLED
sha256 after all cells: 6c8e0e5ac49cb114ec6c73984294ac851aa15299449bf4477a561bf69fd9512a
BYTE-IDENTICAL RESTORE: True
SURVIVORS (this matrix): none
```

| cell | mutation | control | cycle 1 | cycle 2 |
|---|---|---|---|---|
| M1 | `if is_archivable(status)` -> `if True` | GREEN | KILLED | KILLED |
| M2 | `if name in protected` -> `if False` | GREEN | KILLED | KILLED |
| M-INV | `dry_run=not args.execute` -> `dry_run=args.execute` | GREEN | **SURVIVED** | **KILLED** |
| M3 | no-step-id branch keeps -> sweeps to `misc/` | GREEN | **SURVIVED** | **KILLED** |
| M4 | unknown-step branch keeps -> sweeps while WARN still prints | GREEN | **SURVIVED** | **KILLED** |
| F1 | fixture: mark the pending step `done` | -- | flips | flips |

Driver `scratchpad/prove_kills_75_11_4.py` restores in a `finally` and asserts
the sha256 in every cell, so a failed assertion cannot leave a mutant on disk.

## 11. CYCLE 2 -- the lint gate, and the vacuous green that preceded it

```
$ uvx ruff check --select F821,F401,F811 \
    scripts/housekeeping/backfill_handoff_archive.py \
    scripts/housekeeping/verify_handoff_layout.py \
    scripts/housekeeping/handoff_naming.py \
    scripts/housekeeping/quarantine_misattributed_archives.py \
    backend/tests/test_phase_75_11_4_backfill_status_aware.py
All checks passed!
RUFF_EXIT=0
```

**The first attempt was vacuous and is recorded rather than quietly redone.**
`ruff ... $FILES` under zsh passes the whole variable as ONE argument:

```
warning: Failed to lint scripts/...py scripts/...py ... : No such file or directory (os error 2)
All checks passed!
```

That "All checks passed" was a green over **zero files**. Positive control that
the gate can go red at all:

```
$ cp <the test file> lintprobe.py && printf 'import importlib.util\n' >> lintprobe.py
$ uvx ruff check --select F821,F401,F811 lintprobe.py
Found 1 error.
```

## 12. CYCLE 2 -- the two numeric claims, re-derived

```
$ # the function's OWN rule: re.compile(r"handoff/[A-Za-z0-9_./*-]+")
total references      = 577      (cycle 1 claimed 557)
distinct paths        = 386      (cycle 1 claimed 373)
into handoff/current/ = 415      (cycle 1 claimed 395)
distinct basenames    = 381
```

The cycle-1 triple came from the research brief and was never re-derived at
the seam that used it -- and it had been shipped into a PRODUCTION docstring,
where a number reads as measured. Corrected in
`scripts/housekeeping/backfill_handoff_archive.py`, `contract_75.11.4.md` and
`experiment_results_75.11.4.md`.

The "165 mismatched dirs" figure is corrected to **156**: 165 was a regex over
the checker's PROSE output, which admits the *declared* sids and the synthetic
controls while missing `phase-63.3-parked` and `phase-audit-2.10-4.14.20`.
`test_c12_prevention_holds_for_every_guard_created_directory` now derives its
set from `classify()` rather than from that prose.

## 13. CYCLE 2 -- immutable command and regression, re-run

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q
22 passed in 0.82s
EXIT=0
```

---

## 14. CYCLE 3 -- the full mutation matrix, all cells KILLED

Section 10's cycle-2 table is SUPERSEDED: the evaluator found four more
survivors it did not contain. Each mutant below is applied to the REAL file and
the REAL suite is run; restore happens in a `finally` with sha256 asserted. The
hook cell uses a mirrored copy, so `.claude/hooks/**` is never written.

```
sha256 baselines: backfill=6c8e0e5ac49c  verifier=f07a33170cfe
CONTROL: exit=0  27 passed
N5a live_check half dropped (crit 7):            exit=1  3 failed, 24 passed  -> KILLED
N5b command half dropped:                        exit=1  4 failed, 23 passed  -> KILLED
N14 verifier status arm disabled (crit 9):       exit=1  3 failed, 24 passed  -> KILLED
N15 verifier reverted to PREFIX regex (crit 9):  exit=1  3 failed, 24 passed  -> KILLED
restore verified: True     SURVIVORS (this matrix): none
```

```
CONTROL: exit=0  27 passed
M-INV (bare run executes):                       exit=1  3 failed, 24 passed  -> KILLED
M3 (no-step-id branch sweeps):                   exit=1  2 failed, 25 passed  -> KILLED
M4 (unknown-step sweeps, WARN still prints):     exit=1  2 failed, 25 passed  -> KILLED
BYTE-IDENTICAL RESTORE: True     SURVIVORS (this matrix): none
```

```
real hook sha256: 2278ca9910b0bd15  (never written)
CONTROL (faithful mirror): exit=0  2 passed, 25 deselected
MUTANT H5 (archived file created EMPTY instead of copied): exit=1  1 failed  -> KILLED
real hook unchanged: True
```

| cell | mutation | control | c1 | c2 | c3 |
|---|---|---|---|---|---|
| M1 | `if is_archivable(status)` -> `if True` | GREEN | KILLED | KILLED | KILLED |
| M2 | `if name in protected` -> `if False` | GREEN | KILLED | KILLED | KILLED |
| M-INV | `dry_run=not args.execute` -> `dry_run=args.execute` | GREEN | **SURVIVED** | KILLED | KILLED |
| M3 | no-step-id branch sweeps | GREEN | **SURVIVED** | KILLED | KILLED |
| M4 | unknown-step branch sweeps, WARN still prints | GREEN | **SURVIVED** | KILLED | KILLED |
| N5a | protected set drops the `live_check` half | GREEN | -- | **SURVIVED** | KILLED |
| N5b | protected set drops the `command` half | GREEN | -- | KILLED | KILLED |
| N14 | verifier's status arm disabled | GREEN | -- | **SURVIVED** | KILLED |
| N15 | verifier reverted to the dead PREFIX regex | GREEN | -- | **SURVIVED** | KILLED |
| H5 | hook archives an EMPTY file instead of copying | GREEN | -- | **SURVIVED** (WARN) | KILLED |

## 15. CYCLE 3 -- criterion 5's live exposure, which is why C1 mattered

The evaluator re-derived this from the masterplan with the function's own rule
and it is the reason the missing half was blocking rather than cosmetic:

- **381** protected basenames total
- **215** from `verification.command`, **174** from `verification.live_check`
- **166** protected **ONLY** by the `live_check` half
- of those, **18** exist in `handoff/current/` right now and **15** belong to
  CLOSED steps -- i.e. 15 real files whose only protection was a guard half
  that no test could fail

## 16. CYCLE 3 -- gates

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q
27 passed in 1.09s
EXIT=0
$ .venv/bin/python -m pytest <this suite> <36.7> <36.8> -q
104 passed, 1 warning in 3.55s
$ uvx ruff check --select F821,F401,F811 <5 files, separate args>
All checks passed!   EXIT=0
$ python scripts/housekeeping/verify_handoff_layout.py
handoff layout FAIL -- 455 invariant violation(s)     # unchanged, as intended
```

## 17. CYCLE 4 -- the census denominator, disclosed

Surfaced by the cycle-3 evaluator before its rail dropped, and verified here:

```
total phase-* dirs: 845
  agree           440    52.1%
  unclassified    222    26.3%
  mismatch        156    18.5%
  no_contract      27     3.2%

CLASSIFIABLE (agree+mismatch) = 596
mismatch share of CLASSIFIABLE = 156/596 = 26.2%
mismatch share of ALL dirs     = 156/845 = 18.5%
UNCLASSIFIABLE = 249 (29.5%) -- the census CANNOT speak to these
```

Cycles 1-3 quoted "156 of 845" without this. Both numbers are right under
their own denominator, but the 249 is the load-bearing omission: **nearly a
third of the archive tree is unassessed**, and the remediation makes no claim
about it. Stated now rather than left for a reader to discover.

No circularity in the remediation: `classify()` reads only `contract.md` /
`contract_*.md` and never `MISATTRIBUTION_NOTICE.md`, so writing 156 markers
cannot change a single verdict. (Independently confirmed by the cycle-3
evaluator by reading the classifier.)

## 18. CYCLE 5 -- the cycle-4 findings, closed

### 18a. The dry run no longer writes (PRODUCT defect, D1)

```
$ ls -d handoff/archive/phase-*/ | wc -l      # before a bare run
842
$ python scripts/housekeeping/backfill_handoff_archive.py >/dev/null
$ ls -d handoff/archive/phase-*/ | wc -l      # after
842
```

Before the fix, a bare run created a directory for every move it planned. The
three it left on the live tree are gone:

```
removed empty phase-80.5
removed empty phase-81.1
removed empty phase-82.23
```

They were `no_contract` in the census, so the archive denominator moves
**845 -> 842** and the `no_contract` bucket **27 -> 24**. This step's own dry
run had been inflating the number this step reports.

### 18b. The held-back count is 20, not 19

Re-derived READ-ONLY (importlib, no run, so the script's own side effects
cannot perturb the reading):

```
protected basenames in the masterplan: 381
[protected] KEEP on the LIVE tree    : 20
```

The 20th is this file. Section 4's block is a genuine capture from when
`current/` held 761 files and is now labelled as dated; the present-tense
prose in `experiment_results` is corrected to 20.

### 18c. Cycle-5 mutation matrix

```
sha256 baseline: 1b4f88f0df3495f7
CONTROL: exit=0  31 passed
Q3 ROLLING_KEEP_PREFIXES emptied (verdict gate dark):  exit=1  1 failed, 30 passed  -> KILLED
Q5 _safe_target clobbers prior evidence:               exit=1  1 failed, 30 passed  -> KILLED
DRYMK dry run mkdirs again:                            exit=1  1 failed, 30 passed  -> KILLED
restore verified: True     SURVIVORS (this matrix): none
```

Cycle-3 matrix re-run against the cycle-5 suite:

```
CONTROL: exit=0  31 passed
N5a exit=1 (3 failed) | N5b exit=1 (4 failed) | N14 exit=1 (3 failed) | N15 exit=1 (3 failed)
restore verified: True     SURVIVORS (this matrix): none
```

**Every "SURVIVORS" line in these artifacts is now scoped to its own matrix.**
No single run has tested all cells at once and the wording no longer implies
one has.

### 18d. Corrected census

```
total=842  agree=440  unclassified=222  mismatch=156  no_contract=24
judgeable=596   mismatch share of judgeable = 156/596 = 26.2%
```

### 18e. Immutable command, cycle 5

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q
31 passed in 1.00s
EXIT=0
```

### 18f. NOT this step's, and NOT to be committed with it

An uncommitted peer edit landed in `backend/services/autonomous_loop.py` at
**19:42:56Z**, during this step's cycle-4 evaluation -- a `_persist_analysis`
`summary` fix (full-path rows were persisting an empty summary because
`risk_assessment.reason` is a lite-path field). It is real and useful work and
it is **not this step's**. Together with `backend/api/sovereign_api.py` and six
`frontend/src/**` files it must be excluded from this step's commit, which is
why the commit uses an explicit pathspec and never `git add -A`.
