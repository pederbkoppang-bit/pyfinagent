# experiment_results -- step 75.11.4 (GENERATE, 2026-08-17)

Contract: `contract_75.11.4.md`. Research gate: **PASSED** (`wf_d4ad1550-ecf`;
18 sources read in full, 72 distinct URLs, audit-class dry after 8 rounds;
brief `research_brief_75.11.4.md`, 53,030 chars). Verbatim command evidence:
`live_check_75.11.4.md`.

## What was built

| File | Status | Purpose |
|---|---|---|
| `scripts/housekeeping/handoff_naming.py` | NEW | The ONE step-id recogniser both readers share (prefix + suffix + variant), plus `is_archivable` |
| `scripts/housekeeping/backfill_handoff_archive.py` | MODIFIED | Status gate, referenced-path refusal, `git mv` preference, dry-run default, honest summary |
| `scripts/housekeeping/verify_handoff_layout.py` | MODIFIED | Shares the resolver; no-step-id demoted from failure to info; unknown-step reported |
| `scripts/housekeeping/quarantine_misattributed_archives.py` | NEW | Criterion 12 remediation: additive markers, reusing the 86.29 classifier |
| `backend/tests/test_phase_75_11_4_backfill_status_aware.py` | NEW | **27** tests covering criteria 1-13 incl. **8** mutation cells (cycle 2 added M-INV/M3/M4; cycle 3 added N5a/N5b/N14/N15 and made H5 killable) |
| `handoff/archive/phase-*/MISATTRIBUTION_NOTICE.md` | NEW (156) | The remediation itself |

No file under `.claude/hooks/**` was modified. The backfill was never run with
`--execute` against the live tree.

## Criterion-by-criterion

**C1 -- resolver + status gate.** `resolve_step_id` accepts the legacy
`(?:phase-)?<sid>[-.]...` form AND the live `<base>_<sid>.(md|json)` form and
its `_<variant>` tail. Measured on the live tree: 609 suffix, 55 variant, 64
unresolved of 728 (stamped 18:39Z; the tree grew during the step). Archivable
iff status is `done`/`superseded`/`dropped`; unknown ids are KEPT with a WARN.
The old unknown branch MOVED the file to `misc/` while printing "left in
current/ for manual review" -- the summary contradicted the action; now it is
true.

**C2 -- one run, both directions.** `test_c2_pending_kept_and_done_moved_in_the_same_run`
asserts the pending step's `research_brief` and `live_check` stay while the
done sibling archives, and that `misc/` receives nothing.

**C3 -- mutation + fixture.** M1 (`if is_archivable(status)` -> `if True`)
KILLED, control green first. The fixture mutation (mark the pending step
`done`) flips the not-moved assertion, proving the fixture is load-bearing.

**C4 -- idempotency, asserted as CONVERGENCE.** The docstring's "idempotent"
meant only *non-destructive* (`_safe_target` mints `-v2`, `-v3`), which is how
`kill_switch_audit.jsonl` reached `-v4`. The test asserts the second run
reports `done-moved=0`, that `current/` is byte-identical, and that no `-v2`
was minted. The docstring now states the distinction instead of implying the
stronger property.

**C5 + C7 -- referenced-path refusal.** Protected set built from every
`verification.command`/`live_check` in the masterplan. **Re-derived in cycle 2
with the function's own regex: 577 references, 386 distinct paths, 415 into
`handoff/current/`, 381 distinct basenames.** (Cycle 1 shipped 557/373/395 in
both this file and the production docstring; that triple came from the research
brief and was never re-derived at the seam that used it. It does not reproduce,
the masterplan file was untouched during the work, and it is REPLACED here
rather than annotated.) On the live tree **20** files are held back --
re-derived in cycle 5 with the script's own `_masterplan_referenced_names()`
and `_is_rolling_keep()`; the 20th is this step's own `live_check_75.11.4.md`,
i.e. the artifact grew into its own protected set. Earlier cycles said 19,
which was true at capture time and stale by the time it was read. Matched by BASENAME deliberately -- over-protection is
the safe direction for a mover. M2 (`if name in protected` -> `if False`)
KILLED. The discrimination half matters: an unreferenced done-step sibling in
the same run still archives, so the guard is narrow rather than a blanket
refusal.

**Cycle-3 REPLACEMENT.** Cycles 1-2 guarded only the `command` half: the
fixture's `live_check` named a file it never created, so criterion 7's NAMED
mutation was absent and dropping the `live_check` key left the suite green.
Both halves are now separately fixtured and separately killed (N5a/N5b). See
CYCLE 3 -> C1.

**C6 -- safe by default.** A bare invocation is now a dry run; `--execute`
performs; `--dry-run` still accepted as a no-op alias so no documented
invocation becomes an error. **The inversion is confined to argument parsing**
because `test_phase_36_7_*` calls `mod.main(dry_run=False)` after monkeypatching
module globals -- changing the function contract would break the regression
test that protects the kill switch's state file.

**Cycle-2 REPLACEMENT of how this is asserted.** Cycle 1 claimed "a subprocess
drive showing `DRY RUN` and an unmoved file, and an AST check that `main` is
still invoked as `dry_run=not args.execute`". Both were inert and the evaluator
proved it: the "subprocess drive" imported the module (never running
`__main__`) and re-declared its own argparse, and the AST substring is True for
both `not args.execute` and `args.execute`. **The criterion is now asserted by
executing the real script as a subprocess with NO arguments** from a
self-contained repo copy, plus an explicit M-INV cell. See CYCLE 2 -> B1 below.

**C8 + C11 -- the hook archives suffix-named artifacts.** Driven hermetically
via `CLAUDE_PROJECT_DIR`; `contract_99.7.md` and `live_check_99.7.md` land as
`contract.md`/`live_check.md` with `PROVENANCE.md`. **Deviation, stated:** the
criterion says "flipping a scratch step", which would fire
`auto-commit-and-push.sh` -> `git add -A` over a peer session's uncommitted
work. The hermetic drive is deterministic and touches nothing outside tmp.

**C9 -- one convention.** Both readers now import the same definition, and a
test checks that every name the hook DERIVES (`${base}_${short_sid}.md` for its
five bases) resolves to the same sid. `AUDIT_KEEP_GLOBS` stays a literal in each
file because `test_phase_36_8_*` `ast.literal_eval`s it out of both.

**Cycle-3 REPLACEMENT.** Cycles 1-2 asserted this by SOURCE SCAN only -- no
test ever executed `verify_handoff_layout.main()`, so two mutants that restored
the pre-fix behaviour survived 22/22 while keeping the byte-string the scan
checked. The verifier is now RUN against a temp tree and both mutants are
killed (N14/N15). See CYCLE 3 -> C2.

**C10 + C12 -- census and remediation.** No new classifier: the phase-86.29
census already has a recall gate that REFUSES to print if it misses a known
positive, four synthetic controls, and a precision oracle. Re-measured:
**156 mismatched of 845 dirs, precision 0.9936, 43 contestable** (step text
said 129 of 747).

**THE DENOMINATOR'S BOUND, which cycles 1-3 did not state.** "156 of 845" is
not a rate over things the census can judge. Full classification of the live
tree:

| verdict | dirs | share |
|---|---|---|
| agree | 440 | 52.1% |
| **mismatch** | **156** | **18.5%** |
| unclassified | 222 | 26.3% |
| no_contract | 27 | 3.2% |

**249 dirs (29.5%) are UNCLASSIFIABLE -- the census cannot speak to them at
all.** Among the 596 it CAN judge, the mismatch rate is **156/596 = 26.2%**,
not 18.5%. Both figures are correct under their stated denominator; quoting
only 18.5% understates the rate among judgeable directories, and quoting either
without the 249 hides that nearly a third of the tree is unassessed. The
remediation covers the 156 the census can name; it makes no claim about the
249. Remediation is ADDITIVE -- 156 `MISATTRIBUTION_NOTICE.md`
files, nothing moved -- following OCFL's immutable-version-directory principle
and because 8 masterplan references point into `handoff/archive/`.

**C13 -- foreign rolling file refused, legitimate one admitted.** The probe
deliberately omits the derived file, because with it present
`[ -f "$target/$f" ] && continue` short-circuits the guard and the cell would
credit it for the derived branch's work.

## The framing correction that made the fix smaller

The step text says "hook = PREFIX-dash vs backfill = SUFFIX-underscore".
Measured: `archive-handoff.sh` carries BOTH -- a live SUFFIX branch (`:226-242`)
and a legacy PREFIX glob (`:276`) that its own comment records as matching zero
files. The two READERS carried the same byte-identical PREFIX-only regex. So
this was one live writer convention against two prefix-only readers, fixable
entirely inside `scripts/housekeeping/**` -- which is what kept the work inside
the step's stated boundary and left `.claude/hooks/**` untouched.

## Prevention is already closed; this step is remediation

`.claude/hooks/archive-handoff.sh` writes `PROVENANCE.md` unconditionally, so
its presence marks a directory built by the guarded (86.29) hook. Measured:
**24 such directories, and none is in the mismatch set**, while all 156
mismatched dirs lack the marker. `test_c12_prevention_holds_for_every_guard_created_directory`
pins it. **Stated with its bound: 24 is the entire post-guard population, so
this is a small-sample result, not a universal proof.**

One apparent counterexample resolves the other way and is worth recording:
`phase-86.24` is mismatched and its directory was first committed by `630fa95b`
at `2026-08-11 08:42:07`, **4m54s after** the guard commit `974297ce`. That is
not a guard failure -- the commit time is when `git add -A` swept the directory
in, not when the hook created it. It contains no `PROVENANCE.md` and holds the
four bare rolling files, which the guarded hook cannot produce.

## Numbers that did NOT reproduce, and the rule that explains each

| Step text | Measured | Why they differ |
|---|---|---|
| 20 suffix files in `current/` | 579 `.md` / 608 with `.json` | written 2026-07-25; the tree grew ~30x because the hook COPIES and `current/` never drains |
| 13,198 prefix archived / 428 in misc | 507 / 488 | the 13,198 counted ALL files under `phase-*`, not prefix-convention ones |
| 129 of 747 misattributed | 156 of 845 | the 129 was a first-line heuristic the step text itself flagged as needing refinement |
| checker findings "six" (86.105) | 672 reconstructed old / 455 new | the "six" was a sample of a 669-member class recorded as a census |

The 579-vs-608 pair is not a disagreement: 579 is `.md` only, 608 adds the 29
`_<sid>.json` files. Both are correct under their stated rule.

## Open, and explicitly NOT claimed

1. **The layout checker is still RED (455).** Correctly so: 452 files in
   `current/` genuinely belong to closed steps. Archiving them is **86.105's**
   GENERATE, which this step unblocks by making the backfill safe to run.
2. **Root-file handling unchanged.** The dry run still offers to move
   `prompt_leak_redteam_audit.jsonl` to a `-v4` name; 86.105 measured that this
   needs a MERGE (the destination holds 2,035 bytes the source lacks). No
   criterion here covers root files.
3. **The resolver's precision limit is real**, pinned by test, harmless today
   only because the mis-read ids are not live steps.
4. **86.29 remains `status: pending`** even though its hook revision is live.
   Not this step's to close.

---

# CYCLE 2 -- what the cycle-1 FAIL found, and what changed

Cycle 1 returned **FAIL** (`wf_99ff4ce9-d0e`), transcribed verbatim in
`evaluator_critique_75.11.4.md`. The evaluator's own summary of the shape:
*"THE SHIPPED SCRIPT BEHAVES CORRECTLY in all three mutant cases -- verified by
hermetic control runs -- so the defects are in the GUARDS and the CLAIMS, not
the product."* Every blocker below is therefore a defect in **this step's own
verification**, which is exactly what criteria 1/3/6/7 require to be sound.

## B1 -- criterion 6 had SOLE-COVERAGE VACUITY (the blocking one)

Mutant **M-INV**: `main(dry_run=not args.execute)` -> `main(dry_run=args.execute)`.
A bare run then EXECUTES -- reinstating the precise defect this step exists to
remove -- and the whole 19-test suite still passed.

Both halves of the only covering test were inert:

- **The "subprocess drive" re-implemented the thing under test.** It did
  `import backfill_handoff_archive as m` (which never runs `__main__`) and then
  declared its OWN argparse calling `m.main(dry_run=not a.execute)`. The `not`
  under test lives in the SCRIPT's `__main__`; the harness carried a private
  copy and asserted its own correctness.
- **The AST assert could not see the mutation.** `"execute" in ast.dump(kw["dry_run"])`
  is True for BOTH `not args.execute` (`UnaryOp(op=Not(), operand=Attribute(...,
  attr='execute'))`) and `args.execute` (`Attribute(..., attr='execute')`). It
  kills only a full revert to `args.dry_run` -- the safe shape, not the
  dangerous one.

**Fix:** `_fake_repo()` builds a self-contained repo copy (the script derives
`REPO = Path(__file__).resolve().parents[2]`, so a copy at
`<tmp>/scripts/housekeeping/` makes `<tmp>` the repo) and `_run_script()`
executes it as a **subprocess with no arguments** -- the real `__main__`, the
real default. Added `test_c6_m_inv_inverting_the_default_is_caught` as an
explicit cell.

## B2 -- criteria 1's keep-branches were UNFIXTURED (M3, M4 survived)

No fixture put an unresolvable-name or unknown-sid file into `current/`, so
neither keep-branch had a guard that could fail. **Cycle 1 caused this itself**:
renaming `census_99.json` -> `census_99.4.json` (a correct fix for M2's
confound) removed the only place the unknown branch was reached, and nothing
replaced it.

**Fix:** `_mixed_tree` now covers **all four branches in one run** -- open step,
closed step, no-step-id (2 files), unknown sid -- and `test_c2` asserts the
files AND the `[warn] KEEP` line AND `no-step-id=2 unknown-step=1`. Added
`test_c1_m3_...` and `test_c1_m4_...`. M4 is deliberately the nasty shape: it
keeps the WARN print and moves the file anyway, so a guard checking only the log
line would pass.

## B3 -- lint gate red on this step's own file

`F401 importlib.util imported but unused` at the test file's line 26. Removed.

## B4 -- two numeric claims did not reproduce

- **557 / 373 / 395** (shipped into the PRODUCTION docstring of
  `_masterplan_referenced_names`) re-derives as **577 / 386 / 415** under the
  function's own regex. The figure came from the research brief and was never
  re-derived at the seam that used it. Corrected in the docstring, the contract
  and this file. *A number in a docstring is read as measured.*
- **"all 165 mismatched dirs"** three paragraphs after **"156 mismatched of
  845"**. 165 was `set(re.findall(r"phase-[0-9]+(?:\.[0-9]+)*", --list-wrong
  stdout))` over the checker's PROSE: it admits 11 tokens that are not
  mismatched dirs (the *declared* sids, plus the synthetic controls) and MISSES
  two real ones the pattern cannot express (`phase-63.3-parked`,
  `phase-audit-2.10-4.14.20`). Corrected to 156, and
  `test_c12_prevention_holds_for_every_guard_created_directory` now builds its
  set from `classify()` directly, like its sibling already did.

## The kills, proven against the REAL suite

Cycle 1's defect was not that the mutants were equivalent -- it was that the
SUITE passed with them live. So the check is: apply each mutant to the real
file, run the real suite, require a non-zero exit.

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

Driver: `scratchpad/prove_kills_75_11_4.py` -- restores in a `finally` and
asserts the sha256 in every cell, so a failed assertion cannot leave a mutant
on disk.

## Lint, with a positive control so the green is not vacuous

```
$ uvx ruff check --select F821,F401,F811 <the 5 files, passed as SEPARATE args>
All checks passed!   EXIT=0
```

**The first attempt at this check was itself vacuous** and is recorded rather
than quietly redone: `ruff ... $FILES` under zsh passes the whole variable as
ONE argument, so ruff printed `Failed to lint <all five joined>: No such file
or directory` and then `All checks passed!` -- a green over zero files. Positive
control: appending `import importlib.util` to a copy yields `Found 1 error.`, so
the gate can go red.

## Immutable command after the repair

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q
22 passed in 0.82s
EXIT=0
```

---

# CYCLE 3 -- the cycle-2 FAIL, and what changed

Cycle 2 returned **FAIL** (`wf_b3b0a007-8c4`), transcribed verbatim in
`evaluator_critique_75.11.4.md`. It **confirmed every cycle-1 fix landed**
(M-INV/M3/M4 killed in the evaluator's own harness, F401 gone, 577/386/415/381
and 156/0.9936/43 reproducing exactly) and then found DIFFERENT guard defects.
Same shape as cycle 1, in the evaluator's words: *"THE SHIPPED PRODUCT IS
CORRECT IN ALL THREE CASES ... so these are guard defects."*

## C1 -- criterion 7's NAMED mutation was absent (blocking)

The criterion says: *"point a step's verification.live_check at a file the
classifier would otherwise sweep -> the protection test goes red when the guard
is removed."* Dropping `live_check` from `_masterplan_referenced_names`'s key
tuple left the whole 22-test suite green.

**Root cause was the fixture, again.** `_referenced_tree` protected its file
through `verification.command`, and the `live_check` it also wrote named
`handoff/current/live_check_99.3.md` -- **a file the fixture never created**.
So the live_check branch of the protected set could not change any assertion.

**Live exposure the evaluator measured, which is why this is not academic:** of
381 protected basenames, **166 are protected ONLY by the live_check half**; 18
of those exist in `handoff/current/` right now and **15 belong to CLOSED
steps** -- 15 real files whose sole protection was a guard half no test could
fail.

**Fix:** `_referenced_tree` now creates `live_check_99.5.md` (DONE step 99.5)
referenced ONLY by step 99.3's `verification.live_check`, and `test_c5` asserts
both halves. Added `test_c7_m5_...` (drop the live_check half) and
`test_c7_m6_...` (drop the command half); each also asserts the OTHER half
still protects its own file, so a cell cannot pass because everything moved.

## C2 -- criterion 9: NO test ever executed the verifier (blocking)

`verify_handoff_layout.py`'s entire coverage was a source scan
(`assert "from handoff_naming import" in src`) plus an AST literal check.
`_load_script` was only ever called with `"backfill_handoff_archive"`. Two
mutants therefore survived 22/22 while restoring the pre-fix behaviour exactly:

- **N14** `elif is_archivable(status)` -> `elif False`
- **N15** keep the `from handoff_naming import` LINE the guard byte-checked,
  but rebind `resolve_step_id` to the retired PREFIX-only matcher

Live differential: shipped verifier -> `handoff layout FAIL -- 455`; either
mutant -> `FAIL -- 3` (only the root-level findings survive).

**Fix:** `_verifier_tree` + `_point_verifier` now RUN `verify_handoff_layout.main()`
against a temp tree and assert the done-step arm fires on a SUFFIX-named
artifact, that an open step is not reported, and that the no-step-id class is
info. Added `test_c9_n14_...` and `test_c9_n15_...`.

## C3 -- the hook test asserted existence, not content (was WARN)

A mutant creating the archived file EMPTY (`: >` instead of `cp`) satisfied
every existence assert and the negative `"ANOTHER STEP" not in contract.md` --
a zero-byte file trivially contains nothing. `test_c8_c11_...` now asserts
**content equality** with the source.

## The cycle-3 mutation matrix -- 8 cells, all KILLED, controls green first

Each mutant is applied to the REAL file and the REAL suite is run; restore in a
`finally` with sha256 asserted. The hook cells use a mirrored copy so
`.claude/hooks/**` is never written.

```
sha256 baselines: backfill=6c8e0e5ac49c  verifier=f07a33170cfe
CONTROL: exit=0  27 passed
N5a live_check half dropped (crit 7):            exit=1  3 failed, 24 passed  -> KILLED
N5b command half dropped:                        exit=1  4 failed, 23 passed  -> KILLED
N14 verifier status arm disabled (crit 9):       exit=1  3 failed, 24 passed  -> KILLED
N15 verifier reverted to PREFIX regex (crit 9):  exit=1  3 failed, 24 passed  -> KILLED
restore verified: True     SURVIVORS (this matrix): none

CONTROL: exit=0  27 passed
M-INV (bare run executes):                       exit=1  3 failed, 24 passed  -> KILLED
M3 (no-step-id branch sweeps):                   exit=1  2 failed, 25 passed  -> KILLED
M4 (unknown-step sweeps, WARN still prints):     exit=1  2 failed, 25 passed  -> KILLED
BYTE-IDENTICAL RESTORE: True     SURVIVORS (this matrix): none

real hook sha256: 2278ca9910b0bd15  (never written)
CONTROL (faithful mirror): exit=0  2 passed, 25 deselected
MUTANT H5 (archived file empty): exit=1  1 failed, 1 passed  -> KILLED
real hook unchanged: True
```

## Cycle-3 gates

```
$ .venv/bin/python -m pytest <this suite> <36.7> <36.8> -q
104 passed, 1 warning in 3.55s
$ uvx ruff check --select F821,F401,F811 <the 5 files as separate args>
All checks passed!   EXIT=0
$ python scripts/housekeeping/verify_handoff_layout.py
handoff layout FAIL -- 455 invariant violation(s)      # unchanged, as intended
```

---

# CYCLE 5 -- the cycle-4 CONDITIONAL, and what changed

Cycle 4 returned **CONDITIONAL** (`wf_51313030-ddd`), verbatim in
`evaluator_critique_75.11.4.md`. The evaluator states **all 13 immutable
criteria are MET** and that it made at least one guard fail itself for every
one of them. Every finding was WARN-level. All of them are now closed.

## D1 -- a PRODUCT defect: the dry run wrote to disk

`_move` ran `dest_dir.mkdir(parents=True, exist_ok=True)` **before** its
`if dry_run: return`. So a bare invocation -- the thing criterion 6 is about --
created every destination directory it merely planned to use. Measured
consequence: three empty dirs on the live tree (`phase-80.5`, `phase-81.1`,
`phase-82.23`, mtime 18:42:23Z), each of which then classified as
`no_contract` and **inflated this step's own census denominator**.

- **Fixed**: the mkdir now runs after the dry-run return.
- **Cleaned up**: the three empty dirs removed (`rmdir`, empty-checked first).
  The archive denominator is **842**, not 845.
- **Guarded**: `test_c6_a_dry_run_creates_no_directories` snapshots the archive
  listing around a bare run and requires it unchanged, plus asserts
  `would-move` appears so the check is not vacuous. Mutant **DRYMK** (restore
  the old ordering) -> **KILLED**.

This was flagged in cycle 2 and dropped without disposition. That is the
finding behind the evaluator's `criteria-erosion` WARN, and it was the correct
call: the blocking items were fixed each cycle while the cheap ones were not.

## D2 -- two ADJACENT safety properties had no guard

Both were the evaluator's own cells and both SURVIVED cycle 4. Neither is one
of this step's 13 criteria, but both are properties this step's diff either
relies on or claims:

- **Q3** `ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)` -> `()`. Archives a
  done step's `evaluator_critique_<sid>.json`, restoring the phase-81.0 defect
  the file's own comment records as having *"left the verdict gate dark for 13
  consecutive step closes"*. **54 such files live in `handoff/current/` now.**
  Guarded by `test_rolling_keep_prefixes_protects_per_step_verdict_jsons`,
  which also asserts the done-step sibling still archives. -> **KILLED**.
- **Q5** `_safe_target` -> `return dest`. Clobbers prior archived evidence,
  while **this step's own diff ADDS the docstring claim "prior evidence is
  never clobbered"**. Guarded by
  `test_safe_target_never_clobbers_prior_archived_evidence` (asserts the prior
  file still reads `PRIOR EVIDENCE` and that a `-v2` was minted). -> **KILLED**.

## D3 -- the quarantine script had zero direct coverage

174 production lines exercised only through their output. Added
`test_quarantine_tool_is_dry_run_by_default_and_idempotent`: default is a dry
run that writes nothing; `--execute` writes exactly one marker into the
mismatched dir and **none** into the dir that agrees; the notice names both
step ids; a second `--execute` reports `mismatched-needing-marker=0
already-marked=1`.

Writing it surfaced a good property worth recording: with `scripts/qa/` absent
the tool **refuses and exits non-zero** (*"This tool has no classifier of its
own by design; fix the import rather than adding one"*) rather than silently
degrading.

## D4 -- a tautological assertion, removed rather than left to look like cover

`assert "misc-moved=0" in out` could not fail: `misc_moved` is initialised to
0 and, since the sweep branch was deleted, nothing increments it. It is present
under every mutant. Removed; the real property
(`assert not list(archive/misc.iterdir())`) is what kills M3/M4, and the
production constant now carries a comment saying so.

## D5 -- "19 files are held back" was 20

Re-derived READ-ONLY with the script's OWN `_masterplan_referenced_names()`
and `_is_rolling_keep()` over the live tree:

```
protected basenames in the masterplan: 381
[protected] KEEP on the LIVE tree    : 20
```

The 20th is **this step's own `live_check_75.11.4.md`** -- the artifact grew
into its own protected set. The section-4 block in `live_check` is a genuine
dated capture (it self-reports 761 files in `current/`), so it is stale rather
than wrong; it is now annotated as dated, and the present-tense prose is
corrected to 20.

## D6 -- "SURVIVORS (this matrix): none" was unscoped

The string appears under several different matrices, each covering only its
own cells. Every occurrence is now scoped to the matrix that produced it
("SURVIVORS (this matrix): none"). No single run has ever tested all cells at
once, and the artifacts should not read as though one had.

## Cycle-5 matrix -- the three new cells

```
sha256 baseline: 1b4f88f0df3495f7
CONTROL: exit=0  31 passed
Q3 ROLLING_KEEP_PREFIXES emptied:            exit=1  1 failed, 30 passed  -> KILLED
Q5 _safe_target clobbers prior evidence:     exit=1  1 failed, 30 passed  -> KILLED
DRYMK dry run mkdirs again:                  exit=1  1 failed, 30 passed  -> KILLED
restore verified: True     SURVIVORS (this matrix): none
```

The cycle-3 matrix re-run against the cycle-5 suite, unchanged:

```
CONTROL: exit=0  31 passed
N5a exit=1 (3 failed) | N5b exit=1 (4 failed) | N14 exit=1 (3 failed) | N15 exit=1 (3 failed)
restore verified: True     SURVIVORS (this matrix): none
```

## Corrected census (denominator now free of this step's own artefacts)

```
total=842  agree=440  unclassified=222  mismatch=156  no_contract=24
judgeable=596   mismatch share of judgeable = 156/596 = 26.2%
```

156 markers, unchanged. The three removed dirs were `no_contract`, which is
why that bucket drops 27 -> 24 and the total drops 845 -> 842.
