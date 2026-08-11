# Experiment results -- step 86.36

**Step**: `86.36` (phase-86, P2) | **Phase**: GENERATE | **Date**: 2026-08-11
**Concurrency**: peer session `pyfinagent-51` owns 86.25 / 86.34 / 86.29 / 86.38
and confirmed it will not touch this step. All development ran in SCRATCH sinks
because that peer has Q/A runs writing to the live `verdicts/` directory this
hour.

## 0. What the fix is, and why it is a NAME change

phase-86.31 requires the Q/A to write its record on its FIRST tool call so a
drop leaves evidence. The path was derived from the step id alone. Those two
facts compose into the defect: **the retry's opening act erases the prior
attempt's testimony.** The safety property and the hazard are the same write, so
moving the write later would break 86.31 -- the remedy has to be the NAME.

`verdict_wip_<sid>.md` -> `verdict_wip_<sid>__<YYYYMMDDTHHMMSSZ>.md`, plus
bounded retention. A COUNTER was rejected in the contract and the reason held up:
deriving `c<N>` needs read-then-write on the directory, and two concurrent
sessions (routine here, live today) race onto the same N. The writer already
knows its own spawn instant because it writes `WRITTEN:`.

## 1. Files changed

| File | Change |
|---|---|
| `scripts/qa/qa_wip.py` | `resolve_wip_path(..., run_stamp=None)`; new `list_wip_records()`, `prune_wip_records()`; `report()` selects the record for a given spawn and lists priors |
| `.claude/agents/qa.md` | write-first directive now names the stamped path, with the measured rationale |
| `scripts/qa/reproduce_wip_destruction_86_36.py` | **new** -- criterion 1 |
| `scripts/qa/verify_wip_retention_86_36.py` | **new** -- criteria 2, 3, 4 (23 assertions) |
| `scripts/qa/mutation_matrix_86_36.py` | **new** -- criterion 6 (5 cells) |

**`.claude/hooks/qa-write-guard.sh` is UNCHANGED** -- `git diff` empty. The
allowlist keys on the directory, so a new filename inside it was already
permitted. That is criterion 5 satisfied by construction rather than by argument,
and it keeps this step disjoint from 86.33.

**Backward compatibility is deliberate.** `resolve_wip_path()` with no
`run_stamp` still returns the legacy path, so the peer's in-flight records
(`verdict_wip_86.25.md`, `_86.34.md`, `_86.29.md`) still resolve today. Verified
live: all three report `COMPLETE`, `is_verdict=False`.

## 2. Criterion-by-criterion

| # | Criterion (abridged) | Evidence | Status |
|---|---|---|---|
| 1 | REPRODUCE first, with byte counts; state simulation vs real | driven: **4,386 -> 124 bytes**, same path, analysis gone. Corroborated by two REAL instances (86.31 cycle-4 6,239->stub; peer's 86.34 4,921->796) | MET |
| 2 | second cycle does not destroy the first; both readable with own stamps; `audit_memory.py` unchanged | 965B + 110B coexisting, distinct paths, own WRITTEN/COMPLETED; auditor output **byte-identical**, exit 1 both sides | MET |
| 3 | resolves per cycle; STALE + IDENTITY_UNKNOWN still fire; 86.31's assertions still pass | spawn1->cycle1, spawn2->cycle2, future spawn->STALE, junk->IDENTITY_UNKNOWN; **201/201** of `verify_qa_write_first_86_31.py` green (the count is non-deterministic -- see 5b note 1) | MET |
| 4 | no record readable as a verdict, asserted over every record | `verdict` key absent + `is_verdict is False` over 3 reports, with a `>=3` cardinality floor | MET |
| 5 | guard unchanged or strictly tightened | `git diff` on the hook is **EMPTY**; 86.31 checker green | MET |
| 6 | mutation-tested, both named cells | **5/5 KILLED** on named assertions, green control first, subject digest unchanged | MET |

## 3. A design bug my own assertion caught mid-build

The first selection walked records **newest-first** and took the first whose
`WRITTEN >= spawned_at`. For spawn 1 that matched cycle **2**'s record, because
06:40 is also >= 06:00 -- so recovering an older cycle silently handed back a
newer cycle's file. Caught by criterion 3's assertion *"spawn 1 resolves to
cycle 1's record"* on the first run, not by review. Fixed to oldest-first, and
it is now cell **M4** so it cannot come back silently.

## 4. THE MUTATION MATCHER WAS WRONG TWICE, IN OPPOSITE DIRECTIONS

Recorded because a 5/5 result is worth nothing if the scorer cannot discriminate,
and mine could not.

| attempt | rule | consequence |
|---|---|---|
| 1 | `name in output` | the checker prints every assertion name, **including passing ones** -- so any red scored as a named kill. **M5 was a FALSE KILL** and `MIS-ATTRIB` was unreachable. |
| 2 | `"FAIL " + name` | these names are **suffixes** of their line (`FAIL report(...) has NO 'verdict' key`), so a genuine kill was mis-scored MIS-ATTRIB. |
| 3 | name appears on a line starting `FAIL`/`FAILED:` | correct. |

**The only reason attempt 1 was ever discovered is that tightening it turned a
green into a red.** A matrix reporting 5/5 an hour ago was reporting one kill it
could not distinguish from any other failure.

**And the probe I wrote to prove the fix was itself broken** -- I copied the
matrix to `/tmp`, where `REPO = __file__.parents[2]` resolves outside the repo,
so it bailed at the control and I briefly read that as "the discriminator is
still vacuous". Re-run from inside `scripts/qa/`, the probe reports `MIS-ATTRIB`
as intended. **Third probe error in this step**, same family as the six on
2026-08-10.

## 5. Verbatim

```
$ source .venv/bin/activate
$ python scripts/qa/reproduce_wip_destruction_86_36.py
  bytes before / after      : 4386 -> 124   (LOST 4262)
  spawn 1's analysis still recoverable : False
OK -- destruction reproduced                                        exit=0

$ python scripts/qa/verify_wip_retention_86_36.py
ALL GREEN -- 23 passed, 0 failed                                    exit=0

$ python scripts/qa/mutation_matrix_86_36.py
  M1-REVERT-RUN-STAMP        KILLED   [named: the paths are DISTINCT]
  M2-RETAIN-ONLY-ONE         KILLED   [named: prune keeps exactly `keep` records]
  M3-LIST-ONLY-NEWEST        KILLED   [named: list_wip_records sees BOTH, newest first]
  M4-NEWEST-FIRST-SELECTION  KILLED   [named: spawn 1 resolves to cycle 1's record]
  M5-LEAK-A-VERDICT-KEY      KILLED   [named: has NO 'verdict' key]
tracked subject UNCHANGED: True                                     exit=0

$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_qa_write_first_86_31.py'
ALL GREEN -- 201 passed, 0 failed                                   exit=0   (immutable command)

$ git diff --stat -- .claude/hooks/qa-write-guard.sh
(empty)
```

## 5b. CYCLE 2 -- the Q/A's findings, and three corrections to THIS file

**Verdict was CONDITIONAL** (`wf_54b86608-cec`): all 6 criteria MET, two fixable
blockers. Both fixed; the evidence above is re-run at the post-fix tree.

**B1 (BLOCK) -- I CHANGED ONE OF THE TWO PLACES THAT INSTRUCT THE Q/A.** `qa.md`
got the stamped path; `.claude/workflows/qa-verdict.js` STEP 0b -- the **primary**
launch path -- did not, and kept injecting the destructive fixed filename plus the
premise *"the path is FIXED per step"* that this step falsifies. **Zero** stamp
references in that file. And the 86.31 checker's section [6] anchors all PASSED on
the stale text, so nothing guarded it. Fixed: STEP 0b rewritten, and the anchors
extended with `__<STAMP>` + `%Y%m%dT%H%M%SZ` needles for BOTH copies.
**Mutation-proven**: reverting STEP 0b to the fixed filename now drives exit 1 on
two named needles; restoring returns 201/201.

*While extending those anchors I broke them.* The section locator pinned the
literal `phase-86.31)` **including the closing paren**, so my revised heading
`phase-86.31, path revised by phase-86.36)` made the section unlocatable and
every check under it failed for a reason unrelated to its subject. Loosened to a
revision-tolerant prefix.

**B2 (WARN)** -- ruff F401, dead `re` import: a fossil of the two wrong regex
matchers. Removed; ruff now exits 0 on the git-derived 4-file scope.

**THREE CORRECTIONS TO THIS DOCUMENT, from the evaluator's notes:**

1. **"195 passed" does not reproduce, and the count is non-deterministic BY
   CONSTRUCTION.** Section [9] of the 86.31 checker emits one PASS per live WIP
   artifact. There were 7 when I captured it and 9 minutes later (the peer's
   record + mine), hence 197; the 4 new needles make it 201 now. A future reader
   should expect this number to move and should not treat a change as a
   regression.
2. **Two limitations I disclosed are now REFUTED IN THE STEP'S FAVOUR.** "The
   stamped path has never been written by a REAL Q/A / every record on disk is
   legacy-named" was true when written and **false two minutes later**: the
   peer's 86.29 Q/A wrote a stamped record at 06:59:22Z and this step's own Q/A
   at 06:59:57Z. The `qa.md` runtime read works, and the evaluator's own file is
   first-party proof.
3. **3 of the 5 "Verbatim" commands did not run as written** -- bare `python` is
   not on PATH (exit 127); only the immutable line carried the venv activation.
   Fixed above.

**One note I am NOT fixing, and why.** The evaluator's own extra mutant
`DEFAULT_KEEP 3 -> 1` **SURVIVES**: no assertion pins the default, because every
call passes `keep=` explicitly. It is doubly dead -- `prune_wip_records()` has no
production caller either -- so this confirms disclosed residual (3) below rather
than contradicting it. Pinning a default that nothing reads would be a guard
without a subject.

## 6. Scope, and what I cannot verify

- **The Q/A has NOT run.** No verdict is claimed.
- **The stamped path is not yet exercised by a REAL Q/A run.** `qa.md` now
  directs it, and the Workflow rail reads `qa.md` from disk at runtime so it is
  live for the next spawn -- but every record on disk today still uses the legacy
  name. The first real proof arrives with the next Q/A, and until then this is
  verified by simulation plus the live backward-compat check.
- **`prune_wip_records()` is never CALLED in production.** It is tested and
  available; wiring it to a caller is deliberately not done here, because the
  only safe caller is the writer and that is the agent, not this module. Retention
  is therefore currently unbounded in practice -- stated rather than implied.
- **A test-only seam was added** (`PYFIN_QA_WIP_OVERRIDE`) so the mutation matrix
  can drive the checker against a mutant without writing to the tracked file. It
  defaults to the real module. Same shape as 86.34's sweep-root seam.
- **Not touched**: `qa-write-guard.sh` (86.33), anything under `handoff/`, and
  the peer's steps.
