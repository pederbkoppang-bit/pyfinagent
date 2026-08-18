STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 75.11.4
WRITTEN: 2026-08-17T18:51:13Z

# Q/A write-first record -- step 75.11.4 (handoff archive status-aware backfill)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command, git scope, lint, syntax
C. LLM judgment vs 13 immutable criteria + mutation matrix

## Findings (appended as established)

### Prior-attempt evidence
- `qa_wip.py 75.11.4 --spawned-at 2026-08-17T18:51:13Z`: attempt_number=1,
  prior_attempts=0, source_present=TRUE, attempt_number_status=ok,
  records_retained=1 (gauge, = my own WIP), prior_records=[].
- `verdict_history_86_21.py --step 75.11.4 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). Ledger has NO rows for this step; ledger is hand-written so
  absence is weak evidence. Cross-check: prior_attempts(0) > ledger count(0)? NO
  (0 == 0) -> ledger NOT provably stale for this step. sequence: EMPTY per ledger,
  and prior_attempts=0 agrees. Treating as first graded attempt.
- masterplan: status=pending, retry_count=0, max_retries=3 -> certified_fallback=false.

### A. Harness compliance (5 items)
1. research-gate-before-contract: research_brief_75.11.4.md EXISTS (53,037 bytes,
   mtime 20:35:04) < contract (20:38:30). Gate claims: 18 sources in full, 72 URLs,
   audit-class dry after 8 rounds (wf_d4ad1550-ecf). TO VERIFY.
2. contract-before-generate: contract 20:38:30 < handoff_naming.py 20:39:28 <
   verify_handoff_layout 20:40:33 < backfill 20:42:19 < quarantine 20:47:09 <
   test 20:47:34 < live_check 20:49:44 < experiment_results 20:50:30. ORDER OK.
3. experiment_results present: YES (8,573 bytes).
4. log-last: `grep -nE "phase=75\.11\.4" handoff/harness_log.md` -> 0 matches.
   masterplan status still `pending`. OK -- not logged/flipped yet.
5. no-verdict-shopping: prior_attempts=0, no prior evaluator_critique for this step.
   N/A (first attempt).

### B. Deterministic
- IMMUTABLE: `.venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`
  -> 19 passed in 0.62s, **EXIT=0** (captured bare, no pipe).
- 156 MISATTRIBUTION_NOTICE.md untracked files confirmed by
  `git ls-files --others --exclude-standard | grep -c MISATTRIBUTION_NOTICE` -> 156.
  Matches the claim exactly.
- git diff --stat HEAD: step-scoped production changes are exactly
  scripts/housekeeping/backfill_handoff_archive.py and
  scripts/housekeeping/verify_handoff_layout.py. Peer-session uncommitted work
  (backend/api/sovereign_api.py, 5 frontend/src files, perf_results.tsv,
  .claude/.archive-baseline.json) present and NOT this step's -- disclosed in
  live_check section 9.6. No .claude/hooks/** change: CONFIRMED by diff.

- **LINT GATE (qa.md 1a) RED.** Scope DERIVED (git diff --name-only HEAD '*.py'
  UNION git ls-files --others --exclude-standard '*.py'; 6 files, non-empty).
  `uvx ruff check --select F821,F401,F811` bare exit=1 on this step's OWN new file:
    F401 [*] `importlib.util` imported but unused
      --> backend/tests/test_phase_75_11_4_backfill_status_aware.py:26:8
  Confirmed genuinely dead: `grep -n importlib <testfile>` returns ONLY line 26.
  Step-scoped rerun (5 step files only) also exits 1.

### *** FINDING 1 -- SURVIVING MUTANT ON CRITERION 6 (the step's core safety property) ***

Criterion 6: "Default invocation is a DRY-RUN printing the plan; executing
requires an explicit flag."

MUTANT M-INV: `raise SystemExit(main(dry_run=not args.execute))`
           -> `raise SystemExit(main(dry_run=args.execute))`
(drop the `not` -- a bare run then EXECUTES, and `--execute` dry-runs.)

BEHAVIOURAL DIFFERENTIAL (hermetic fake repo in scratchpad; REPO is derived from
`Path(__file__).resolve().parents[2]` so a copy at
<scratch>/m6/<name>/scripts/housekeeping/ is fully isolated -- the real repo was
NEVER written):
  CONTROL, bare invocation: rc=0, current/ = [research_brief_99.1.md,
    research_brief_99.2.md] (BOTH kept), "DRY RUN" in stdout = True
  MUTANT, bare invocation:  rc=0, current/ = [research_brief_99.1.md]
    -> research_brief_99.2.md WAS MOVED to handoff/archive/phase-99.2/,
    "DRY RUN" in stdout = False
So the mutant is behaviour-changing and NOT equivalent: it reinstates exactly the
"a bare run moves files" defect this step exists to remove.

SUITE RESULT: **SURVIVED.** Ran the REAL suite with `pytest.main(..., plugins=[...])`
repointing the test module's `HOUSEKEEPING` constant at the mutated copy (no repo
write, no plugin file):
  - `-k c6`  : CONTROL 2 passed; MUTANT 2 passed
  - full file: MUTANT **19 passed** in 0.63s
No test in the suite can fail on this defect.

WHY IT SURVIVES (both halves of test_c6_bare_invocation_is_a_dry_run are inert
for this mutation):
 (a) RE-IMPLEMENTED DRIVE (vacuity shape #7). The "subprocess drive" writes its
     OWN harness -- `import backfill_handoff_archive as m` (which does NOT run
     `if __name__ == "__main__"`), then re-declares argparse and calls
     `m.main(dry_run=not a.execute)`. The `not` under test lives in the SCRIPT's
     __main__ block; the harness carries a private copy. The subprocess therefore
     asserts its own correctness, never the script's default.
 (b) NON-DISCRIMINATING AST ASSERT (shape #1/#2). `assert "dry_run" in kw and
     "execute" in kw["dry_run"]` reads `ast.dump` of the keyword value. Measured:
       `dry_run=not args.execute` -> UnaryOp(op=Not(), operand=Attribute(...,
          attr='execute', ...))   -> "execute" in dump = True
       `dry_run=args.execute`     -> Attribute(..., attr='execute', ...)
                                  -> "execute" in dump = True
     The substring cannot see the `Not()`. It DOES kill a full revert to
     `dry_run=args.dry_run` ("execute" not in dump = False) -- naming the kill
     mechanism precisely: the AST assert covers the REVERT shape only, not the
     INVERSION shape, and the inversion is the dangerous one.

This is sole coverage for criterion 6 (no other test exercises the default), so
per qa.md 4c this is a BLOCKING vacuity, not a WARN.

### FINDING 2 -- TWO MORE SURVIVING MUTANTS: the sweep branches are unfixtured

MUTANT M3: the no-step-id branch `no_sid_kept += 1; continue`
        -> `_move(p, MISC, dry_run)` before the continue (the sweep restored).
MUTANT M4: the unknown-step branch (currently keep+WARN)
        -> keep the WARN print AND `_move(p, MISC, dry_run)`.

SUITE: **BOTH SURVIVED** -- 19 passed, 19 passed.
BEHAVIOURAL DIFFERENTIAL (hermetic fake repo, `--execute`):
  CONTROL: current/ = [INCIDENT_2026-08-14_credential_exposure.md,
      day_report_2026-08-17.md, research_brief_77.9.md]; archive/misc/ = []
  M3     : archive/misc/ = [INCIDENT_..., day_report_...]  <- the 664-file class
  M4     : archive/misc/ = [research_brief_77.9.md] while "[warn] KEEP" STILL
      PRINTS -- i.e. M4 restores verbatim the defect experiment_results C1 claims
      to have fixed ("the summary contradicted the action; now it is true").
Neither mutant is equivalent.

ROOT CAUSE: no fixture in the suite puts an unresolvable-name file OR an
unknown-sid file into `current/`. `_mixed_tree` and `_referenced_tree` contain
only names that resolve to steps present in the fixture masterplan. live_check
section 8 records that `census_99.json` (sid 99, unknown) was deliberately
RENAMED to `census_99.4.json` to stop the unknown-step branch confounding M2 --
a correct fix for M2 that removed the only place that branch was exercised, and
nothing replaced it.

IMPACT ON CRITERION 1: the clause "unknown ids are left in place with a WARN
line" has NO guard that can fail. The shipped behaviour IS correct (verified by
my own CONTROL run above), but the criterion is unguarded.

### Mutation matrix (independent; control observed GREEN first in every cell)

| cell | mutation | control | result |
|---|---|---|---|
| M-INV | `dry_run=not args.execute` -> `dry_run=args.execute` | GREEN | **SURVIVED** |
| M3 | no-step-id branch keeps -> sweeps to misc/ | GREEN | **SURVIVED** |
| M4 | unknown-step branch keeps -> sweeps to misc/ | GREEN | **SURVIVED** |
| M5 | `is_archivable(status) or status=='pending'` | GREEN | KILLED (c2, c3-M1, c4) |
| M6 | `if name in protected` -> `and False` | GREEN | KILLED (c5, c7-M2) |
| H1 | hook: `rolling_declares_step ...` -> `if true` | GREEN | KILLED (c13) |
| H2 | hook: derived `${base}_${short_sid}.md` src disabled | GREEN | KILLED (c8/c11) |

H1/H2 confirm criteria 8/11/13 rest on REAL behavioural guards driving the REAL
hook -- these are not source scans. Author cells M1/M2 reproduce (they are the
M5/M6 shapes).

### FINDING 3 -- a "measured" number shipped into production source does not reproduce

experiment_results C5 and the docstring of `_masterplan_referenced_names`
(scripts/housekeeping/backfill_handoff_archive.py) both state:
  "Measured 2026-08-17: 557 handoff-path references across 373 distinct paths,
   395 of them into handoff/current/."
Re-derived with the function's OWN rule (`re.compile(r"handoff/[A-Za-z0-9_./*-]+")`
over every `verification.command` + `verification.live_check`):
  total=577  distinct paths=386  into handoff/current/=415  distinct basenames=381
Not a drift artifact: `.claude/masterplan.json` is CLEAN vs HEAD and its mtime is
2026-08-17T17:58:03 -- BEFORE GENERATE started (research 20:35, contract 20:38).
Replayed across the last 8 revisions of that file the triple is stable at
577/386/415 (576 on the two oldest). Four alternative operationalizations tried
(star-less, \w, extension-required, command-only, live_check-only, steps-only);
none yields 557/373/395.

### FINDING 4 -- internal contradiction: 165 vs 156 mismatched dirs

experiment_results line 107: "while all **165** mismatched dirs lack the marker."
Same document line 82-83: "**156 mismatched of 845 dirs**". Census re-run
reproduces 156 / precision 0.9936 / contestable 43 EXACTLY.
The 165 is the size of `set(re.findall(r"phase-[0-9]+(?:\.[0-9]+)*", stdout))`
over the census's `--list-wrong` PROSE, which is not the mismatch set:
 - 11 tokens are NOT mismatched dirs: phase-82.54, 62.6, 80.2, 10.5, 45.0,
   76.9.2, 62.2, 40.8 (these are the *declared* sids in the top-8 summary), plus
   the synthetic controls phase-99.7 / phase-99.8, plus phase-63.3 (a truncation).
 - 2 REAL mismatched dirs are MISSED: `phase-63.3-parked` and
   `phase-audit-2.10-4.14.20` -- the regex cannot express those names.
The substantive claim ("no guarded dir is mismatched") is TRUE -- I re-derived
`guarded & mismatch == set()` from `classify` directly. Only the LABEL/count is
wrong. Consequence for the guard: `test_c12_prevention_holds_for_every_guard_
created_directory` builds `wrong` from that prose regex instead of from
`classify` (which its sibling `test_c12_every_mismatched_directory_carries_a_
quarantine_marker` uses correctly), so it under-covers 2 directories. Today
conservative; not vacuous, but not the census either.

### Claims that DID reproduce (checked, cleared -- not findings)
- immutable cmd 19 passed EXIT=0; adjacent suites 77 passed.
- bare backfill dry run: done-moved=436 misc-moved=0 audit-moved=1 log-moved=2
  root-kept=1 ambiguous=8 no-step-id=58 unknown-step=8 -- EXACT. protected 19->20
  and open-step 148->151 differ by exactly the +4 files the tree grew (761->765),
  which the artifact stamps and warns about up front. current/ unchanged after.
- verifier: [info] 59, [warn] 9, FAIL 455, EXIT=1 -- EXACT.
- the 58/59 and 8/9 gap between the two readers is NOT drift: symmetric
  difference is exactly {research_brief_phase83.md} and {census_78.json}, both of
  which the backfill short-circuits into `protected` before classifying. Verified
  by computing both sets and diffing them, not by counting.
- resolver census 609/55/64 of 728 (stamped 18:39Z) -> now 614/55/64 of 733:
  +5 suffix, +5 total, consistent with the stated growth.
- 845 archive dirs, 24 PROVENANCE dirs, 156 MISATTRIBUTION_NOTICE files -- EXACT.
- hook line claims: suffix branch at :226-242 (`src="$CURRENT_DIR/${base}_${short_sid}.md"`
  at :227), legacy prefix glob at :276 -- EXACT.
- c9's `or base in hook_src` IS an OR-escape-hatch, but the first clause is
  genuinely satisfied (the joined-bases literal is present once in the hook), so
  the hatch is latent, not load-bearing today. NOTE only.
- `_masterplan_referenced_names` returning set() on an unreadable masterplan is
  NOT a live fail-open: `_step_statuses()` runs first and has no try/except, so
  the script raises before any move. Cleared.
- No `.claude/hooks/**` modification (git diff). Peer-session uncommitted work
  present exactly as disclosed and not attributable to this step.

### C. Criterion-by-criterion
 1 PARTIAL  resolver + status refusal MET (M5/M1 killed); "unknown ids left in
            place with a WARN line" UNGUARDED (M4 survives). Behaviour correct.
 2 MET      test_c2, both directions in one run; misc-moved=0 asserted.
 3 MET      M1 killed; fixture mutation asserted by test_c3_fixture_is_load_bearing.
 4 MET      convergence asserted (done-moved=0, byte-identical, no -v2 minted).
 5 MET      M6/M2 killed; discrimination half present (sibling still archives).
 6 NOT MET  sole guard cannot fail on the inversion (FINDING 1). Product correct.
 7 MET      M2/M6 killed.
 8 MET*     real hook driven hermetically; H2 kills. Deviation from "flipping a
            scratch step" DECLARED with a defensible reason (auto-commit git add
            -A over a peer session's work). Intent ("not by reading the glob")
            satisfied -- the real hook executed.
 9 MET      shared resolver asserted by AST in both readers; hook literal present;
            c8 drives the real hook. Weak dedicated test (NOTE).
10 MET      whole-tree checker exists, runs, has a recall gate + controls. The
            PROPERTY is currently false (156 dirs), which criterion 12's
            "or explicitly quarantined with a marker" contemplates.
11 MET      live_check_99.7.md -> live_check.md observed; H2 kills.
12 MET      156 markers, re-measured not inherited (156/845 vs the step text's
            129/747), idempotent on re-run; the marker test re-derives from
            classify() and carries an anti-vacuity `flagged > 0` guard.
13 MET      H1 kills; discriminates (foreign refused, declaring admitted).

### Research gate (harness audit item 1) -- CLEAN
brief envelope: brief_status=COMPLETE, gate_passed=true,
external_sources_read_in_full=18 (floor 5), urls_collected=72, snippet_only=54,
internal_files_inspected=9, recency_scan_performed=true, coverage.dry=true.
Independently re-derived: 72 distinct http(s) URLs in the brief -- matches
urls_collected exactly. Recency-scan section present at 2.4 (4 findings).
Contract cites the brief 3x.

### Repo hygiene at end of evaluation
HEAD unchanged during evaluation: 8f281795 (checked at start and end).
`grep -c MUTANT` = 0 in all four mutated-in-copy files; git status on
scripts/housekeeping/ and .claude/hooks/ shows only the two intended M's and the
two intended ??'s. Immutable command re-run at the END: EXIT=0. All mutation work
was done on copies under the session scratchpad; the repo was never written.

### Verdict returned: FAIL
Blocking: criterion 6 sole-coverage guard vacuity (M-INV survives the whole
suite while the mutant's bare run MOVES files); criterion 1's unknown-id/WARN
clause unguarded (M4 survives) and the no-step-id branch unguarded (M3 survives);
required lint gate exits 1 on this step's own new file.
NOTE FOR THE NEXT SPAWN: the PRODUCT CODE IS CORRECT in all three mutant cases --
I verified the shipped behaviour by hermetic control runs. The defects are in the
GUARDS and in two numeric CLAIMS, not in the shipped script.

COMPLETED: 2026-08-17T19:02:02Z
