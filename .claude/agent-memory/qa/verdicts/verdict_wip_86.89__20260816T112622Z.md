STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.89
WRITTEN: 2026-08-16T11:26:22Z

## Q/A cycle 1 for step 86.89 -- write-first record

### Prior-attempt evidence
- `qa_wip.py 86.89 --spawned-at 2026-08-16T11:26:22Z`: attempt_number=1,
  prior_attempts=0, attempt_number_status=ok, source_present=true,
  records_retained=1 (own record), prior_records=[].
- `verdict_history_86_21.py --step 86.89 --evidence-only`: status=no_rows_for_step,
  verdicts=(none). Ledger is hand-appended; absence is weak evidence.
- sequence: no prior verdicts recorded for 86.89. Consistent with cycle 1.

### A. Harness compliance -- CLEAN (5/5)
1. research_brief_86.89.md 13:18, tracked, envelope brief_status=COMPLETE,
   gate_passed=true, external_sources_read_in_full=22 (floor 5), urls_collected=45
   (floor 10), recency_scan_performed=true, audit_class=true with coverage.dry=true
   after 2 dry rounds over 8. Contract §1 cites run wf_abfa4db8-f13; contract §7
   cites the brief. Research < contract (13:22) < results (13:24) < live_check (13:25).
2. contract before generate: OK.
3. experiment_results present.
4. log-last: `grep -F "phase=86.89" handoff/harness_log.md` -> 0 rows; masterplan
   status = "pending". Not yet logged, not yet flipped.
5. no verdict-shopping: cycle 1, attempt_number 1, no prior verdict.

### B. Deterministic
- IMMUTABLE CMD `bash -c 'source .venv/bin/activate && python
  scripts/qa/verify_matrix_coverage_86_85.py'` -> **exit 0**.
  "RESULT: OK -- every enumerated guard is touched by at least one cell."
  guards 15, covered 15, uncovered 0, cell problems 0.
- ruff --select F821,F401,F811 over a DERIVED scope
  (`git show --name-only b0edad8e -- '*.py'` UNION `git diff --name-only HEAD -- '*.py'`
   = scripts/qa/verify_cell_vacuity_86_89.py, backend/api/sovereign_api.py; non-empty
   asserted first, xargs not an unquoted var) -> "All checks passed!" exit 0.
- ast.parse verify_cell_vacuity_86_89.py -> OK.
- No frontend/backend diff from this step. b0edad8e = 4 files, +648/-0, no prod code.
  The uncommitted backend/api/sovereign_api.py + 5 frontend files are mtime
  2026-08-14 -- two days older, PRE-EXISTING, not 86.89's.
- NEW checker reproduces as filed: 14 cells, demanding 9, VACUOUS 5
  ['M5','M6','M9','M11','M12'], ALL GREEN 7 passed 0 failed, exit 0; matrix sha
  159331457e... identical before and after.
- Independent re-derivation of the "no guard has >1 cell" claim: every guard's
  `cells:` line carries exactly one cell id. Reproduces.

### FINDINGS (all EXECUTED; controls run first)

**F-1 [BLOCKING for criterion 6] The shipped LICENCE sentence is FALSE on its own
run.** verify_cell_vacuity_86_89.py:184 prints unconditionally
  "DOES: 'every cell in this matrix demands at least one enumerated guard'."
in the SAME output that prints "VACUOUS : 5 ['M5','M6','M9','M11','M12']". No
baseline carve-out. experiment_results §6 and live_check §5 repeat the identical
sentence. Criterion 6 governs exactly this object. The two NOT-bullets (guard-set
completeness, population recall) are honest but neither is the missing carve-out.
FIX: "every cell OUTSIDE the acknowledged baseline demands at least one enumerated
guard; 5 baselined cells demand nothing."

**F-2 [WARN] The module docstring makes a false claim about its own behaviour --
MEASURED.** :33-34 "Read-only on the repo: the matrix is mutated IN MEMORY via a
temp copy". False in both halves; no tempfile import exists. Instrumented run_gate
to sample the on-disk file at every invocation:
    gate invocations sampled          : 15
    DISTINCT on-disk matrix sha256s   : 15
    samples equal to the pristine sha : 1
    on-disk byte sizes seen (min/max) : 11090 / 11763   (pristine 11763)
    final restore byte-identical      : True
The write is STRUCTURALLY REQUIRED: run_gate() shells a subprocess and
verify_matrix_coverage_86_85.py does `from mutation_matrix_86_85 import CELLS` off
sys.path, so an in-memory mutation would be invisible to it. The 86.85 gate really
is "ZERO REPO WRITES" (:79); the sentence appears to have been inherited into a
file where it is untrue. Restore is a `finally`, so SIGKILL / power loss / a
concurrent session leaves a repo file one cell short in a repo whose auto-commit
hook runs `git add -A`.

**F-3 [WARN] The cardinality floor catches DELETION but NOT NEUTERING. EXECUTED,
controls RED first.**
    B1c CONTROL phantom "M99" in KNOWN_VACUOUS        -> rc=1 KILLED via [5]
    B1  same state, [5] condition replaced by True    -> rc=0 SURVIVED, "ALL GREEN",
                                                         "ok [floor] 7 assertions ran"
    B2  M6 un-baselined, [4] condition replaced True  -> rc=0 SURVIVED, floor ok
    B2d M6 un-baselined, [4] block DELETED            -> rc=1 KILLED via floor (6<7)
`emitted = len(PASSED)+len(FAILURES)` counts assertions, not their bite. live_check
§4 frames the floor as the remedy for V2/V3 without stating it is deletion-only.
Direct answer to Main's judge_these C: NO, the floor is not sufficient.

**F-4 [WARN, material] The KNOWN_VACUOUS baseline CAN hide a real regression.
EXECUTED.** [4] and [5] are pure ID-SET operations; nothing binds a baselined id to
its CONTENT. Mutant F1: keep cell id "M6", swap its payload from the ordering
mutation (`return out` -> `return out[::-1]`) to a benign
`EXIT_OK = 0` -> `EXIT_OK = 0  # repurposed-benign`.
    rc=0 SURVIVED. Output byte-equivalent: demanding 9, VACUOUS 5 incl. M6,
    ALL GREEN 7 passed 0 failed. Matrix restored byte-identical: True.
Not abstract: M6 IS the ordering cell -- the 86.85 cycle-1 Q/A's QA-M1, the defect
that opened this series. Repurposed, the matrix stops mutating ordering entirely and
BOTH gates stay green (the 86.85 gate never demanded it; the 86.89 checker excuses it
by id). The per-member reasons in the KNOWN_VACUOUS comment (M6=ordering, M9=dedup
key, M11/M12=cycle fallback, M5=event-time) are a declaration nothing verifies --
which is the phrase criterion 3 uses. FIX: pin each baselined id to a fingerprint of
its `find` anchor.

**F-5 [Missing evidence for criterion 4] Per-member RED HOLDS -- and the author's
stated reason for not demonstrating it is FALSE. EXECUTED (faithful: cell absent ON
DISK, shipped checker unmodified).**
    cell M5  absent -> exit=1 RED  FAIL [5]
    cell M6  absent -> exit=1 RED  FAIL [5]
    cell M8  absent -> exit=1 RED  FAIL [0] CONTROL (gate: "a guard has no cell")
    cell M9  absent -> exit=1 RED  FAIL [5]
    cell M11 absent -> exit=1 RED  FAIL [5]
    cell M12 absent -> exit=1 RED  FAIL [5]
    matrix restored byte-identical: True
6 of 6. Main's judge_these B asserts "Under the NEW check, dropping a KNOWN-VACUOUS
cell does NOT turn it red -- that is the finding". That is true only of the drop
INTERNAL to the loop; under the criterion's own operationalization (remove the cell,
run the gate) the shipped mechanism goes RED on every member. The criterion is
therefore satisfiable and was argued away untested. Separately, [5]'s message for a
DELETED cell -- "['M6'] now demand a guard" -- is factually wrong: it is gone, not
fixed. Red for the right reason, diagnosed wrongly.
Also EXECUTED, per-member baseline sensitivity (A-series): removing each id from
KNOWN_VACUOUS one at a time -> rc=1 each, [4] naming exactly ['M5'] / ['M6'] /
['M9'] / ['M11'] / ['M12'].

**F-6 [WARN] "STANDING" does not reproduce.** `grep -rn verify_cell_vacuity` over the
repo returns only the three 86.89 handoff artifacts and the file's own docstring.
NOTHING invokes it -- no test, hook, CI, matrix or harness call. Contrast: the 86.85
gate is genuinely standing (`mutation_matrix_86_85.py:247` imports and calls
`cov.main()`), which is why it fires on every matrix run. experiment_results §1 calls
the new file "the **standing** cell-vacuity check" and contract P1 says "make the
drop-a-cell probe STANDING"; the research rationale quoted is "the one-off drop-a-cell
probe should be standing, not a comment". As shipped it is a manually-run script.

**F-7 [NOTE] Criterion 1's "verbatim output" carries no command.** contract §2,
experiment_results §2 and live_check §1 all show the reproduction OUTPUT with no
invocation line and no committed probe; the ad-hoc probe (rebuilt 3x, honestly
disclosed) was not kept. I re-derived the figure by an independent route (F-5: M8
absent -> old gate RED; M5/M6/M9/M11/M12 absent -> old gate GREEN = 1 of 5 / 1 of 4),
so the NUMBER is sound, but the artifact is not regenerable by a reader.

**F-8 [NOTE] Cross-reference error.** experiment_results:167 cites "§9 of
live_check_86.89.md". live_check has §1-§5; the mutation matrix is §4.

**F-9 [NOTE] Author claims verified, not disputed.** V1 (KNOWN_VACUOUS = every cell)
-> rc=1 killed via [5], mechanism correctly credited. V4 (cell parser matches
nothing) -> rc=1 killed via [0]. Assertion [1] is NOT redundant: it has a genuine
unique kill (gate forced always-green AND every cell baselined -> only [1] fires).
My initial suspicion that [1] was subsumed by [4] was WRONG and is withdrawn.

**F-10 [NOTE] Criterion 5's named shape is untested.** The check scores each cell on
one bit (`rc2 != 0`) and cannot say WHICH guard a "demanding" cell demands, so the
ast.Try shape the criterion names -- a cell credited with coverage it does not have --
is invisible to it per-cell. The GLOBAL version is caught by the [0] control
(verified: M8 absent -> control FAILED, "every cell result below would be
unscorable"). Undisclosed bound.

**F-11 [NOTE] "every new guard" is not literally covered.** 8 assertion sites exist
([0]x2, [1], [2], [3], [4], [5], [floor]); the matrix's 5 cells reach [5], [1],
[0]-parser, [3] and the baseline. [2] "every cell is scorable" has no cell.

### CRITERION MAP
C1 MET (reproduced, ordering correct, independently re-derived) -- F-7 NOTE on the
   missing command.
C2 MET as measured-and-stated (4 of 4 classified on members 1-4, 0 on member 5, both
   labelled Recall_SD). Undisclosed: the shipped exit code is 0 on all four.
C3 PARTIALLY MET. Cell list IS a derivation (V4 proves the parser fails closed). But
   the exit code is gated on the KNOWN_VACUOUS DECLARATION. Set-level verification is
   real and bites (A-series 5/5, B1c). Content-level is unverified -- F-4.
C4 NOT MET IN THE ARTIFACTS (Missing_Assumption). The property HOLDS -- I measured
   6/6 RED -- but no covering evidence exists in experiment_results or live_check, and
   the masterplan live_check field explicitly requires "the per-member RED
   demonstration". Argued away on a premise F-5 refutes.
C5 PARTIALLY MET. V1 + [1] + [3] are real and reproduce; the ast.Try shape the
   criterion names is untested and the bound is undisclosed (F-10).
C6 NOT MET. F-1: the licence sentence states a completeness claim its own run
   contradicts, in exactly the object the criterion governs.
C7 MET. No verdict surface in the new file (grep clean); commit adds 1 script + 3 md;
   immutable command still exit 0; no masterplan / qa-verdict.js edit.
C8 PARTIALLY MET. Control GREEN first: yes, reproduced. Byte-identical restore: yes,
   verified on every run. But the matrix covers DELETION only (F-3), and [2] has no
   cell (F-11).

### Code-review heuristics
No security / trading-domain / money-path surface (no production code). #17
illusory-guard fires at WARN on the floor (deletion-only) -- genuine behavioural
guards [4]/[5] coexist and bite, so WARN not BLOCK. #13 sycophantic-all-pass: N/A.

### Worst-of-lenses
correctness CONDITIONAL (mechanism right; licence sentence wrong) /
does-it-reproduce CONDITIONAL (all reproduces except F-6 "standing" and F-7's absent
command) / scope-honesty CONDITIONAL (strong disclosure overall; three undisclosed
bounds: F-3, F-4, C2-exit-code). min = CONDITIONAL.

COMPLETED: 2026-08-16T11:39:32Z
(An earlier save of this line carried an INVENTED time. Corrected against
`date -u +%Y-%m-%dT%H:%M:%SZ` read at that instant. Final integrity check at the
same moment: mutation_matrix_86_85.py 159331457e..., verify_matrix_coverage_86_85.py
8de8a89744..., verify_cell_vacuity_86_89.py edb37d34c4... -- all byte-identical to
their pre-mutation state; `git status --short scripts/qa/` empty.)
