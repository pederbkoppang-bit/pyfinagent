STATUS: INCOMPLETE -- not a verdict
STEP: 90.1
WRITTEN: 2026-08-20T20:10:11Z

# Q/A write-first record -- step 90.1, CYCLE 3 (Main says attempt 4 of 5)

Spawn context (from Main, ADVISORY only):
- Commit b5a9b9d6 (cycle-1 3bf0b0fe, cycle-2 1fc7b2e6). DO NOT grade 1c2f25b3.
- Changed since cycle 2: scripts/qa/mutation_matrix_90_1.py, scripts/harness/attempt_gate.py
- Main claims 3 cycle-2 WARNs fixed: (1) criterion_5_clause3 smoke-import ERROR scoring,
  (2) criteria-erosion / containment tautology replaced, (3) illusory-guard M11 decoy moved to 900s.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command exit code + git status/diff + lint + smoke
C. LLM judgment vs 6 immutable criteria + adversarial attacks Main invited

## Findings log (appended as established)

### A. HARNESS COMPLIANCE (5 items) -- CLEAN so far
- research gate: handoff/current/research_brief_90.1.md envelope brief_status=COMPLETE,
  external_sources_read_in_full=10, urls_collected=25, recency_scan_performed=true, gate_passed=true. OK (>=5, >=10).
- contract-before-generate mtimes: research 21:12:47 < contract 21:15:34 < attempt_gate.py 22:07:24
  < mutation_matrix 22:08:14 < experiment_results 22:09:37. ORDER OK.
- experiment_results_90.1.md present (28144 b), evaluator_critique_90.1.md present (33954 b), live_check present.
- log-last: `grep -cF 'phase=90.1' handoff/harness_log.md` = 0, exit 1. masterplan status still "pending". OK.
- no-verdict-shopping: cycle-3 commit b5a9b9d6 changed scripts/harness/attempt_gate.py (+36) and
  scripts/qa/mutation_matrix_90_1.py (+61). EVIDENCE CHANGED. OK.
- IMMUTABILITY: criteria sha + command sha byte-identical at 3bf0b0fe / 1fc7b2e6 / b5a9b9d6 / HEAD
  (crit_sha=f98626019b331382, cmd_sha=e5ae167bc5ca36c5). b5a9b9d6 does NOT touch .claude/masterplan.json.

### B. DETERMINISTIC
- IMMUTABLE COMMAND: `python3 scripts/harness/attempt_gate.py --self-test` -> SELF-TEST PASSED, exit 0.
  `python3 scripts/qa/mutation_matrix_90_1.py --verify` -> exit 0.
  KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True.
- ledger sha256 BEFORE and AFTER the whole run: fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
  == fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2 (c6 re-derived independently).
- real tree md5 before==after True (matrix self-reported AND re-derived by me on every custom cell).
- git status --short: only handoff/audit/*.jsonl (gate's own append-only audit) + my WIP file. No unintended prod change.

### C. ADVERSARIAL -- criterion 5 clause 3 ("a mutant that fails to run scores ERROR")
FINDING QX (REPRODUCED 3x, both subject modules): the cycle-3 smoke-import probe closes the
IMPORT-time seam only. A mutant that PARSES, IMPORTS CLEANLY, and then CANNOT RUN still scores KILLED.
  QX1  attempt_gate.py   `return handle_hook()` -> deferred `import nonexistent_module_zzz_qa` in main()  => KILLED
  QX2  attempt_gate.py   `return handle_hook()` -> `return handle_hook_v2()` (NameError at call time)     => KILLED
  QX3  attempt_outcomes.py  `def main(...)` + deferred missing import                                      => KILLED
All three: anchor count 1, ast.parse OK, smoke-import exit 0, real tree md5 unchanged.
QX2 is the REALISTIC authoring error (a repl naming a helper that does not exist).
This is the cycle-2 defect relocated ONE SEAM later: parse -> import -> RUN.

Failure-count is NOT an incidental discriminator: QX3 fails 5 of 25 checks -- EXACTLY the
same count as shipped cell M3 (a genuine kill). Shipped cells fail 1-5 of 25; QX1/QX2 fail 9.
So nothing in the current scoring separates apparatus failure from a real kill.

THE GAP IS CLOSABLE -- verified discriminator (so the criterion is achievable):
on the hook drive, QX1/QX2 exit rc=1 with an unhandled `Traceback` on stderr, while
ALL 16 shipped cells (N0..M14) exit rc=0 with EMPTY stderr. QX3 likewise: the resolver's
no-arg invocation gives rc=1 + Traceback vs rc=2 + no traceback for every shipped cell.
An "unhandled traceback in a drive => ERROR" rule scores QX1/QX2/QX3 ERROR and leaves all
15 kills untouched. (NOTE: a benign-path smoke such as `--status 9.1` does NOT work --
QX1/QX2 return rc=0 there because the mutation sits on the hook branch only. Measured.)

### C2. M11 tolerance band -- Main's ATTACK claim #3 is FALSIFIED BY EXECUTION
Main asserts: "Tolerances 1-899 survive, and that is correct rather than a residual gap:
they sit below the documented ambiguity threshold, so there is no ambiguity for the guard
to catch."
Independent sweep reproduces his matrix result exactly (0 KILLED; 1/60/300/899 SURVIVED;
900/901/1800/3600/86400 KILLED).
But on the REAL corpus AS OF HIS OWN COMMIT b5a9b9d6 (98 rows), re-resolved from scratch:
  tol=30  ambiguous_match=0   {C:46 F:11 NV:21 P:11 U:5}
  tol=300 ambiguous_match=0   (identical -- so <=300 IS an equivalent mutant; that half of
                               his claim is CORRECT)
  tol=386 ambiguous_match=1   {C:45 ... U:6}      <-- ambiguity FIRST APPEARS AT 386s
  tol=500 ambiguous_match=2
  tol=899 ambiguous_match=6   {C:43 F:10 NV:20 P:10 U:11}  5 graded outcomes LOST
On the CURRENT ledger (95 attempt rows) the same binary search gives first divergence at
tolerance 386 (last matching 385), and summed total_tokens:
  tol=30 20,806,242 | tol=386 20,528,843 (-1.3%) | tol=899 19,439,217 (-6.6%)
  | tol=3600 3,637,821 | tol=86400 1,655,453
So the band [386,899] IS real, measurable damage that M11 does not catch. The claim was
FALSE WHEN MADE, not merely drifted.
ROOT CAUSE: attempt_outcomes.py:35 docstring says "Ambiguity first appears at 900s",
measured over "the real 89 attempt rows". That number was ALREADY stale at b5a9b9d6, and
Main borrowed it to certify the survivor band as equivalent.
CLASSIFICATION: criterion 5 names only three mutants (M1, M2, the ERROR clause); M11 is
not one of them. Criterion 1's "UNKNOWN only where no run record exists" HOLDS at the
shipped default (ambiguous_match=0 at tol=30). So this is a RESIDUAL guard-quality +
scope-honesty finding, NOT a numbered-criterion miss.

### C3. Prose defect inside the changed file
scripts/qa/mutation_matrix_90_1.py drive_join comment says "Moved to 950s" while the code
plants `900_000` ms = 900 s (offset_ms units confirmed at _plant_run_record). The comment's
own stated property ("ambiguous for anything at or past the documented threshold") is only
true at 900s; at 950s tol=900 would SURVIVE. Leftover from the interim state Main discloses
in experiment_results W3. Code right, comment wrong. NOTE-level.


